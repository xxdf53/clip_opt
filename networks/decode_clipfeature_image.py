"""
================================================================================
  decode_clipfeature_image.py — CLIP 图像特征 → 自然语言文本解码
================================================================================
  来源：基于 CLIPCap (rmokady/CLIP_prefix_caption) 框架改造。
  用途：将 C2P-CLIP 检测模型提取的图像特征（经检测头加权后）解码为自然语言，
        实现"模型解释"——告诉你模型在图像中"看到了什么"。

  整体数据流：
    image_path
    ↓  get_image_features()
    image_features  (CLIP ViT-L/14 视觉编码, shape: [1, 768])
    ↓  get_text()
        ├── 通过检测 fc 层计算 Fake/Real 概率
        ├── [可选] 用 fc 权重缩放特征 (cal_detection_feat)
        ├── L2 归一化 → MLP 映射 → prefix embeddings
        └── GPT-2 自回归生成文本
    generated text (如 "a cat sitting on a chair")

  模块对照：
    MLP               — 通用多层感知机
    ClipCaptionModel  — CLIP prefix → GPT-2 的桥梁模型 (核心来自 CLIPCap)
    generate2()       — top-p (nucleus) 采样自回归文本生成
    parse_args()      — CLI 参数解析
    get_text()        — 特征 → 文本的核心管线（含检测加权）
    get_clip_model()  — 加载 CLIP (仅视觉塔，删除文本塔以节省显存)
    get_clipcap_model() — 加载 ClipCaptionModel + GPT-2 tokenizer
    get_image_features() — 图像 → CLIP 视觉特征
    __main__          — 编排入口
================================================================================
"""

# 环境提示：建议用 conda 创建独立环境
# conda -n create clip-text-decoder python=3.8.5
import argparse
import os
from torch import nn
import numpy as np
import torch  # 2.3.1+cu118
# conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # 4.25.0
import skimage.io as io           # 读取各种格式图像
import PIL.Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import time
import warnings

warnings.filterwarnings('ignore')

# 类型别名 — 提升代码可读性
N = type(None)    # NoneType
T = torch.Tensor
D = torch.device


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     类 MLP — 通用多层感知机                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class MLP(nn.Module):
    """
    [类] 简单的多层感知机 (Multi-Layer Perceptron)。

    结构: Linear → Tanh → Linear → Tanh → ... → Linear
    最后一层无激活函数（用于回归/投影）。

    参数:
      sizes  — 每层的维度列表，例如 (768, 3840, 7680)
      bias   — 线性层是否使用偏置
      act    — 激活函数类（默认 nn.Tanh）
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: 直接通过 Sequential 容器"""
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            # 添加线性层
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            # 除最后一层外，每层后加激活函数
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              类 ClipCaptionModel — CLIP → GPT-2 映射模型                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class ClipCaptionModel(nn.Module):
    """
    [类] CLIPCap 核心模型：将 CLIP 图像特征作为 prefix 注入 GPT-2 生成文本。

    工作流程:
      CLIP image feature (768-d)
        ↓  clip_project (MLP)
      prefix embeddings  (prefix_length × gpt_embedding_size)
        ↓  与文本 token embeddings 拼接
      GPT-2  →  自回归生成 caption

    架构来源: Mokady et al., "ClipCap: CLIP Prefix for Image Captioning"
    """

    # FIXME 注释：原本打算用 functools.lru_cache 缓存零 token，
    # 但现未启用（每次调用都创建新张量）
    # @functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        """
        [方法] 生成全零的虚拟 token，用作 prefix 部分的 label 占位符。

        注意: GPT-2 中 label=-100 才会在损失中被忽略，
        这里填充的是 0（代表实际 token），所以训练时 prefix 部分
        的损失并不会被正确忽略——这是一个潜在问题。
        
        返回: shape [batch_size, prefix_length] 的全零 LongTensor
        """
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
        self,
        tokens: T,                          # 文本 token IDs, shape [B, L]
        prefix: T,                          # CLIP 图像特征, shape [B, 768]
        mask: Optional[T] = None,           # attention mask
        labels: Optional[T] = None,         # 训练时的标签（自回归：输入右移一位）
    ):
        """
        [方法] 前向传播: CLIP prefix + 文本 tokens → GPT-2 → logits/loss。

        步骤:
          1. 文本 tokens → GPT-2 word embedding (wte)
          2. CLIP prefix → MLP → reshape 为 [B, prefix_len, gpt_emb_dim]
          3. 拼接: [prefix_projections; embedding_text] 沿 dim=1
          4. 若提供 labels → 为 prefix 部分填充虚拟 label
          5. GPT-2 前向 → 返回 (logits, loss)
        """
        # 文本 token → GPT-2 嵌入
        embedding_text = self.gpt.transformer.wte(tokens)
        # CLIP 特征 → MLP → prefix embeddings
        # reshape: [B, 768*prefix_len] → [B, prefix_len, gpt_embedding_size]
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        # 拼接 prefix 和文本嵌入：[B, prefix_len + text_len, embed_dim]
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            # 为 prefix 部分创建虚拟标签（全零占位）
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        # GPT-2 前向，返回的 out 包含 .logits 和 .loss（若 labels 非空）
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        """
        [构造] 初始化 CLIP → GPT-2 映射模型。

        参数:
          prefix_length  — CLIP 特征映射为多少个 prefix token（默认 10）
          prefix_size    — CLIP 特征维度（768 for ViT-L/14）

        子模块:
          self.gpt            — GPT-2 (base, 124M 参数)
          self.clip_project   — MLP: 768 → 3840 → 7680
            (7680 = gpt_embedding_size(768) × prefix_length(10))
            中间层 3840 是一半的输出维度，作为 bottleneck
        """
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        # 加载预训练 GPT-2 (约 124M 参数)
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        # GPT-2 的嵌入维度 = 768 (对于 gpt2-base)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        # MLP: 768 → ((768*10)//2) = 3840 → 7680
        # 中间层维度 = 一半的最终输出维度，作为 bottleneck
        self.clip_project = MLP(
            (
                prefix_size,                                     # 768  (输入)
                (self.gpt_embedding_size * prefix_length) // 2,  # 3840 (中间)
                self.gpt_embedding_size * prefix_length,          # 7680 (输出)
            )
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║           函数 generate2() — top-p (nucleus) 采样文本生成                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def generate2(
    model,                    # ClipCaptionModel（但只用其内部的 self.gpt）
    tokenizer,                # GPT-2 tokenizer
    tokens=None,              # 已有的 token IDs（prompt 模式）
    prompt=None,              # 文本 prompt 字符串（未使用，被 tokens/embed 覆盖）
    embed=None,               # 预计算的 prefix embeddings（CLIP 特征经 MLP 后的结果）
    entry_count=1,            # 生成多少个候选（只取第 0 个）
    entry_length=67,          # 最大生成长度（token 数）
    top_p=0.8,               # nucleus 采样的累积概率阈值
    temperature=1.0,          # softmax 温度（1.0 = 不变）
    stop_token: str = ".",   # 遇到句号停止生成
):
    """
    [函数] 使用 top-p (nucleus) 采样从 GPT-2 自回归生成文本。

    两种启动模式:
      a) embed 模式: 给定 CLIP prefix embeddings → 直接作为初始 hidden state
      b) prompt 模式: 给定文本 prompt → 编码后逐步生成

    采样算法 (每个 token):
      1. GPT-2 前向 → logits
      2. 温度缩放: logits /= temperature
      3. top-p 过滤: 按概率从高到低累加，超过 p 的 token 置为 -inf
      4. argmax 取最优 token（实际等价于 top-p 过滤后贪心解码）
      5. 该 token 的 embedding 拼接到 generated 序列尾
      6. 若遇到 stop_token → 提前终止

    注意: 虽然有 entry_count 参数支持多条生成，但只返回 generated_list[0]。
    """
    model.eval()
    generated_num = 0        # 未使用，保留可能是扩展预留
    generated_list = []
    # 编码停止符（默认 "."）
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        for entry_idx in range(entry_count):
            # ── 初始化 generated（GPT-2 的输入 embeddings） ──
            if embed is not None:
                # embed 模式: CLIP prefix embeddings 作为起点
                # shape: [1, prefix_len, gpt_emb_dim] = [1, 10, 768]
                generated = embed
            else:
                # prompt 模式: 将文本 token 转换为 embeddings
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)

            # ── 自回归生成循环 ──
            for i in range(entry_length):
                # GPT-2 前向
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits           # [B, seq_len, vocab_size]
                # 只取最后一个位置的 logits，并施加温度缩放
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                # ── top-p (nucleus) 过滤 ──
                # 按概率降序排序
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # 计算 softmax 后的累积概率
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                # 标记累积概率超过 top_p 的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保证至少保留第一个 token（概率最高的那个）
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                # 将被移除的 token 的 logit 置为 -inf
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                # argmax 选择——在 top-p 过滤后的集合中做贪心解码
                next_token = torch.argmax(logits, -1).unsqueeze(0)   # [1, 1]
                # token → embedding（用于下一次前向）
                next_token_embed = model.gpt.transformer.wte(next_token)

                # 累积 token IDs（用于最终 decode）
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                # 累积 embeddings（用于下一次前向）
                generated = torch.cat((generated, next_token_embed), dim=1)

                # 遇到停止符 → 提前终止
                if stop_token_index == next_token.item():
                    break

            # 将 token IDs 解码为字符串
            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    函数 parse_args() — CLI 参数解析                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def parse_args():
    """
    [函数] 解析命令行参数并打印配置摘要。

    参数说明:
      --prefix_length       CLIP 特征映射为的 prefix token 数量 (默认 10)
      --model_path          ClipCaptionModel 预训练权重路径/URL
      --image_path          待分析的输入图像路径 [必需]
      --fc_path             C2P-CLIP 检测头的 fc 权重路径/URL
      --cal_detection_feat  是否用检测 fc 层缩放 CLIP 特征后再解码
      --device              运行设备 (cuda:N 或 cpu)
    """
    parser = argparse.ArgumentParser(description='decode detection feature to text')
    parser.add_argument('--prefix_length',      type=int,  default=10,
                        help='CLIP 特征映射为多少个 prefix token')
    parser.add_argument('--model_path',         type=str,  default='https://www.now61.com/f/Xljmi0/coco_prefix_latest.pt',
                        help='ClipCaptionModel 预训练权重 (URL 或本地路径)')
    parser.add_argument('--image_path',         type=str,  default='',
                        help='输入图像路径', required=True)
    parser.add_argument('--fc_path',            type=str,  default='https://www.now61.com/f/qwvoH5/fc_parameters.pth',
                        help='C2P-CLIP 检测 fc 层权重 (URL 或本地路径)')
    parser.add_argument('--cal_detection_feat', action="store_true",
                        help='是否用检测 fc 层缩放 CLIP 特征 (增强 Fake/Real 方向信号)')
    parser.add_argument('--device',             type=str,  default='cuda:0',
                        help='运行设备: cuda:n 或 cpu')
    args = parser.parse_args()

    # ── 内嵌函数: 打印参数摘要（含与默认值的对比） ──
    def print_options(parser, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    print_options(parser, args)
    return args


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   函数 get_text() — 核心管线: CLIP 特征 → 检测加权 → GPT-2 文本             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_text(
    image_features,          # CLIP 图像特征, shape [1, 768]
    tokenizer,               # GPT-2 tokenizer
    model,                   # ClipCaptionModel
    fc_path,                 # 检测头 fc 权重路径
    cal_detection_feat=True, # 是否用检测权重缩放特征
    prefix_length=10,        # prefix token 数量
    device='cpu',
):
    """
    [函数] 核心管线: CLIP 图像特征 → 文本解码。

    这是整个脚本最关键的函数，连接了"检测"和"解释"两个阶段:

    阶段 1 — 检测:
      image_features (768-d) → fc 层 (768×1) → sigmoid → Fake/Real 概率

    阶段 2 — 特征重标定 (仅当 cal_detection_feat=True):
      image_features = image_features * weight + bias
      → 沿着"真假判别方向"缩放原始 CLIP 特征
      → 这样解码出的文本倾向于反映模型做出判断的依据
      → 例如: weight = [0.2, -0.5, 0.1, ...]  (768维)
        每个维度的值是该维度对分类的贡献大小
        image_features * weight 放大关键维度，削弱无关维度

    阶段 3 — 文本生成:
      重标定特征 → L2 归一化 → MLP → prefix embeddings → GPT-2 生成文本

    返回: 生成的自然语言文本（如 "a dog running on a beach."）
    """
    # 加载检测头 fc 参数 (支持 URL 或本地路径)
    mod = torch.hub.load_state_dict_from_url(fc_path, map_location="cpu", progress=True) \
        if fc_path.startswith("http") \
        else torch.load(fc_path, map_location="cpu")
    # 提取 fc 层的 weight (768×1) 和 bias
    weight, bias = mod['fc.weight'].to(device), mod['fc.bias'].to(device)

    with torch.no_grad():
        # ── 阶段 1: 计算 Fake/Real 概率 ──
        prob = nnf.linear(image_features, weight, bias).sigmoid().cpu().numpy()[0][0]
        dict_prob = {False: 'Fake', True: 'Real'}
        # 调试输出（当前被注释）:
        # print(f'\nPredicted prob: {prob}, {dict_prob[prob<0.5]}')

        # ── 阶段 2: 沿检测方向缩放特征 ──
        # torch.mul: element-wise 乘法，每个维度独立乘以对应的 weight 值
        # + bias: 加上偏移量
        # 效果: 放大对分类判别贡献大的维度，削弱贡献小的维度
        if cal_detection_feat:
            image_features = torch.mul(image_features, weight) + bias

        # ── 阶段 3: 归一化 → MLP → GPT-2 ──
        # L2 归一化到单位球面
        image_features /= image_features.norm(2, dim=-1, keepdim=True)
        # MLP: 768 → 7680 (= prefix_length × gpt_emb_dim)
        # reshape: [1, 7680] → [1, prefix_length, gpt_emb_dim]
        prefix_embed = model.clip_project(image_features).reshape(1, prefix_length, -1)
        # 自回归生成文本 (embed 模式)
        generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

    return generated_text_prefix


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║       函数 get_clip_model() — 加载 CLIP (仅保留视觉塔)                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_clip_model(clip_name='openai/clip-vit-large-patch14', device='cpu'):
    """
    [函数] 加载 CLIP 模型，仅保留视觉编码器。

    删除的模块及其原因:
      text_model        — 不需要文本编码
      text_projection   — 不需要文本投影
      logit_scale       — 不需要图文匹配温度参数

    保留:
      vision_model      — ViT-L/14 图像编码器
      visual_projection — 视觉特征投影 (→ 768-d)

    这样节省约 40% 显存。

    返回: (clipmodel, processor)
      clipmodel  — 精简后的 CLIPModel (只剩视觉塔)
      processor  — CLIPProcessor (图像预处理: resize + normalize)
    """
    clipmodel = CLIPModel.from_pretrained(clip_name)
    processor = CLIPProcessor.from_pretrained(clip_name)
    # 删除文本侧模块，释放显存
    del clipmodel.text_model
    del clipmodel.text_projection
    del clipmodel.logit_scale
    clipmodel = clipmodel.to(device)
    return clipmodel, processor


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║    函数 get_clipcap_model() — 加载 ClipCaptionModel + GPT-2 tokenizer      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_clipcap_model(model_path, prefix_length=10, device='cpu'):
    """
    [函数] 加载 ClipCaptionModel 及其 GPT-2 tokenizer。

    步骤:
      1. 加载 GPT-2 tokenizer
      2. 实例化 ClipCaptionModel(prefix_length=10, prefix_size=768)
      3. 从 URL 或本地路径加载预训练权重
      4. 验证权重 keys 完全匹配 (完整性校验)
      5. 切换到 eval 模式并移至目标设备

    参数:
      model_path — 预训练权重 URL 或本地 .pt 文件路径

    返回: (model, tokenizer)
    """
    # GPT-2 分词器
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # 实例化 CLIPCap 模型: CLIP feature (768-d) → MLP → GPT-2 prefix
    model = ClipCaptionModel(prefix_length, prefix_size=768)
    # 加载预训练权重 (支持 URL 或本地路径)
    pretrained = torch.hub.load_state_dict_from_url(model_path, map_location="cpu", progress=True) \
        if model_path.startswith("http") \
        else torch.load(model_path, map_location="cpu")
    model.load_state_dict(pretrained)
    # 完整性校验: 确认权重字典的键完全对应
    assert pretrained.keys() == model.state_dict().keys(), \
        "预训练权重 keys 与模型 state_dict keys 不匹配"
    model = model.eval()
    model = model.to(device)
    return model, tokenizer


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   函数 get_image_features() — 图像 → CLIP 视觉特征                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_image_features(image_path, clipmodel, processor, device='cpu'):
    """
    [函数] 从图像文件中提取 CLIP 视觉特征。

    步骤:
      1. skimage.io.imread → numpy array (H, W, C)
      2. PIL.Image.fromarray  → PIL Image
      3. CLIPProcessor → resize (224×224) + normalize (ImageNet std/mean) + to tensor
      4. pixel_values 移到 GPU
      5. clipmodel.get_image_features → ViT forward → CLS token → visual_projection
         → 768-d 归一化特征向量

    参数:
      image_path — 图像文件路径
      clipmodel  — 精简后的 CLIPModel (只有视觉塔)
      processor  — CLIPProcessor

    返回: image_features — shape [1, 768] 的归一化视觉特征
    """
    with torch.no_grad():
        # 读取图像: 支持各种格式 (png, jpg, webp, bmp, …)
        image = PIL.Image.fromarray(io.imread(image_path))
        # CLIP 预处理: 缩放到 224×224 + 归一化
        inputs = processor(images=image, return_tensors="pt")
        # 移至 GPU
        inputs['pixel_values'] = inputs['pixel_values'].to(device)
        # 提取特征 (内部: ViT forward → CLS token → visual_projection)
        image_features = clipmodel.get_image_features(**inputs)
    return image_features


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          主 入 口                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    """
    主流程 (5 步管线):

      opt = parse_args()                           # 1. 解析命令行参数
      clipmodel, processor = get_clip_model()       # 2. 加载 CLIP 视觉编码器
      model, tokenizer    = get_clipcap_model()     # 3. 加载 CLIPCap 模型
      image_features = get_image_features()          # 4. 提取 CLIP 视觉特征
      text = get_text(image_features, ...)           # 5. 检测加权 + 解码为文本
      print(text)                                    # 6. 输出结果

    示例用法:
      python decode_clipfeature_image.py \\
        --image_path ./assets/DALLE/DALLE_2_Cowboy_In_Swamp_Close_Up_Outpaint_1.png \\
        --cal_detection_feat
    """
    opt = parse_args()
    device = torch.device(opt.device)

    # 步骤 1: 加载 CLIP 视觉编码器（删除文本塔以节省显存）
    clipmodel, processor = get_clip_model(
        clip_name='openai/clip-vit-large-patch14', device=device)

    # 步骤 2: 加载 CLIP → GPT-2 映射模型
    model, tokenizer = get_clipcap_model(opt.model_path, device=device)

    # 步骤 3: 图像 → CLIP 特征
    image_features = get_image_features(
        opt.image_path, clipmodel, processor, device=device)

    # 步骤 4: CLIP 特征 → 检测加权 → GPT-2 文本解码
    text = get_text(
        image_features, tokenizer, model,
        opt.fc_path, opt.cal_detection_feat, device=device)

    # 步骤 5: 输出生成的自然语言描述
    print(text)
