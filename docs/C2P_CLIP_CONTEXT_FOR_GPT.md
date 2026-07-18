# C2P-CLIP 项目可迁移上下文

> 用途：将本文件完整提供给另一个 GPT 账号，使其能够继续本项目的工程调试、实验分析和创新方案研究。
>
> 更新时间：2026-07-18（Asia/Shanghai）

## 1. 项目身份与目标

- 项目目录：`D:/github-ware/C2P-CLIP-DeepfakeDetection`
- 项目：C2P-CLIP deepfake / AI-generated image detection。
- 原论文：C2P-CLIP，论文题目为 *C2p-clip: Injecting category common prompt in clip to enhance generalization in deepfake detection*。
- 基础模型：本地 Hugging Face `openai/clip-vit-large-patch14`，服务器必须使用本地模型目录，不能依赖联网下载。
- 当前研究目标：在复现 C2P-CLIP 的基础上，评估中间层 patch/local feature 是否能提升跨生成器泛化，并寻找有文献依据、可验证的创新方向。

## 2. 必须遵守的约束

1. 服务器是 Linux，已有完整虚拟环境；不要假设本地 Windows 环境可执行训练。
2. 推理阶段只输入图像，不依赖 Caption、文本提示或额外外部模型。
3. 保留用户已有改动，不执行 `git reset --hard`、`git checkout --` 等破坏性操作。
4. 不把数据集、模型、缓存、生成图片、`__pycache__` 或论文输出提交到 GitHub。
5. 尚未通过对照实验确认的研究想法不要直接全部实现，应先做公平消融和小规模验证。
6. 完整 CNN_synth 测试集预期为 13 个生成器、90,329 张图；若只看到 12 个生成器，需要检查 `seeingdark`。

## 3. 当前代码状态

### 3.1 已完成并已推送

当前 `main` 已推送到 GitHub：

```text
commit: b38c0013b2632ad2530ff05c745abc7104ff0629
message: feat: test official model on nested binary datasets
remote: origin/main
```

本次提交包含：

- `scripts/test_airplane_official.py`
- `utils/binary_dataset_layout.py`
- `utils/binary_metrics.py`
- `tests/test_binary_dataset_layout.py`
- `tests/test_binary_metrics.py`
- `README.md`
- `docs/superpowers/specs/2026-07-17-official-cnn-synth-test-design.md`
- `docs/superpowers/plans/2026-07-17-official-cnn-synth-test.md`

当前未跟踪但不应提交的用户文件包括：

```text
CNN_synth_testset/
paper_rewriting_output/
```

### 3.2 官方模型测试脚本

`scripts/test_airplane_official.py` 支持：

- 官方原始 `state_dict` checkpoint，不是 `train.py` 的 `{model, total_steps}` LoRA checkpoint。
- 直接二分类目录和递归嵌套语义类别目录。
- 自动按顶层生成器分组，并对每组输出 ACC、Real ACC、Fake ACC、AP、数量。
- 输出 AUROC、ECE、Brier、raw-logit 类别统计、宏平均和所有图片合并后的总体指标。
- 纯图像 `ImageFolder` 推理，不使用 Caption/tokenizer。
- 本地 `--clip_path`，不需要 Hugging Face 联网。

服务器测试官方模型：

```bash
cd /home/ac/data/xxxf/clip_opt_github

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/test_airplane_official.py \
  --dataroot ./CNN_synth_testset \
  --model_path ./C2P_CLIP_release_20240901.pth \
  --clip_path ./clip-vit-large-patch14 \
  --batch_size 64 \
  --gpu 0 \
  --num_workers 4 \
  --predictions_csv ./official_cnn_synth_predictions.csv
```

如果显存不足，将 `--batch_size 64` 改为 `32` 或 `16`。完整测试应出现 13 个生成器和总数 90,329。

### 3.3 自训练 LoRA 模型测试

自训练 checkpoint 通过 `scripts/test_checkpoint.py`，必须使用训练时一致的 LoRA 和局部特征参数。

无局部特征模型示例：

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/test_checkpoint.py \
  --dataroot ./CNN_synth_testset \
  --checkpoint ./c2p_checkpoints/baseline/model.pth \
  --clip_path ./clip-vit-large-patch14 \
  --batch_size 64 \
  --gpu 0 \
  --num_workers 4 \
  --lora_r 6 \
  --lora_alpha 6 \
  --lora_dropout 0.8 \
  --predictions_csv ./baseline_cnn_synth_predictions.csv
```

局部特征模型示例：

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/test_checkpoint.py \
  --dataroot ./CNN_synth_testset \
  --checkpoint ./c2p_checkpoints/local/model.pth \
  --clip_path ./clip-vit-large-patch14 \
  --batch_size 64 \
  --gpu 0 \
  --num_workers 4 \
  --lora_r 6 \
  --lora_alpha 6 \
  --lora_dropout 0.8 \
  --use_local_features \
  --local_layer 12 \
  --local_dim 256 \
  --local_dropout 0.1 \
  --local_pool mean_std \
  --predictions_csv ./local_cnn_synth_predictions.csv
```

`--lora_r` 是 LoRA 的低秩维度，必须与训练 checkpoint 一致；它不是一个可以随意更换的测试超参数。训练使用 `lora_r=6` 的 checkpoint 不能用 `--lora_r 16` 加载。

### 3.4 Logit 分布分析

`scripts/plot_logit_dist.py` 可以对比基线与局部模型，并使用共享 bins 和横轴：

```bash
python scripts/plot_logit_dist.py \
  --dataroot ./my_first_test \
  --checkpoint ./c2p_checkpoints/baseline/model.pth \
  --checkpoint_label Baseline \
  --compare_checkpoint ./c2p_checkpoints/local/model.pth \
  --compare_label Local \
  --compare_use_local_features \
  --clip_path ./clip-vit-large-patch14 \
  --lora_r 6 --lora_alpha 6 --lora_dropout 0.8 \
  --local_layer 12 --local_dim 256 \
  --local_dropout 0.1 --local_pool mean_std \
  --save ./baseline_vs_local_logits.png
```

## 4. 训练行为与配置

核心文件：

- `scripts/train.py`
- `networks/trainer.py`
- `options/base_options.py`
- `options/train_options.py`

`networks/trainer.py` 中的训练模型：

- CLIP ViT-L/14 vision tower。
- 对 `q_proj/k_proj/v_proj` 使用 PEFT LoRA。
- 视觉 backbone 默认冻结基础权重，LoRA 可训练。
- 分类损失为 `BCEWithLogitsLoss`。
- 训练总损失为图文对比损失加分类损失：

```text
loss = contrastive_loss + claloss * BCEWithLogitsLoss
```

当前默认选项中：

```text
batch_size=64
lr=1e-4
claloss=0.5
lora_r=16
lora_alpha=32
lora_dropout=0.1
total_steps=1000
eval_freq=200
```

用户此前的部分实验配置曾使用：`batch_size=32`、`claloss=8.0`、`lr=0.0004`、`lora_r=16`、`lora_alpha=32`、`lora_dropout=0.1`、`total_steps=1000`、四个训练类别 `car,cat,chair,horse`。比较模型时必须记录并匹配这些配置。

### 评估频率

`utils/evaluation_schedule.py` 和 `scripts/train.py` 当前行为是：

- `eval_freq > 0` 时，在 `step % eval_freq == 0` 进行评估。
- 训练结束时，如果最后一步没有刚好评估，再进行一次最终评估。
- `--eval_freq 0` 可以关闭训练期间的周期性评估，但仍保留最终评估。
- 默认 `eval_freq=200`，`total_steps=1000` 时会在 200、400、600、800、1000 步附近触发评估，因此若要避免频繁遍历多个测试集，应显式使用 `--eval_freq 0`。

`tests/test_evaluation_schedule.py` 测试的是这个调度策略，不是模型准确率。

## 5. 局部特征当前实现

文件：`networks/trainer.py`。

启用 `--use_local_features` 后：

1. 读取 CLIP vision tower 第 `local_layer` 层的 hidden states。
2. 去掉 CLS token，仅保留 patch tokens。
3. 对 patch token 做 LayerNorm。
4. `local_pool=mean` 时做 patch mean；`mean_std` 时拼接 patch mean 和 patch standard deviation。
5. 通过 `local_projector` 投影到 `local_dim=256`。
6. 与最终全局 image embedding 拼接，输入两层 MLP 分类头。

当前默认局部配置：

```text
local_layer=12
local_dim=256
local_dropout=0.1
local_pool=mean_std
```

重要限制：

- mean/std 池化丢失了 patch 的空间位置，因此它更准确地说是“中间层 patch 分布统计”，不是显式局部区域建模。
- 无局部模型的分类头是 `Linear(768, 1)`；局部模型的分类头是 `Linear(768+256,256) -> GELU -> Dropout -> Linear(256,1)`。因此当前结果存在分类头容量混杂。
- 局部分支没有专用局部监督，主要通过 BCE 分类损失训练；图文对比损失仍作用于全局 image embedding。
- 官方、基线和局部模型现在共用 `utils/binary_evaluation.py` 的纯图像数据发现、预处理、DataLoader、指标和 CSV 导出逻辑；LoRA 前向使用 `model(img, None, None, cla=True)`，不再在评测 DataLoader 中读取 Caption/tokenizer。

## 6. 当前 CNN_synth 测试结果

三次截图按用户提供顺序为：官方模型、自训练基线、局部特征模型。

注意：三次结果都只有 12 个生成器、总数 89,969，缺少预期的 `seeingdark` 360 张图。三者在同一子集上比较仍然公平，但不是完整 90,329 张测试结果。

### 6.1 宏观结果

后两行的 Real/Fake 宏平均根据截图逐项数值计算，存在四舍五入误差。

| 模型 | Macro ACC | Macro Real ACC | Macro Fake ACC | Macro AP |
|---|---:|---:|---:|---:|
| 官方模型 | 95.45% | 99.14% | 91.76% | 99.64% |
| 自训练基线 | 约 83.96% | 约 80.50% | 约 87.41% | 约 98.69% |
| 局部特征 | 85.13% | 约 82.83% | 约 87.42% | 98.29% |

### 6.2 自训练基线与局部模型的逐生成器差异

局部模型相对基线的 ACC 变化：

```text
biggan             +3.15
crn                +0.83
cyclegan           +2.50
deepfake           +6.30
gaugan             +4.79
imle               +0.83
progan             +0.07
san                +0.91
stargan            +0.02
stylegan           -3.56
stylegan2          -3.71
whichfaceisreal    +1.95
```

局部模型的 AP 变化大多为下降，尤其是 StyleGAN 和 StyleGAN2。总体上 ACC 提高约 1.17 个百分点，但 AP 下降约 0.40 个百分点。

### 6.3 关键诊断

CRN/IMLE：

- AP 接近 99.9%，说明样本排序能力很强。
- 但基线 Real ACC 约 0.09%，局部模型约 1.75%，Fake ACC 均约 100%。
- 说明大量 logit 都落在 sigmoid 0.5 阈值的假图侧，主要是校准/偏置问题，而不是完全无法区分真假。

DeepFake：

- Real ACC 为 100%。
- Fake ACC 从 24.43% 提升到 37.06%。
- AP 仍在 97% 以上。
- 说明不同生成器存在方向不同的 logit 偏移，单一全局 0.5 阈值不一定适合所有生成器。

官方模型虽然仍在 SAN、StyleGAN、StyleGAN2、WhichFaceIsReal 上有较低 Fake ACC，但其宏观 ACC 和 AP 明显高于两个自训练模型。

## 7. 当前结论：局部方案是否有创新意义

### 工程价值

有一定价值：局部分支在 BigGAN、CycleGAN、DeepFake、GauGAN 等数据集提升了固定阈值 ACC，且不增加额外外部模型，符合纯图像推理约束。

### 论文创新性

当前 mean/std patch 统计后直接拼接的方案不足以单独作为强创新点，原因是：

1. 中间层特征池化和全局拼接属于常见范式。
2. 空间信息被 mean/std 完全压缩。
3. 局部模型使用了更强的 MLP 分类头，尚未排除容量因素。
4. 局部分支没有专门的 patch 级监督或一致性约束。
5. ACC 上升但 AP 下降，说明目前更像 logit 校准变化，而非稳定的跨生成器伪影表征提升。

### 最低优先级的公平消融

先固定数据、训练步数、随机种子和优化器，比较：

```text
A: Global feature + Linear
B: Global feature + 与局部模型相同的 MLP
C: Global + layer-12 mean + MLP
D: Global + layer-12 mean_std + MLP
```

每组至少 3 个 seed。主要指标用 Macro AP，同时报告 ACC、Real/Fake ACC、ECE 或 Brier score、logit 均值/标准差和分离度。阈值只能在独立验证集选择，不能根据测试生成器单独调阈值。

## 8. 值得继续研究的方向

以下是研究候选，不代表已经实现：

1. **多层 patch 伪影证据**：同时使用浅层、中层和深层 patch token，比较 mean、top-k、attention pooling，避免固定单层和全局平均丢失异常区域。
2. **全局分数加门控局部残差**：

   ```text
   final_logit = global_logit + adaptive_gate * local_logit
   ```

   gate 初始化接近 0，使局部分支不能轻易覆盖全局分支，重点缓解 StyleGAN 系列退化。
3. **局部排序/一致性损失**：除了 BCE，对 patch 级伪影证据加入排序或全局-局部一致性约束，直接改善 AP，而不只优化 0.5 阈值 ACC。
4. **生成器无关的 logit 校准**：在独立验证集学习全局温度/偏置或不确定性校准，分析 CRN/IMLE 与 DeepFake 的相反 logit 漂移。校准本身通常不是充分的论文创新，但可作为方法组件和诊断工具。
5. **效率优化**：当前 `output_hidden_states=True` 会保留所有层 hidden states；如果只需少数层，可以用选定层 hook 或受控 forward，降低显存开销。

这些方向必须先检索真实论文并做公平消融，不能只凭名称包装成创新。

## 9. 研究优先级

1. 先补齐 `seeingdark`，确认完整 90,329 张图结果。
2. 先缩小官方发布模型与自训练基线之间约 10 个百分点的差距，核对 LoRA 配置、全局 batch、学习率、步数、CLIP 权重版本、采样和 checkpoint 选择。
3. 做 Global-MLP 对照，确认局部收益是否真实。
4. 以 AP 和跨生成器稳定性为主指标，不只看固定 0.5 阈值 ACC。
5. 只有局部特征在公平对照和多 seed 下稳定改善 AP，才继续发展门控、多层 patch 或局部损失方案。

## 10. 给下一个 GPT 的工作指令

请先阅读本文件和以下核心代码：

```text
networks/trainer.py
scripts/train.py
scripts/test_checkpoint.py
scripts/test_airplane_official.py
scripts/plot_logit_dist.py
options/base_options.py
utils/evaluation_schedule.py
```

开始任何代码修改前：

1. 检查 `git status`，保留用户未提交文件。
2. 确认用户是要工程实现、诊断，还是仅要研究建议。
3. 如果是新研究方案，先做最小公平消融，不要直接替换现有训练流程。
4. 运行适合风险范围的语法检查、单元测试和小规模验证。
5. 提交时只包含明确相关文件，不提交数据集和模型。
