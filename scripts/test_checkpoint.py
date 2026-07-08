"""
=============================================================================
 test_checkpoint.py — 测试自训练 C2P-CLIP LoRA 模型
=============================================================================
 用法（在 VSCode 终端 / Git Bash 中运行）:

   # 1) 测试 UniversalFakeDetect 的 15 个 held-out 类别（评估跨类别泛化）
   python scripts/test_checkpoint.py \
       --dataroot ./UniversalFakeDetect/test/ \
       --checkpoint ./checkpoints/model_epoch_9_total_steps_810_testacc_50.069.pth

   # 2) 测试 GenImage（跨生成器泛化）
   python scripts/test_checkpoint.py \
       --dataroot ./GenImage_Dataset/test/ \
       --checkpoint ./checkpoints/model_epoch_9_total_steps_810_testacc_50.069.pth

   # 3) 测试 Chameleon
   python scripts/test_checkpoint.py \
       --dataroot ./Chameleon/test/ \
       --checkpoint ./checkpoints/model_epoch_9_total_steps_810_testacc_50.069.pth

   # 4) 指定 batch_size / GPU
   python scripts/test_checkpoint.py \
       --dataroot ./UniversalFakeDetect/test/ \
       --checkpoint ./checkpoints/model_epoch_9_total_steps_810_testacc_50.069.pth \
       --batch_size 64 --gpu 0

=============================================================================
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix

from data import create_dataloader
from networks.trainer import CLIPModel_lora

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description='Test self-trained C2P-CLIP LoRA checkpoint')
    parser.add_argument('--dataroot',    type=str, required=True,
                        help='test dataset root (e.g., ./UniversalFakeDetect/test/)')
    parser.add_argument('--checkpoint',  type=str,
                        default=os.path.join(ROOT, 'checkpoints',
                                             'model_epoch_9_total_steps_810_testacc_50.069.pth'),
                        help='path to .pth checkpoint')
    parser.add_argument('--clip_path',   type=str,
                        default=os.path.join(ROOT, 'clip-vit-large-patch14'),
                        help='path to CLIP ViT-L/14 model')
    parser.add_argument('--batch_size',  type=int, default=64)
    parser.add_argument('--gpu',         type=int, default=0)
    parser.add_argument('--lora_r',      type=int, default=16)
    parser.add_argument('--lora_alpha',  type=int, default=32)
    parser.add_argument('--lora_dropout',type=float, default=0.1)
    return parser.parse_args()


def load_checkpoint(checkpoint_path, clip_path, lora_r, lora_alpha, lora_dropout, device):
    """
    加载训练保存的 LoRA checkpoint。

    checkpoint 格式（由 base_model.py save_networks 保存）:
      {
          'model': <nn.DataParallel(CLIPModel_lora) 的 state_dict>,
          'total_steps': int
      }

    关键处理：state_dict 中的 key 带 'module.' 前缀（DataParallel 导致），
    加载到单卡时需去除。
    """
    print(f'Loading checkpoint: {checkpoint_path}')
    raw = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    state_dict = raw['model']
    total_steps = raw['total_steps']
    print(f'  Total training steps: {total_steps}')

    # 去除 DataParallel 的 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v

    # 创建与训练时相同结构的模型
    model = CLIPModel_lora(
        name=clip_path,
        num_classes=1,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()

    print(f'  Model loaded successfully.')
    return model


class OptForDataLoader:
    """构造 DataLoader 所需的最小配置对象（模拟 TestOptions）"""
    def __init__(self, dataroot, batch_size, clip_path):
        self.dataroot      = dataroot + os.sep
        self.imgroot       = dataroot + os.sep
        self.textroot      = os.path.join(ROOT, 'Genimage_CNNDetection_CLIP_prefix_caption', 'train')
        self.clip          = clip_path
        self.mode          = 'binary'
        self.isTrain       = False
        self.batch_size    = batch_size
        self.loadSize      = 224
        self.cropSize      = 224
        self.no_resize     = False
        self.no_crop       = False
        self.no_flip       = True
        self.serial_batches = True
        self.class_bal     = False
        self.num_threads   = 4
        self.classes       = []
        self.rz_interp     = ['bilinear']
        self.blur_prob     = 0
        self.blur_sig      = [0.5]
        self.jpg_prob      = 0
        self.jpg_method    = ['pil']
        self.jpg_qual      = [75]


def test_one_dir(model, dataroot, batch_size, clip_path, device):
    """对单个目录（含 0_real/1_fake 子文件夹）做推理，返回 (acc, ap, r_acc, f_acc, cm, n)"""
    dl_opt = OptForDataLoader(dataroot, batch_size, clip_path)
    data_loader = create_dataloader(dl_opt)
    n_images = len(data_loader.dataset)

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in data_loader:
            # batch = (path, img, text, input_ids, attention_mask, label)
            img, label = batch[1].to(device), batch[5]
            # cla=True → 只走分类头，跳过对比损失分支
            logits = model(img, None, None, cla=True)
            y_pred.extend(logits.sigmoid().flatten().cpu().tolist())
            y_true.extend(label.flatten().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred > 0.5)
    acc  = accuracy_score(y_true, y_pred > 0.5) * 100
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5) * 100
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5) * 100
    ap   = average_precision_score(y_true, y_pred) * 100
    return acc, ap, r_acc, f_acc, cm, n_images


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # ── 1. 加载模型 ──
    model = load_checkpoint(
        args.checkpoint, args.clip_path,
        args.lora_r, args.lora_alpha, args.lora_dropout, device
    )

    dataroot = args.dataroot.rstrip('/').rstrip('\\')

    # ── 2. 判断测试目录结构 ──
    # 情况 A: dataroot 下直接有 0_real/1_fake → 单个测试集
    # 情况 B: dataroot 下有多个子文件夹（如 UniversalFakeDetect/test/airplane/, ...）
    #        每个子文件夹内含 0_real/1_fake → 多子集汇总
    subdirs = sorted(os.listdir(dataroot))
    has_real_fake = '0_real' in subdirs or '1_fake' in subdirs

    if has_real_fake:
        # 情况 A: 单数据集
        test_sets = {os.path.basename(dataroot): dataroot}
    else:
        # 情况 B: 多子集
        test_sets = {}
        for d in subdirs:
            full = os.path.join(dataroot, d)
            if os.path.isdir(full):
                test_sets[d] = full

    # ── 3. 逐子集测试 ──
    print(f'\n{"="*60}')
    print(f'Testing on: {dataroot}')
    print(f'Test subsets: {len(test_sets)}')
    print(f'Model: {args.checkpoint}')
    print(f'{"="*60}\n')

    all_accs, all_aps = [], []
    t_start = time.time()

    for idx, (name, path) in enumerate(test_sets.items()):
        acc, ap, r_acc, f_acc, cm, n = test_one_dir(
            model, path, args.batch_size, args.clip_path, device
        )
        all_accs.append(acc)
        all_aps.append(ap)
        print(f"({idx:2d} | {name:20s})  n={n:>7d}  "
              f"ACC={acc:6.2f}%  Real_ACC={r_acc:6.2f}%  Fake_ACC={f_acc:6.2f}%  AP={ap:6.2f}%")

    elapsed = time.time() - t_start

    # ── 4. 汇总 ──
    print(f'\n{"="*60}')
    mean_acc = np.mean(all_accs) if all_accs else 0
    mean_ap  = np.mean(all_aps) if all_aps else 0
    print(f'{"Mean":>24s}  ACC={mean_acc:6.2f}%  AP={mean_ap:6.2f}%')
    print(f'Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'{"="*60}')
