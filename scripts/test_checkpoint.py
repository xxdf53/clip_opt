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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import time
import argparse
import torch

from networks.trainer import CLIPModel_lora
from utils.binary_dataset_layout import discover_binary_groups
from utils.binary_evaluation import (
    evaluate_groups,
    format_diagnostics,
    format_metrics,
    write_predictions_csv,
)
from utils.checkpoint_loading import (
    extract_training_state_dict,
    resolve_local_fusion,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Test self-trained C2P-CLIP LoRA checkpoint')
    parser.add_argument('--dataroot',    type=str, required=True,
                        help='test dataset root (e.g., ./UniversalFakeDetect/test/)')
    parser.add_argument('--checkpoint',  type=str,
                        default=os.path.join(str(ROOT), 'checkpoints',
                                             'model_epoch_9_total_steps_810_testacc_50.069.pth'),
                        help='path to .pth checkpoint')
    parser.add_argument('--clip_path',   type=str,
                        default=os.path.join(str(ROOT), 'clip-vit-large-patch14'),
                        help='path to CLIP ViT-L/14 model')
    parser.add_argument('--batch_size',  type=int, default=64)
    parser.add_argument('--gpu',         type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument(
        '--predictions_csv',
        help='optional CSV path for per-image raw logits and scores',
    )
    parser.add_argument('--lora_r',      type=int, default=16)
    parser.add_argument('--lora_alpha',  type=int, default=32)
    parser.add_argument('--lora_dropout',type=float, default=0.1)
    parser.add_argument('--use_local_features', action='store_true')
    parser.add_argument('--local_layer', type=int, default=12)
    parser.add_argument('--local_dim', type=int, default=256)
    parser.add_argument('--local_dropout', type=float, default=0.1)
    parser.add_argument('--local_pool', choices=['mean', 'mean_std'],
                        default='mean_std')
    parser.add_argument('--local_fusion',
                        choices=['auto', 'concat', 'residual_gate'],
                        default='auto',
                        help='auto-detect legacy concat or residual-gate checkpoints')
    parser.add_argument('--local_gate_init', type=float, default=0.01,
                        help='constructor value; checkpoint restores the learned gate')
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error('--batch_size must be positive')
    if args.num_workers < 0:
        parser.error('--num_workers cannot be negative')
    return args


def load_checkpoint(checkpoint_path, clip_path, lora_r, lora_alpha,
                    lora_dropout, device, use_local_features=False,
                    local_layer=12, local_dim=256, local_dropout=0.1,
                    local_pool='mean_std', local_fusion='auto',
                    local_gate_init=0.01):
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

    new_state_dict, total_steps = extract_training_state_dict(raw)
    print('  Total training steps: '
          f'{total_steps if total_steps is not None else "unknown"}')
    resolved_local_fusion = resolve_local_fusion(
        new_state_dict,
        requested=local_fusion,
        use_local_features=use_local_features,
    )
    if use_local_features:
        print(f'  Local fusion: {resolved_local_fusion}')

    # 创建与训练时相同结构的模型
    model = CLIPModel_lora(
        name=clip_path,
        num_classes=1,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_local_features=use_local_features,
        local_layer=local_layer,
        local_dim=local_dim,
        local_dropout=local_dropout,
        local_pool=local_pool,
        local_fusion=resolved_local_fusion,
        local_gate_init=local_gate_init,
    )
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()

    gate = model.local_gate_value()
    if gate is not None:
        print(f'  Learned local gate: {gate.detach().item():.6f}')

    print(f'  Model loaded successfully.')
    return model


def resolve_existing_path(path, description, expect_directory):
    resolved = Path(path).expanduser().resolve()
    exists = resolved.is_dir() if expect_directory else resolved.is_file()
    if not exists:
        expected = 'directory' if expect_directory else 'file'
        raise FileNotFoundError(
            f'{description} {expected} not found: {resolved}')
    return resolved


def resolve_device(gpu):
    if not torch.cuda.is_available():
        print('CUDA is unavailable; using CPU.')
        return torch.device('cpu')
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise ValueError(
            f'gpu must be in [0, {torch.cuda.device_count() - 1}], got {gpu}')
    return torch.device(f'cuda:{gpu}')


def lora_forward_logits(model, images):
    return model(images, None, None, cla=True)


def main(argv=None):
    args = parse_args(argv)
    dataroot = resolve_existing_path(
        args.dataroot, 'dataset root', expect_directory=True)
    checkpoint = resolve_existing_path(
        args.checkpoint, 'checkpoint', expect_directory=False)
    clip_path = resolve_existing_path(
        args.clip_path, 'CLIP model', expect_directory=True)
    groups = discover_binary_groups(dataroot)
    device = resolve_device(args.gpu)

    model = load_checkpoint(
        str(checkpoint), str(clip_path),
        args.lora_r, args.lora_alpha, args.lora_dropout, device,
        use_local_features=args.use_local_features,
        local_layer=args.local_layer,
        local_dim=args.local_dim,
        local_dropout=args.local_dropout,
        local_pool=args.local_pool,
        local_fusion=args.local_fusion,
        local_gate_init=args.local_gate_init,
    )

    print(f'Dataset: {dataroot}')
    print(f'Generators: {len(groups)}')
    print(f'Device: {device}')
    print(f'Model: {checkpoint}')
    t_start = time.time()
    print('\n' + '=' * 92)
    summary = evaluate_groups(
        groups,
        forward_logits=lambda images: lora_forward_logits(model, images),
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        on_group_complete=lambda index, name, metrics, stats: (
            print(f'[{index:02d}] ' + format_metrics(name, metrics)),
            print('     ' + format_diagnostics(metrics, stats)),
        ),
    )
    elapsed = time.time() - t_start

    print('=' * 92)
    print('     ' + format_metrics('Macro mean', summary['macro_metrics']))
    print('     ' + format_diagnostics(summary['macro_metrics']))
    print('     ' + format_metrics('Overall', summary['overall_metrics']))
    print('     ' + format_diagnostics(
        summary['overall_metrics'], summary['overall_logit_stats']))
    if args.predictions_csv:
        output_path = write_predictions_csv(
            summary['predictions'], args.predictions_csv)
        print(f'Predictions: {output_path}')
    print(f'Elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)')


if __name__ == '__main__':
    main()
