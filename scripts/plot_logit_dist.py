"""Plot raw-logit distributions from self-trained C2P-CLIP checkpoints.

Examples:
  # Baseline LoRA model
  python scripts/plot_logit_dist.py \
    --dataroot ./my_first_test \
    --checkpoint ./c2p_checkpoints/baseline/model.pth \
    --clip_path ./clip-vit-large-patch14 \
    --lora_r 6 --lora_alpha 6 --lora_dropout 0.8

  # Local-feature LoRA model
  python scripts/plot_logit_dist.py \
    --dataroot ./my_first_test \
    --checkpoint ./c2p_checkpoints/local/model.pth \
    --clip_path ./clip-vit-large-patch14 \
    --lora_r 6 --lora_alpha 6 --lora_dropout 0.8 \
    --use_local_features --local_layer 12 --local_dim 256 \
    --local_dropout 0.1 --local_pool mean_std

  # Baseline and local model on shared axes
  python scripts/plot_logit_dist.py \
    --dataroot ./my_first_test \
    --checkpoint ./c2p_checkpoints/baseline/model.pth \
    --compare_checkpoint ./c2p_checkpoints/local/model.pth \
    --compare_use_local_features \
    --clip_path ./clip-vit-large-patch14 \
    --lora_r 6 --lora_alpha 6 --lora_dropout 0.8
"""

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.datasets import _TranslateDuplicate, pil_loader
from networks.trainer import CLIPModel_lora
from utils.checkpoint_loading import (
    LOCAL_FUSIONS,
    extract_training_state_dict,
    parse_gate_override,
    resolve_local_fusion,
)
from utils.logit_distribution import build_shared_bin_edges, compute_logit_stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Plot raw-logit distributions from train.py LoRA checkpoints')
    parser.add_argument(
        '--dataroot', required=True,
        help='binary image dataset containing 0_real/ and 1_fake/')
    parser.add_argument(
        '--checkpoint', required=True,
        help='primary train.py checkpoint containing model and total_steps')
    parser.add_argument('--checkpoint_label', default='Baseline')
    parser.add_argument(
        '--compare_checkpoint',
        help='optional second train.py checkpoint plotted on the same axes')
    parser.add_argument('--compare_label', default='Local')
    parser.add_argument('--compare_use_local_features', action='store_true')
    parser.add_argument(
        '--compare_local_fusion',
        choices=['auto'] + list(LOCAL_FUSIONS), default='auto')
    parser.add_argument(
        '--compare_gate_override', type=parse_gate_override, default=None,
        metavar='learned|FLOAT')
    parser.add_argument(
        '--clip_path', default=str(ROOT / 'clip-vit-large-patch14'),
        help='local CLIP ViT-L/14 model directory')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--use_local_features', action='store_true')
    parser.add_argument('--local_layer', type=int, default=12)
    parser.add_argument('--local_dim', type=int, default=256)
    parser.add_argument('--local_dropout', type=float, default=0.1)
    parser.add_argument(
        '--local_pool', choices=['mean', 'mean_std'], default='mean_std')
    parser.add_argument(
        '--local_fusion', choices=['auto'] + list(LOCAL_FUSIONS),
        default='auto')
    parser.add_argument('--local_gate_init', type=float, default=0.01)
    parser.add_argument(
        '--gate_override', type=parse_gate_override, default=None,
        metavar='learned|FLOAT')
    parser.add_argument('--bins', type=int, default=100)
    parser.add_argument('--save', default='logit_distribution.png')
    args = parser.parse_args(argv)

    if args.compare_use_local_features and not args.compare_checkpoint:
        parser.error('--compare_use_local_features requires --compare_checkpoint')
    return args


def resolve_device(gpu):
    if not torch.cuda.is_available():
        print('CUDA is unavailable; using CPU.')
        return torch.device('cpu')
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise ValueError(
            f'gpu must be in [0, {torch.cuda.device_count() - 1}], got {gpu}')
    return torch.device(f'cuda:{gpu}')


def build_image_loader(dataroot, batch_size, num_workers, device):
    transform = transforms.Compose([
        _TranslateDuplicate(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    dataset = datasets.ImageFolder(
        root=str(Path(dataroot).expanduser()),
        transform=transform,
        loader=pil_loader,
    )
    expected_classes = {'0_real': 0, '1_fake': 1}
    if dataset.class_to_idx != expected_classes:
        raise ValueError(
            f'dataroot must contain exactly 0_real/ and 1_fake/; '
            f'found {dataset.class_to_idx}')

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
    )


def load_lora_checkpoint(checkpoint_path, args, device, use_local_features,
                         local_fusion='auto'):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')

    print(f'Loading checkpoint: {checkpoint_path}')
    payload = torch.load(
        str(checkpoint_path), map_location='cpu', weights_only=True)
    state_dict, total_steps = extract_training_state_dict(payload)
    print(f'  Total training steps: '
          f'{total_steps if total_steps is not None else "unknown"}')
    resolved_local_fusion = resolve_local_fusion(
        state_dict,
        requested=local_fusion,
        use_local_features=use_local_features,
    )
    if use_local_features:
        print(f'  Local fusion: {resolved_local_fusion}')

    model = CLIPModel_lora(
        name=args.clip_path,
        num_classes=1,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_local_features=use_local_features,
        local_layer=args.local_layer,
        local_dim=args.local_dim,
        local_dropout=args.local_dropout,
        local_pool=args.local_pool,
        local_fusion=resolved_local_fusion,
        local_gate_init=args.local_gate_init,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    gate = model.local_gate_value()
    if gate is not None:
        print(f'  Learned local gate: {gate.detach().item():.6f}')
    return model


def collect_raw_logits(model, data_loader, device, gate_override=None):
    real_logits = []
    fake_logits = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            logits = model.forward_components(
                images, gate_override=gate_override)['final_logits']
            logits = logits.flatten().cpu()
            labels = labels.flatten()
            real_logits.extend(logits[labels == 0].tolist())
            fake_logits.extend(logits[labels == 1].tolist())

    compute_logit_stats(real_logits, fake_logits)
    return real_logits, fake_logits


def release_device_memory(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def analyze_checkpoint(label, checkpoint, use_local_features, local_fusion,
                       gate_override, args, data_loader, device):
    model = load_lora_checkpoint(
        checkpoint, args, device, use_local_features=use_local_features,
        local_fusion=local_fusion)
    if (gate_override is not None
            and model.local_fusion not in (
                'residual_gate', 'adaptive_residual')):
        raise ValueError('gate override requires a gated local checkpoint')
    try:
        real_logits, fake_logits = collect_raw_logits(
            model, data_loader, device, gate_override=gate_override)
    finally:
        del model
        release_device_memory(device)

    return {
        'label': label,
        'checkpoint': str(Path(checkpoint).expanduser().resolve()),
        'real': real_logits,
        'fake': fake_logits,
        'stats': compute_logit_stats(real_logits, fake_logits),
    }


def print_stats(result):
    stats = result['stats']
    print(f"\n[{result['label']}] {result['checkpoint']}")
    print(f"  Real logits: mean={stats['real_mean']:.3f}, "
          f"std={stats['real_std']:.3f}, n={len(result['real'])}")
    print(f"  Fake logits: mean={stats['fake_mean']:.3f}, "
          f"std={stats['fake_std']:.3f}, n={len(result['fake'])}")
    print(f"  Separation (d'-like): {stats['separation']:.2f}")


def plot_results(results, bins, save_path):
    bin_edges = np.asarray(build_shared_bin_edges(
        [distribution
         for result in results
         for distribution in (result['real'], result['fake'])],
        bins=bins,
    ))
    colors = [('seagreen', 'firebrick'), ('royalblue', 'darkorange')]

    plt.figure(figsize=(11, 6.5))
    for index, result in enumerate(results):
        real_color, fake_color = colors[index]
        histtype = 'stepfilled' if index == 0 else 'step'
        alpha = 0.35 if index == 0 else 0.9
        linewidth = 1.2 if index == 0 else 2.0
        plt.hist(
            result['real'], bins=bin_edges, histtype=histtype,
            alpha=alpha, linewidth=linewidth, color=real_color,
            label=f"{result['label']} Real (n={len(result['real'])})")
        plt.hist(
            result['fake'], bins=bin_edges, histtype=histtype,
            alpha=alpha, linewidth=linewidth, color=fake_color,
            label=f"{result['label']} Fake (n={len(result['fake'])})")

    plt.axvline(
        x=0.0, color='black', linestyle='--', linewidth=1.5,
        label='Decision boundary (logit=0)')
    plt.xlim(bin_edges[0], bin_edges[-1])
    plt.xlabel('Raw classifier logit', fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.title('Real and Fake Logit Distributions', fontsize=15)
    plt.legend(fontsize=10)
    plt.tight_layout()

    save_path = Path(save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'\nSaved plot to: {save_path}')


def main(argv=None):
    args = parse_args(argv)
    args.clip_path = str(Path(args.clip_path).expanduser().resolve())
    if not Path(args.clip_path).is_dir():
        raise FileNotFoundError(
            f'CLIP model directory not found: {args.clip_path}')

    device = resolve_device(args.gpu)
    print(f'Device: {device}')
    data_loader = build_image_loader(
        args.dataroot, args.batch_size, args.num_workers, device)
    print(f'Images: {len(data_loader.dataset)}')

    results = [analyze_checkpoint(
        args.checkpoint_label,
        args.checkpoint,
        args.use_local_features,
        args.local_fusion,
        args.gate_override,
        args,
        data_loader,
        device,
    )]
    if args.compare_checkpoint:
        results.append(analyze_checkpoint(
            args.compare_label,
            args.compare_checkpoint,
            args.compare_use_local_features,
            args.compare_local_fusion,
            args.compare_gate_override,
            args,
            data_loader,
            device,
        ))

    for result in results:
        print_stats(result)
    plot_results(results, args.bins, args.save)


if __name__ == '__main__':
    main()
