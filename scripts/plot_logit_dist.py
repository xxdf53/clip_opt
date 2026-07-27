"""Plot raw-logit distributions from self-trained C2P-CLIP checkpoints.

One or two baseline/Logit Anchor checkpoints can be compared with shared bins
and axes. Both checkpoints must use the same LoRA configuration.
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

from utils.binary_dataset_layout import discover_binary_groups
from utils.binary_evaluation import build_group_dataset, build_transform
from utils.checkpoint_loading import load_self_trained_checkpoint
from utils.logit_distribution import build_shared_bin_edges, compute_logit_stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Plot raw logits from train.py LoRA checkpoints')
    parser.add_argument(
        '--dataroot',
        required=True,
        help='binary dataset root with direct or nested 0_real/1_fake folders',
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='primary train.py checkpoint containing model and total_steps',
    )
    parser.add_argument('--checkpoint_label', default='Primary')
    parser.add_argument(
        '--compare_checkpoint',
        help='optional second checkpoint plotted on the same axes',
    )
    parser.add_argument('--compare_label', default='Comparison')
    parser.add_argument(
        '--clip_path',
        default=str(ROOT / 'clip-vit-large-patch14'),
        help='local CLIP ViT-L/14 model directory',
    )
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--bins', type=int, default=100)
    parser.add_argument('--save', default='logit_distribution.png')
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        parser.error('--batch_size must be positive')
    if args.num_workers < 0:
        parser.error('--num_workers cannot be negative')
    if args.bins <= 0:
        parser.error('--bins must be positive')
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
    groups = discover_binary_groups(dataroot)
    leaves = [
        leaf
        for group_leaves in groups.values()
        for leaf in group_leaves
    ]
    dataset = build_group_dataset(leaves, build_transform())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
    )


def load_lora_checkpoint(checkpoint_path, args, device):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    print(f'Loading checkpoint: {checkpoint_path}')
    model, total_steps = load_self_trained_checkpoint(
        checkpoint_path=checkpoint_path,
        clip_path=args.clip_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=device,
    )
    steps = total_steps if total_steps is not None else 'unknown'
    print(f'  Total training steps: {steps}')
    return model


def collect_raw_logits(model, data_loader, device):
    real_logits = []
    fake_logits = []

    with torch.no_grad():
        for images, labels, _paths in data_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images, None, None, cla=True).flatten().cpu()
            labels = labels.flatten()
            real_logits.extend(logits[labels == 0].tolist())
            fake_logits.extend(logits[labels == 1].tolist())

    compute_logit_stats(real_logits, fake_logits)
    return real_logits, fake_logits


def release_device_memory(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def analyze_checkpoint(label, checkpoint, args, data_loader, device):
    model = load_lora_checkpoint(checkpoint, args, device)
    try:
        real_logits, fake_logits = collect_raw_logits(
            model, data_loader, device)
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
    distributions = [
        distribution
        for result in results
        for distribution in (result['real'], result['fake'])
    ]
    bin_edges = np.asarray(build_shared_bin_edges(distributions, bins=bins))
    colors = [('seagreen', 'firebrick'), ('royalblue', 'darkorange')]

    plt.figure(figsize=(11, 6.5))
    for index, result in enumerate(results):
        real_color, fake_color = colors[index]
        histtype = 'stepfilled' if index == 0 else 'step'
        alpha = 0.35 if index == 0 else 0.9
        linewidth = 1.2 if index == 0 else 2.0
        plt.hist(
            result['real'],
            bins=bin_edges,
            histtype=histtype,
            alpha=alpha,
            linewidth=linewidth,
            color=real_color,
            label=f"{result['label']} Real (n={len(result['real'])})",
        )
        plt.hist(
            result['fake'],
            bins=bin_edges,
            histtype=histtype,
            alpha=alpha,
            linewidth=linewidth,
            color=fake_color,
            label=f"{result['label']} Fake (n={len(result['fake'])})",
        )

    plt.axvline(
        x=0.0,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label='Decision boundary (logit=0)',
    )
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
        args,
        data_loader,
        device,
    )]
    if args.compare_checkpoint:
        results.append(analyze_checkpoint(
            args.compare_label,
            args.compare_checkpoint,
            args,
            data_loader,
            device,
        ))

    for result in results:
        print_stats(result)
    plot_results(results, args.bins, args.save)


if __name__ == '__main__':
    main()
