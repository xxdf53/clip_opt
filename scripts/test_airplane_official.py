"""Evaluate an official C2P-CLIP model on direct or nested binary datasets.

Examples:
  python scripts/test_airplane_official.py \
    --dataroot ./CNN_synth_testset \
    --model_path ./C2P_CLIP_release_20240901.pth \
    --clip_path ./clip-vit-large-patch14

  # Backward-compatible positional paths
  python scripts/test_airplane_official.py \
    ./my_first_test ./C2P_CLIP_release_20240901.pth
"""

import argparse
import sys
import time
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

from data.datasets import _TranslateDuplicate, pil_loader
from scripts.inference import C2P_CLIP
from utils.binary_dataset_layout import discover_binary_groups
from utils.binary_metrics import compute_binary_metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Test an official C2P-CLIP model on binary datasets')
    parser.add_argument('legacy_dataroot', nargs='?', help=argparse.SUPPRESS)
    parser.add_argument('legacy_model_path', nargs='?', help=argparse.SUPPRESS)
    parser.add_argument('--dataroot', help='dataset root to scan recursively')
    parser.add_argument('--model_path', help='official raw state_dict checkpoint')
    parser.add_argument(
        '--clip_path', default=str(ROOT / 'clip-vit-large-patch14'),
        help='local CLIP ViT-L/14 model directory')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args(argv)

    args.dataroot = args.dataroot or args.legacy_dataroot
    args.model_path = args.model_path or args.legacy_model_path
    if not args.dataroot:
        parser.error('--dataroot is required')
    if not args.model_path:
        parser.error('--model_path is required')
    if args.batch_size <= 0:
        parser.error('--batch_size must be positive')
    if args.num_workers < 0:
        parser.error('--num_workers cannot be negative')
    return args


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


def build_transform():
    return transforms.Compose([
        _TranslateDuplicate(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


def build_group_dataset(binary_leaves, transform):
    leaf_datasets = []
    for leaf in binary_leaves:
        dataset = datasets.ImageFolder(
            root=str(leaf), transform=transform, loader=pil_loader)
        expected_classes = {'0_real': 0, '1_fake': 1}
        if dataset.class_to_idx != expected_classes:
            raise ValueError(
                f'{leaf} must contain exactly 0_real/ and 1_fake/; '
                f'found {dataset.class_to_idx}')
        leaf_datasets.append(dataset)

    if len(leaf_datasets) == 1:
        return leaf_datasets[0]
    return ConcatDataset(leaf_datasets)


def load_official_model(model_path, clip_path, device):
    print(f'Loading official model: {model_path}')
    state_dict = torch.load(
        str(model_path), map_location='cpu', weights_only=True)
    if not isinstance(state_dict, Mapping):
        raise ValueError('official checkpoint must be a raw state_dict mapping')
    if 'model' in state_dict and isinstance(state_dict['model'], Mapping):
        raise ValueError(
            'received a train.py LoRA checkpoint; use test_checkpoint.py instead')

    model = C2P_CLIP(name=str(clip_path), num_classes=1)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print('Model loaded successfully.')
    return model


def infer_dataset(model, dataset, args, device):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    labels = []
    scores = []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            batch_scores = model(images).sigmoid().flatten().cpu().tolist()
            scores.extend(batch_scores)
            labels.extend(batch_labels.flatten().tolist())
    return labels, scores


def format_metrics(name, metrics):
    return (
        f'{name:20s} n={metrics["n"]:>7d}  '
        f'ACC={metrics["acc"]:6.2f}%  '
        f'Real_ACC={metrics["real_acc"]:6.2f}%  '
        f'Fake_ACC={metrics["fake_acc"]:6.2f}%  '
        f'AP={metrics["ap"]:6.2f}%'
    )


def main(argv=None):
    args = parse_args(argv)
    dataroot = resolve_existing_path(
        args.dataroot, 'dataset root', expect_directory=True)
    model_path = resolve_existing_path(
        args.model_path, 'model', expect_directory=False)
    clip_path = resolve_existing_path(
        args.clip_path, 'CLIP model', expect_directory=True)
    groups = discover_binary_groups(dataroot)
    device = resolve_device(args.gpu)

    print(f'Dataset: {dataroot}')
    print(f'Generators: {len(groups)}')
    print(f'Device: {device}')
    model = load_official_model(model_path, clip_path, device)
    transform = build_transform()

    all_labels = []
    all_scores = []
    group_metrics = []
    start_time = time.time()
    print('\n' + '=' * 92)
    for index, (group_name, binary_leaves) in enumerate(groups.items()):
        dataset = build_group_dataset(binary_leaves, transform)
        labels, scores = infer_dataset(model, dataset, args, device)
        metrics = compute_binary_metrics(labels, scores)
        group_metrics.append(metrics)
        all_labels.extend(labels)
        all_scores.extend(scores)
        print(f'[{index:02d}] ' + format_metrics(group_name, metrics))

    macro_metrics = {
        key: float(np.mean([metrics[key] for metrics in group_metrics]))
        for key in ('acc', 'real_acc', 'fake_acc', 'ap')
    }
    macro_metrics['n'] = sum(metrics['n'] for metrics in group_metrics)
    overall_metrics = compute_binary_metrics(all_labels, all_scores)
    elapsed = time.time() - start_time

    print('=' * 92)
    print('     ' + format_metrics('Macro mean', macro_metrics))
    print('     ' + format_metrics('Overall', overall_metrics))
    print(f'Elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)')


if __name__ == '__main__':
    main()
