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

import torch

from scripts.inference import C2P_CLIP
from utils.binary_dataset_layout import discover_binary_groups
from utils.binary_evaluation import (
    build_group_dataset,
    build_transform,
    evaluate_groups,
    format_diagnostics,
    format_metrics,
    write_predictions_csv,
)


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
    parser.add_argument(
        '--predictions_csv',
        help='optional CSV path for per-image raw logits and scores',
    )
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


def official_forward_logits(model, images):
    return model(images)


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
    start_time = time.time()
    print('\n' + '=' * 92)
    summary = evaluate_groups(
        groups,
        forward_logits=lambda images: official_forward_logits(model, images),
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        on_group_complete=lambda index, name, metrics, stats: (
            print(f'[{index:02d}] ' + format_metrics(name, metrics)),
            print('     ' + format_diagnostics(metrics, stats)),
        ),
    )
    elapsed = time.time() - start_time

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
