"""Evaluate one or more self-trained C2P-CLIP LoRA checkpoints.

The checkpoint must be produced by scripts/train.py and contain ``model`` and
``total_steps``. Evaluation is image-only: captions and text prompts are not
loaded.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from utils.binary_dataset_layout import discover_binary_groups
from utils.binary_evaluation import (
    average_prediction_sets,
    evaluate_groups,
    format_diagnostics,
    format_metrics,
    summarize_predictions,
    write_predictions_csv,
)
from utils.checkpoint_loading import load_self_trained_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Test a self-trained C2P-CLIP LoRA checkpoint')
    parser.add_argument(
        '--dataroot',
        required=True,
        help='binary dataset root with direct or nested 0_real/1_fake folders',
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        nargs='+',
        help=(
            'one or more train.py .pth files; multiple checkpoints are '
            'combined by uniform raw-logit averaging'
        ),
    )
    parser.add_argument(
        '--clip_path',
        default=str(ROOT / 'clip-vit-large-patch14'),
        help='local CLIP ViT-L/14 model directory',
    )
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument(
        '--predictions_csv',
        help='optional CSV path for per-image raw logits and scores',
    )
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    args = parser.parse_args(argv)

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


def load_checkpoint(
    checkpoint_path,
    clip_path,
    lora_r,
    lora_alpha,
    lora_dropout,
    device,
):
    print(f'Loading checkpoint: {checkpoint_path}')
    model, total_steps = load_self_trained_checkpoint(
        checkpoint_path=checkpoint_path,
        clip_path=clip_path,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        device=device,
    )
    steps = total_steps if total_steps is not None else 'unknown'
    print(f'  Total training steps: {steps}')
    print('  Model loaded successfully.')
    return model


def lora_forward_logits(model, images):
    return model(images, None, None, cla=True)


def print_aggregate_summary(summary):
    print('     ' + format_metrics('Macro mean', summary['macro_metrics']))
    print('     ' + format_diagnostics(summary['macro_metrics']))
    print('     ' + format_metrics('Overall', summary['overall_metrics']))
    print('     ' + format_diagnostics(
        summary['overall_metrics'], summary['overall_logit_stats']))


def print_group_summary(summary):
    for index, generator in enumerate(summary['group_metrics']):
        metrics = summary['group_metrics'][generator]
        stats = summary['group_logit_stats'][generator]
        print(f'[{index:02d}] ' + format_metrics(generator, metrics))
        print('     ' + format_diagnostics(metrics, stats))


def main(argv=None):
    args = parse_args(argv)
    dataroot = resolve_existing_path(
        args.dataroot, 'dataset root', expect_directory=True)
    checkpoints = [
        resolve_existing_path(path, 'checkpoint', expect_directory=False)
        for path in args.checkpoint
    ]
    clip_path = resolve_existing_path(
        args.clip_path, 'CLIP model', expect_directory=True)
    groups = discover_binary_groups(dataroot)
    device = resolve_device(args.gpu)

    print(f'Dataset: {dataroot}')
    print(f'Generators: {len(groups)}')
    print(f'Device: {device}')
    print(f'Checkpoints: {len(checkpoints)}')
    start_time = time.time()
    prediction_sets = []
    summary = None
    for model_index, checkpoint in enumerate(checkpoints, start=1):
        print('\n' + '=' * 92)
        print(f'Model {model_index}/{len(checkpoints)}: {checkpoint}')
        model = load_checkpoint(
            checkpoint_path=checkpoint,
            clip_path=clip_path,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            device=device,
        )
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
        print('-' * 92)
        print_aggregate_summary(summary)
        prediction_sets.append(summary['predictions'])
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if len(prediction_sets) > 1:
        ensemble_predictions = average_prediction_sets(prediction_sets)
        summary = summarize_predictions(ensemble_predictions)
        print('\n' + '=' * 92)
        print('Uniform raw-logit ensemble')
        print_group_summary(summary)
        print('-' * 92)
        print_aggregate_summary(summary)

    elapsed = time.time() - start_time
    if args.predictions_csv:
        output_path = write_predictions_csv(
            summary['predictions'], args.predictions_csv)
        print(f'Predictions: {output_path}')
    print(f'Elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)')


if __name__ == '__main__':
    main()
