import csv
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

from data.datasets import _TranslateDuplicate, pil_loader
from utils.binary_metrics import compute_binary_metrics
from utils.logit_distribution import compute_logit_stats


PREDICTION_FIELDS = (
    'generator',
    'path',
    'label',
    'raw_logit',
    'score',
)


def format_metrics(name, metrics):
    return (
        f'{name:20s} n={metrics["n"]:>7d}  '
        f'ACC={metrics["acc"]:6.2f}%  '
        f'Real_ACC={metrics["real_acc"]:6.2f}%  '
        f'Fake_ACC={metrics["fake_acc"]:6.2f}%  '
        f'AP={metrics["ap"]:6.2f}%'
    )


def format_diagnostics(metrics, logit_stats=None):
    text = (
        f'AUROC={metrics["roc_auc"]:6.2f}%  '
        f'ECE={metrics["ece"]:6.2f}%  '
        f'Brier={metrics["brier"]:.4f}'
    )
    if logit_stats is not None:
        text += (
            f'  Logit R={logit_stats["real_mean"]:.3f}'
            f'±{logit_stats["real_std"]:.3f}'
            f'  F={logit_stats["fake_mean"]:.3f}'
            f'±{logit_stats["fake_std"]:.3f}'
            f'  Sep={logit_stats["separation"]:.3f}'
        )
    return text


class PathImageFolder(datasets.ImageFolder):
    """ImageFolder that also returns the source path for every image."""

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.samples[index][0]
        return image, label, path


def build_transform(crop_size=224):
    """Build the deterministic CLIP preprocessing used for evaluation."""
    return transforms.Compose([
        _TranslateDuplicate(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


def build_group_dataset(binary_leaves, transform):
    """Concatenate sorted binary leaves without losing paths or labels."""
    leaves = sorted(
        (Path(leaf).expanduser().resolve() for leaf in binary_leaves),
        key=lambda path: path.as_posix(),
    )
    if not leaves:
        raise ValueError('at least one binary leaf is required')

    leaf_datasets = []
    for leaf in leaves:
        dataset = PathImageFolder(
            root=str(leaf),
            transform=transform,
            loader=pil_loader,
        )
        expected_classes = {'0_real': 0, '1_fake': 1}
        if dataset.class_to_idx != expected_classes:
            raise ValueError(
                f'{leaf} must contain exactly 0_real/ and 1_fake/; '
                f'found {dataset.class_to_idx}'
            )
        leaf_datasets.append(dataset)

    if len(leaf_datasets) == 1:
        return leaf_datasets[0]
    return ConcatDataset(leaf_datasets)


def evaluate_dataset(dataset, generator, forward_logits, device,
                     batch_size=64, num_workers=4):
    """Collect per-image logits from one deterministic image-only dataset."""
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if num_workers < 0:
        raise ValueError('num_workers cannot be negative')

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
    )
    predictions = []
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device, non_blocking=True)
            logits = forward_logits(images)
            if not isinstance(logits, torch.Tensor):
                raise TypeError('forward_logits must return a torch.Tensor')
            logits = logits.detach().flatten().cpu()
            labels = labels.detach().flatten().cpu()
            if logits.numel() != labels.numel():
                raise ValueError(
                    'forward_logits output size must match the label batch size'
                )
            scores = logits.sigmoid()
            for path, label, raw_logit, score in zip(
                    paths, labels.tolist(), logits.tolist(), scores.tolist()):
                predictions.append({
                    'generator': generator,
                    'path': str(Path(path).resolve()),
                    'label': int(label),
                    'raw_logit': float(raw_logit),
                    'score': float(score),
                })
    return predictions


def evaluate_groups(groups: Mapping[str, Sequence[Path]], forward_logits,
                    device, batch_size=64, num_workers=4, transform=None,
                    on_group_complete: Callable | None = None):
    """Evaluate every discovered generator using one shared data contract."""
    if not groups:
        raise ValueError('at least one generator group is required')
    transform = transform or build_transform()

    predictions = []
    group_metrics = {}
    group_logit_stats = {}
    for index, (generator, binary_leaves) in enumerate(groups.items()):
        dataset = build_group_dataset(binary_leaves, transform)
        group_predictions = evaluate_dataset(
            dataset,
            generator=generator,
            forward_logits=forward_logits,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        metrics = compute_binary_metrics(
            [record['label'] for record in group_predictions],
            [record['score'] for record in group_predictions],
        )
        logit_stats = compute_logit_stats(
            [
                record['raw_logit'] for record in group_predictions
                if record['label'] == 0
            ],
            [
                record['raw_logit'] for record in group_predictions
                if record['label'] == 1
            ],
        )
        group_metrics[generator] = metrics
        group_logit_stats[generator] = logit_stats
        predictions.extend(group_predictions)
        if on_group_complete is not None:
            on_group_complete(index, generator, metrics, logit_stats)

    macro_metrics = {
        key: float(np.mean([
            metrics[key] for metrics in group_metrics.values()
        ]))
        for key in (
            'acc', 'real_acc', 'fake_acc', 'ap',
            'roc_auc', 'ece', 'brier',
        )
    }
    macro_metrics['n'] = sum(
        metrics['n'] for metrics in group_metrics.values()
    )
    overall_metrics = compute_binary_metrics(
        [record['label'] for record in predictions],
        [record['score'] for record in predictions],
    )
    overall_logit_stats = compute_logit_stats(
        [
            record['raw_logit'] for record in predictions
            if record['label'] == 0
        ],
        [
            record['raw_logit'] for record in predictions
            if record['label'] == 1
        ],
    )
    return {
        'group_metrics': group_metrics,
        'group_logit_stats': group_logit_stats,
        'macro_metrics': macro_metrics,
        'overall_metrics': overall_metrics,
        'overall_logit_stats': overall_logit_stats,
        'predictions': predictions,
    }


def write_predictions_csv(predictions, output_path):
    """Write the shared per-image prediction format."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(predictions)
    return output_path
