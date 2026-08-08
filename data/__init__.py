import torch
import numpy as np
from pathlib import Path
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import dataset_folder
from utils.data_loading import should_drop_last_batch

import os


def _dataset_paths(dataset):
    """Return sample paths in the same global index order as a dataset."""
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        paths = []
        for child in dataset.datasets:
            paths.extend(_dataset_paths(child))
        return paths
    if hasattr(dataset, 'samples'):
        return [sample[0] for sample in dataset.samples]
    raise TypeError(
        'training manifests require datasets exposing .samples or '
        'ConcatDataset children that expose .samples')


def _read_training_manifest(dataset, training_root, manifest_path):
    """Map a portable relative-path manifest to dataset indices."""
    training_root = Path(training_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f'training manifest not found: {manifest_path}')

    entries = []
    for raw_line in manifest_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if line and not line.startswith('#'):
            entries.append(Path(line).as_posix())
    if not entries:
        raise ValueError(f'training manifest has no image entries: {manifest_path}')
    if len(entries) != len(set(entries)):
        raise ValueError(f'training manifest contains duplicate paths: {manifest_path}')

    path_to_index = {}
    for index, sample_path in enumerate(_dataset_paths(dataset)):
        relative_path = Path(sample_path).resolve().relative_to(training_root)
        path_to_index[relative_path.as_posix()] = index

    missing = [entry for entry in entries if entry not in path_to_index]
    if missing:
        preview = ', '.join(missing[:3])
        raise FileNotFoundError(
            f'{len(missing)} manifest images are absent from {training_root}; '
            f'first entries: {preview}')
    return [path_to_index[entry] for entry in entries]


def get_dataset(opt):
    classes = os.listdir(opt.dataroot) if len(opt.classes) == 0 else opt.classes
    if '0_real' not in classes or '1_fake' not in classes:
        dset_lst = []
        for cls in classes:
            root = os.path.join(opt.dataroot, cls)
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
        return torch.utils.data.ConcatDataset(dset_lst)
    return dataset_folder(opt, opt.dataroot)

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def collate(batch):
    collated = (
        [item[0] for item in batch],
        torch.stack([item[1] for item in batch]),
        [item[2] for item in batch],
        torch.stack([item[3] for item in batch]),
        torch.stack([item[4] for item in batch]),
        torch.tensor([item[5] for item in batch]),
    )
    if len(batch[0]) == 7:
        collated += (torch.tensor([item[6] for item in batch]),)
    return collated


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    manifest_path = getattr(opt, 'train_manifest', '')
    data_seed = getattr(opt, 'data_seed', None)
    if opt.isTrain and manifest_path:
        if opt.class_bal:
            raise ValueError('--train_manifest cannot be combined with --class_bal')
        if data_seed is None:
            raise ValueError('--train_manifest requires --data_seed')
        indices = _read_training_manifest(dataset, opt.dataroot, manifest_path)
        dataset = torch.utils.data.Subset(dataset, indices)
        shuffle = False
        sampler = None

    generator = None
    if opt.isTrain and data_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(data_seed)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              drop_last=should_drop_last_batch(
                                                  opt.isTrain,
                                                  getattr(opt, 'keep_last_batch', False),
                                              ),
                                              num_workers=int(opt.num_threads),
                                              collate_fn=collate,
                                              generator=generator)
    return data_loader
