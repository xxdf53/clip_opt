"""Create a fixed, batch-balanced G10 training subset manifest."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.create_training_manifest import discover_images


MANIFEST_SCHEMA = 'c2p-g10-stratified-manifest-v1'
DEFAULT_CATEGORIES = ('car', 'cat', 'chair', 'horse')
DEFAULT_LABELS = ('0_real', '1_fake')


def _stable_key(data_seed, namespace, value):
    payload = f'{data_seed}\0{namespace}\0{value}'.encode('utf-8')
    return hashlib.sha256(payload).digest()


def discover_strata(dataroot, categories, labels):
    dataroot = Path(dataroot).resolve()
    strata = {}
    for category in categories:
        for label in labels:
            stratum = f'{category}/{label}'
            stratum_root = dataroot / category / label
            if not stratum_root.is_dir():
                raise FileNotFoundError(
                    f'G10 stratum directory not found: {stratum_root}')
            strata[stratum] = [
                f'{stratum}/{relative_path}'
                for relative_path in discover_images(stratum_root)
            ]
    return strata


def select_batch_balanced_paths(
    strata,
    per_stratum,
    batch_size,
    data_seed,
):
    if per_stratum <= 0:
        raise ValueError('--per_stratum must be positive')
    if batch_size <= 0:
        raise ValueError('--batch_size must be positive')

    stratum_names = sorted(strata)
    stratum_count = len(stratum_names)
    if not stratum_count:
        raise ValueError('at least one stratum is required')
    if batch_size % stratum_count != 0:
        raise ValueError(
            f'batch_size {batch_size} must be divisible by '
            f'{stratum_count} strata')

    per_batch = batch_size // stratum_count
    if per_stratum % per_batch != 0:
        raise ValueError(
            f'per_stratum {per_stratum} must be divisible by '
            f'{per_batch} samples per stratum per batch')

    selected_by_stratum = {}
    for stratum in stratum_names:
        paths = strata[stratum]
        if len(paths) < per_stratum:
            raise ValueError(
                f'stratum {stratum} has {len(paths)} images, '
                f'but {per_stratum} were requested')
        ranked = sorted(
            paths,
            key=lambda path: _stable_key(
                data_seed, f'select:{stratum}', path),
        )
        selected_by_stratum[stratum] = ranked[:per_stratum]

    total_steps = per_stratum // per_batch
    ordered_paths = []
    for step in range(total_steps):
        batch = []
        start = step * per_batch
        stop = start + per_batch
        for stratum in stratum_names:
            batch.extend(selected_by_stratum[stratum][start:stop])
        batch.sort(
            key=lambda path: _stable_key(
                data_seed, f'batch:{step}', path),
        )
        ordered_paths.extend(batch)

    if len(ordered_paths) != len(set(ordered_paths)):
        raise ValueError('selected G10 manifest paths are not unique')
    return ordered_paths, selected_by_stratum, per_batch, total_steps


def validate_selected_captions(paths, textroot):
    textroot = Path(textroot).resolve()
    missing = []
    empty = []
    unreadable = []
    for relative_path in paths:
        caption_path = (textroot / relative_path).with_suffix('.txt')
        if not caption_path.is_file():
            missing.append(caption_path)
            continue
        try:
            caption = caption_path.read_text(encoding='utf-8')
        except (OSError, UnicodeError):
            unreadable.append(caption_path)
            continue
        if not caption.strip():
            empty.append(caption_path)

    if missing or unreadable or empty:
        examples = [*missing[:1], *unreadable[:1], *empty[:1]]
        preview = ', '.join(str(path) for path in examples)
        raise ValueError(
            'selected captions failed validation: '
            f'missing={len(missing)} unreadable={len(unreadable)} '
            f'empty={len(empty)}; examples: {preview}')


def write_manifest_bundle(
    dataroot,
    textroot,
    output,
    metadata_output,
    categories,
    labels,
    per_stratum,
    batch_size,
    data_seed,
):
    dataroot = Path(dataroot).resolve()
    textroot = Path(textroot).resolve()
    output = Path(output).resolve()
    metadata_output = Path(metadata_output).resolve()
    if not dataroot.is_dir():
        raise FileNotFoundError(f'training root not found: {dataroot}')
    if not textroot.is_dir():
        raise FileNotFoundError(f'caption root not found: {textroot}')
    for path in (output, metadata_output):
        if path.exists():
            raise FileExistsError(f'refusing to overwrite existing file: {path}')

    strata = discover_strata(dataroot, categories, labels)
    ordered, selected, per_batch, total_steps = select_batch_balanced_paths(
        strata,
        per_stratum=per_stratum,
        batch_size=batch_size,
        data_seed=data_seed,
    )
    validate_selected_captions(ordered, textroot)

    source_counts = {
        stratum: len(paths) for stratum, paths in sorted(strata.items())
    }
    selected_counts = {
        stratum: len(paths) for stratum, paths in sorted(selected.items())
    }
    lines = [
        '# c2p-training-manifest-v1',
        f'# schema={MANIFEST_SCHEMA}',
        f'# data_seed={data_seed}',
        f'# selected_count={len(ordered)}',
        f'# batch_size={batch_size}',
        f'# total_steps={total_steps}',
        f'# strata={len(strata)}',
        f'# per_stratum={per_stratum}',
        f'# per_stratum_per_batch={per_batch}',
        *ordered,
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    manifest_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()

    metadata = {
        'schema': MANIFEST_SCHEMA,
        'data_seed': data_seed,
        'dataroot': str(dataroot),
        'textroot': str(textroot),
        'categories': list(categories),
        'labels': list(labels),
        'source_counts': source_counts,
        'selected_counts': selected_counts,
        'selected_count': len(ordered),
        'batch_size': batch_size,
        'total_steps': total_steps,
        'strata': len(strata),
        'per_stratum': per_stratum,
        'per_stratum_per_batch': per_batch,
        'selected_caption_count': len(ordered),
        'missing_caption_count': 0,
        'manifest': str(output),
        'manifest_sha256': manifest_sha256,
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    metadata_sha256 = hashlib.sha256(
        metadata_output.read_bytes()).hexdigest()
    return metadata, manifest_sha256, metadata_sha256


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Create a deterministic category/authenticity-stratified G10 '
            'training manifest with balanced global batches.'))
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--textroot', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--metadata_output')
    parser.add_argument('--data_seed', required=True, type=int)
    parser.add_argument('--categories', nargs='+', default=DEFAULT_CATEGORIES)
    parser.add_argument('--labels', nargs='+', default=DEFAULT_LABELS)
    parser.add_argument('--per_stratum', type=int, default=1600)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    output = Path(args.output)
    metadata_output = (
        Path(args.metadata_output)
        if args.metadata_output
        else output.with_suffix('.json')
    )
    metadata, manifest_sha256, metadata_sha256 = write_manifest_bundle(
        dataroot=args.dataroot,
        textroot=args.textroot,
        output=output,
        metadata_output=metadata_output,
        categories=args.categories,
        labels=args.labels,
        per_stratum=args.per_stratum,
        batch_size=args.batch_size,
        data_seed=args.data_seed,
    )
    print(f'manifest={Path(args.output).resolve()}')
    print(f'manifest_sha256={manifest_sha256}')
    print(f'metadata={metadata_output.resolve()}')
    print(f'metadata_sha256={metadata_sha256}')
    print(
        f'selected_count={metadata["selected_count"]} '
        f'total_steps={metadata["total_steps"]} '
        f'batch_size={metadata["batch_size"]}')
    for stratum, count in metadata['selected_counts'].items():
        print(
            f'{stratum}: source={metadata["source_counts"][stratum]} '
            f'selected={count}')
    print(
        f'missing_captions={metadata["missing_caption_count"]} '
        f'balanced_batch_samples_per_stratum='
        f'{metadata["per_stratum_per_batch"]}')


if __name__ == '__main__':
    main()
