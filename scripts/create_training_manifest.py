"""Create a portable, fixed-order training subset manifest."""

import argparse
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.datasets import IMG_EXTENSIONS


def discover_images(root):
    extensions = {extension.lower() for extension in IMG_EXTENSIONS}
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob('*')
        if path.is_file() and path.suffix.lower() in extensions
    )


def main():
    parser = argparse.ArgumentParser(
        description='Create a seeded fixed-order image manifest.')
    parser.add_argument('--dataroot', required=True, help='training image root')
    parser.add_argument('--output', required=True, help='new manifest path')
    parser.add_argument('--data_seed', required=True, type=int)
    parser.add_argument('--count', required=True, type=int)
    args = parser.parse_args()

    root = Path(args.dataroot).resolve()
    output = Path(args.output).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'training root not found: {root}')
    if output.exists():
        raise FileExistsError(
            f'refusing to overwrite existing manifest: {output}')

    images = discover_images(root)
    if args.count <= 0 or args.count > len(images):
        raise ValueError(
            f'count must be in [1, {len(images)}], got {args.count}')

    generator = torch.Generator().manual_seed(args.data_seed)
    indices = torch.randperm(len(images), generator=generator)[:args.count]
    selected = [images[index] for index in indices.tolist()]

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# c2p-training-manifest-v1',
        f'# data_seed={args.data_seed}',
        f'# source_count={len(images)}',
        f'# selected_count={len(selected)}',
        *selected,
    ]
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f'manifest={output}')
    print(f'sha256={digest}')
    print(f'source_count={len(images)} selected_count={len(selected)}')


if __name__ == '__main__':
    main()
