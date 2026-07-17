from pathlib import Path


def discover_binary_groups(root):
    """Group paired 0_real/1_fake leaf directories by top-level generator."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'dataset root not found: {root}')

    groups = {}
    directories = [root]
    directories.extend(sorted(
        (path for path in root.rglob('*') if path.is_dir()),
        key=lambda path: path.as_posix(),
    ))

    for directory in directories:
        child_names = {
            child.name for child in directory.iterdir() if child.is_dir()
        }
        has_real = '0_real' in child_names
        has_fake = '1_fake' in child_names
        if has_real != has_fake:
            missing = '1_fake' if has_real else '0_real'
            raise ValueError(f'{directory} is missing {missing}')
        if not has_real:
            continue

        if directory == root:
            group_name = root.name
        else:
            group_name = directory.relative_to(root).parts[0]
        groups.setdefault(group_name, []).append(directory.resolve())

    if not groups:
        raise ValueError(f'no paired 0_real/1_fake directories found under {root}')

    return {
        group_name: sorted(paths, key=lambda path: path.as_posix())
        for group_name, paths in sorted(groups.items())
    }
