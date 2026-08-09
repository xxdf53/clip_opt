from collections.abc import Mapping
from pathlib import Path

import torch


def extract_training_state_dict(payload):
    """Return a train.py state dict without a leading DataParallel prefix."""
    if not isinstance(payload, Mapping) or 'model' not in payload:
        raise ValueError("checkpoint is missing a 'model' state_dict")

    state_dict = payload['model']
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint 'model' must be a state_dict mapping")

    normalized = {
        key.removeprefix('module.'): value
        for key, value in state_dict.items()
    }
    return normalized, payload.get('total_steps')


def has_patch_residual_head(state_dict):
    """Detect PRH checkpoints without changing the legacy file format."""
    return any(
        key.startswith('patch_residual_head.')
        for key in state_dict
    )


def has_symmetric_prototype_head(state_dict):
    """Detect SPH checkpoints from their prototype parameters."""
    required_keys = {
        'model.fc.real_prototype',
        'model.fc.fake_prototype',
    }
    return required_keys.issubset(state_dict)


def load_self_trained_checkpoint(
    checkpoint_path,
    clip_path,
    lora_r,
    lora_alpha,
    lora_dropout,
    device,
):
    """Build and strictly load a self-trained LoRA checkpoint."""
    from networks.trainer import CLIPModel_lora

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')

    payload = torch.load(
        str(checkpoint_path),
        map_location='cpu',
        weights_only=True,
    )
    state_dict, total_steps = extract_training_state_dict(payload)
    patch_residual_head = has_patch_residual_head(state_dict)
    symmetric_prototype_head = has_symmetric_prototype_head(state_dict)
    if patch_residual_head and symmetric_prototype_head:
        raise ValueError('checkpoint cannot contain both PRH and SPH heads')

    model = CLIPModel_lora(
        name=str(clip_path),
        num_classes=1,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        patch_residual_head=patch_residual_head,
        symmetric_prototype_head=symmetric_prototype_head,
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            'checkpoint does not match the inferred self-trained '
            'architecture. Verify lora_r, lora_alpha and lora_dropout; '
            'retired local-feature checkpoints require an older Git revision.'
        ) from error

    model.to(device)
    model.eval()
    return model, total_steps
