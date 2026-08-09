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


def residual_vib_dim(state_dict):
    """Return the latent size encoded by an RVIB checkpoint, if present."""
    weight = state_dict.get('residual_vib.mu.weight')
    if weight is None:
        return None
    if not hasattr(weight, 'shape') or len(weight.shape) != 2:
        raise ValueError('invalid residual_vib.mu.weight in checkpoint')
    return int(weight.shape[0])


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
    vib_dim = residual_vib_dim(state_dict)

    model = CLIPModel_lora(
        name=str(clip_path),
        num_classes=1,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        residual_vib=vib_dim is not None,
        vib_dim=vib_dim or 64,
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            'checkpoint does not match the inferred self-trained '
            'architecture. Verify lora_r, lora_alpha and lora_dropout; '
            'retired PRH/SPH checkpoints require an older Git revision.'
        ) from error

    model.to(device)
    model.eval()
    return model, total_steps
