from collections.abc import Mapping
from pathlib import Path

import torch


_RRSD_STATE_KEYS = frozenset({
    'rrsd.head.0.weight',
    'rrsd.head.0.bias',
    'rrsd.head.2.weight',
    'rrsd.head.2.bias',
    'rrsd.gate',
    'rrsd.real_prototype',
    'rrsd.real_count',
    'rrsd.max_delta',
})


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


def infer_rrsd_max_correction(state_dict):
    """Infer the optional RRSD architecture from a normalized state dict."""
    present = _RRSD_STATE_KEYS.intersection(state_dict)
    prefixed = {key for key in state_dict if key.startswith('rrsd.')}
    if not present and not prefixed:
        return 0.0
    missing = _RRSD_STATE_KEYS - present
    unexpected = prefixed - _RRSD_STATE_KEYS
    if missing or unexpected:
        details = []
        if missing:
            details.append('missing: ' + ', '.join(sorted(missing)))
        if unexpected:
            details.append('unexpected: ' + ', '.join(sorted(unexpected)))
        raise ValueError(
            'incomplete or incompatible RRSD checkpoint state ('
            + '; '.join(details)
            + ')'
        )
    max_correction = state_dict['rrsd.max_delta']
    if not torch.is_tensor(max_correction) or max_correction.numel() != 1:
        raise ValueError('RRSD max_delta must be a scalar tensor')
    value = float(max_correction.item())
    if value <= 0:
        raise ValueError('RRSD max_delta must be positive')
    return value


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
    rrsd_max_correction = infer_rrsd_max_correction(state_dict)
    model = CLIPModel_lora(
        name=str(clip_path),
        num_classes=1,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        rrsd_max_correction=rrsd_max_correction,
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            'checkpoint does not match the inferred self-trained '
            'architecture. Verify lora_r, lora_alpha and lora_dropout; '
            'retired experimental checkpoints require an older Git revision.'
        ) from error

    model.to(device)
    model.eval()
    return model, total_steps
