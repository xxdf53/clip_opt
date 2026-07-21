from collections.abc import Mapping


LOCAL_FUSIONS = (
    'concat', 'residual_gate', 'adaptive_residual', 'bounded_residual')


def extract_training_state_dict(payload):
    """Extract a train.py state_dict and remove leading DataParallel prefixes."""
    if not isinstance(payload, Mapping) or 'model' not in payload:
        raise ValueError("checkpoint is missing a 'model' state_dict")

    state_dict = payload['model']
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint 'model' must be a state_dict mapping")

    normalized = {
        key[len('module.'):] if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }
    return normalized, payload.get('total_steps')


def resolve_local_fusion(state_dict, requested='auto', use_local_features=False):
    """Resolve local fusion from checkpoint keys and reject mismatches early."""
    valid = ('auto',) + LOCAL_FUSIONS
    if requested not in valid:
        raise ValueError(
            f'local fusion must be one of {valid}, got {requested}')
    if not use_local_features:
        if requested not in ('auto', 'concat'):
            raise ValueError(
                f"--local_fusion '{requested}' requires --use_local_features")
        return 'concat'

    has_residual_gate = 'local_gate_logit' in state_dict
    has_adaptive_gate = any(
        key.startswith('local_gate_network.') for key in state_dict)
    has_bounded_alpha = 'residual_alpha' in state_dict
    has_bounded_scale = 'residual_scale' in state_dict
    if has_bounded_alpha != has_bounded_scale:
        raise ValueError(
            'bounded residual checkpoint must contain residual_alpha and '
            'residual_scale')
    has_bounded_residual = has_bounded_alpha and has_bounded_scale
    has_concat_head = any(key.startswith('model.fc.0.') for key in state_dict)

    detected_heads = sum((
        has_concat_head,
        has_residual_gate,
        has_adaptive_gate,
        has_bounded_residual,
    ))
    if detected_heads > 1:
        raise ValueError(
            'checkpoint contains multiple incompatible local fusion heads')
    if has_bounded_residual:
        detected = 'bounded_residual'
    elif has_adaptive_gate:
        detected = 'adaptive_residual'
    elif has_residual_gate:
        detected = 'residual_gate'
    elif has_concat_head:
        detected = 'concat'
    else:
        raise ValueError(
            'unable to infer local fusion from checkpoint keys; make sure '
            '--use_local_features matches the checkpoint')

    if requested != 'auto' and requested != detected:
        raise ValueError(
            f"checkpoint uses local_fusion='{detected}', but "
            f"--local_fusion '{requested}' was requested")
    return detected


def parse_gate_override(value):
    """Parse `learned` or a fixed gate in [0, 1] for evaluation."""
    if value is None or str(value).lower() == 'learned':
        return None
    try:
        gate = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("gate override must be 'learned' or a number") from error
    if not 0.0 <= gate <= 1.0:
        raise ValueError('gate override must be in [0, 1]')
    return gate


def select_baseline_initialization_state(source_state, target_state):
    """Select shape-compatible non-local weights from a baseline checkpoint."""
    local_prefixes = (
        'local_norm.',
        'local_projector.',
        'local_classifier.',
        'local_gate_logit',
        'local_gate_network.',
        'residual_alpha',
        'residual_scale',
    )
    if any(key.startswith(local_prefixes) for key in source_state):
        raise ValueError('initialization checkpoint must be a non-local baseline')

    compatible = {
        key: value
        for key, value in source_state.items()
        if (key.startswith('model.fc.') or 'lora_' in key)
        and key in target_state
        and getattr(value, 'shape', None) == getattr(target_state[key], 'shape', None)
    }
    required = {'model.fc.weight', 'model.fc.bias'}
    missing = sorted(required.difference(compatible))
    if missing:
        raise ValueError(
            'baseline initialization is missing compatible global head keys: '
            + ', '.join(missing))
    if not any('lora_' in key for key in compatible):
        raise ValueError('baseline initialization contains no compatible LoRA weights')
    return compatible
