from collections.abc import Mapping


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
    if requested not in ('auto', 'concat', 'residual_gate'):
        raise ValueError(
            "local fusion must be 'auto', 'concat', or 'residual_gate', "
            f'got {requested}')
    if not use_local_features:
        return 'concat' if requested == 'auto' else requested

    has_residual_gate = (
        'local_gate_logit' in state_dict
        or any(key.startswith('local_classifier.') for key in state_dict)
    )
    has_concat_head = any(key.startswith('model.fc.0.') for key in state_dict)

    if has_residual_gate and has_concat_head:
        raise ValueError(
            'checkpoint contains both concat and residual-gate local heads')
    if has_residual_gate:
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
