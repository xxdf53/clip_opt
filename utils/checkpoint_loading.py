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
