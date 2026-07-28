"""Text construction helpers for Counterfactual Prompt Decomposition."""


def cpd_is_enabled(opt):
    """Return whether either CPD training objective is active."""
    return (
        float(getattr(opt, 'cpd_direction_weight', 0.0)) > 0.0
        or float(getattr(opt, 'cpd_content_weight', 0.0)) > 0.0
    )


def cpd_schedule_scale(step, start_step=0, warmup_steps=0):
    """Return the CPD multiplier for a delayed linear warmup."""
    if step < 0:
        raise ValueError('CPD schedule step cannot be negative')
    if start_step < 0:
        raise ValueError('--cpd_start_step cannot be negative')
    if warmup_steps < 0:
        raise ValueError('--cpd_warmup_steps cannot be negative')
    if step <= start_step:
        return 0.0
    if warmup_steps == 0:
        return 1.0
    return min((step - start_step) / warmup_steps, 1.0)


def split_category_prompts(cates):
    """Return fake and real prompt strings using the C2P category convention."""
    if len(cates) < 2 or len(cates) % 2 != 0:
        raise ValueError(
            '--cates must contain equally sized fake and real prompt halves')
    midpoint = len(cates) // 2
    fake_prompt = ' '.join(cates[:midpoint]).strip()
    real_prompt = ' '.join(cates[midpoint:]).strip()
    if not fake_prompt or not real_prompt:
        raise ValueError('fake and real category prompts cannot be empty')
    return fake_prompt, real_prompt


def _prompted_caption(prompt, caption):
    caption = caption.strip()
    return f'{prompt}. {caption} {prompt}.'


def build_counterfactual_captions(caption, cates):
    """Build texts in binary-label order: real (0), then fake (1)."""
    fake_prompt, real_prompt = split_category_prompts(cates)
    return (
        _prompted_caption(real_prompt, caption),
        _prompted_caption(fake_prompt, caption),
    )


def build_label_caption(caption, cates, target):
    """Build the original C2P text corresponding to a binary target."""
    if int(target) not in (0, 1):
        raise ValueError(f'binary target must be 0 or 1, got {target}')
    counterfactuals = build_counterfactual_captions(caption, cates)
    return counterfactuals[int(target)]
