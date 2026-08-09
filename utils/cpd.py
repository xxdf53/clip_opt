"""Configuration and text helpers for Counterfactual Prompt Decomposition."""

from utils.captions import split_category_prompts


def cpd_is_enabled(opt):
    """Return whether either CPD training objective is active."""
    return (
        float(getattr(opt, 'cpd_direction_weight', 0.0)) > 0.0
        or float(getattr(opt, 'cpd_content_weight', 0.0)) > 0.0
    )


def cpd_schedule_scale(step, start_step=0, warmup_steps=0):
    """Return the delayed linear-warmup multiplier for CPD."""
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


def build_counterfactual_captions(caption, cates):
    """Build texts in binary-label order: real (0), then fake (1)."""
    fake_prompt, real_prompt = split_category_prompts(cates)
    caption = caption.strip()
    return (
        f'{real_prompt}. {caption} {real_prompt}.',
        f'{fake_prompt}. {caption} {fake_prompt}.',
    )
