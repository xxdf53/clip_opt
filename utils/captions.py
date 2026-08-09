"""Caption construction used by the original C2P-CLIP training path."""


def split_category_prompts(cates):
    if len(cates) < 2 or len(cates) % 2 != 0:
        raise ValueError(
            '--cates must contain equally sized fake and real prompt halves')
    midpoint = len(cates) // 2
    fake_prompt = ' '.join(cates[:midpoint]).strip()
    real_prompt = ' '.join(cates[midpoint:]).strip()
    if not fake_prompt or not real_prompt:
        raise ValueError('fake and real category prompts cannot be empty')
    return fake_prompt, real_prompt


def build_label_caption(caption, cates, target):
    if int(target) not in (0, 1):
        raise ValueError(f'binary target must be 0 or 1, got {target}')
    fake_prompt, real_prompt = split_category_prompts(cates)
    prompt = real_prompt if int(target) == 0 else fake_prompt
    return f'{prompt}. {caption.strip()} {prompt}.'
