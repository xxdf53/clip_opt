"""Training-only objectives retained after experimental cleanup."""

import torch
import torch.nn.functional as F


FAKE_REWEIGHTING_MODES = ('hard', 'random', 'uniform')


def _validate_feature_pair(first, second, names):
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError(f'{names} must both have shape [batch, features]')
    if first.shape != second.shape:
        raise ValueError(f'{names} must have identical shapes')


def counterfactual_prompt_components(real_embeddings, fake_embeddings):
    """Split paired prompts into authenticity direction and content center."""
    _validate_feature_pair(
        real_embeddings,
        fake_embeddings,
        'real/fake text embeddings',
    )
    real_embeddings = F.normalize(real_embeddings, p=2, dim=-1)
    fake_embeddings = F.normalize(fake_embeddings, p=2, dim=-1)
    direction = F.normalize(fake_embeddings - real_embeddings, p=2, dim=-1)
    center = F.normalize(0.5 * (fake_embeddings + real_embeddings), p=2, dim=-1)
    return direction, center


def cpd_direction_loss(image_residual, direction, labels, margin=0.1):
    """Align the LoRA residual with the label-conditioned prompt direction."""
    if margin < 0:
        raise ValueError(f'CPD direction margin cannot be negative: {margin}')
    _validate_feature_pair(
        image_residual,
        direction,
        'image residual/authenticity direction',
    )
    labels = labels.flatten().to(dtype=image_residual.dtype)
    if labels.numel() != image_residual.shape[0]:
        raise ValueError('labels must contain one value per image residual')
    label_sign = labels.mul(2.0).sub(1.0)
    projection = (image_residual * direction.detach()).sum(dim=-1)
    return F.softplus(margin - label_sign * projection).mean()


def cpd_content_rejection_loss(image_residual, content_center):
    """Discourage the task residual from following shared image content."""
    _validate_feature_pair(
        image_residual,
        content_center,
        'image residual/content center',
    )
    alignment = (image_residual * content_center.detach()).sum(dim=-1)
    return alignment.square().mean()


def cpd_diagnostics(
    image_residual,
    direction,
    content_center,
    labels,
    prompt_gap,
):
    """Return detached CPD observables for training logs."""
    labels = labels.flatten().to(dtype=image_residual.dtype)
    label_sign = labels.mul(2.0).sub(1.0)
    projection = (
        image_residual.detach() * direction.detach()
    ).sum(dim=-1)
    content_alignment = (
        image_residual.detach() * content_center.detach()
    ).sum(dim=-1).abs()
    return {
        'cpd_signed_projection': (label_sign * projection).mean(),
        'cpd_content_alignment': content_alignment.mean(),
        'cpd_prompt_gap': prompt_gap.detach().flatten().mean(),
    }


def _flatten_binary_inputs(logits, labels):
    logits = logits.flatten()
    labels = labels.flatten().to(dtype=logits.dtype)
    if logits.numel() != labels.numel():
        raise ValueError('logits and labels must contain the same samples')
    return logits, labels


def _hard_class_reweighting_loss(
    logits,
    labels,
    *,
    fraction,
    select_fake,
):
    class_name = 'fake' if select_fake else 'real'
    if not 0 < fraction < 1:
        raise ValueError(
            f'hard-{class_name} fraction must be in (0, 1), got {fraction}')

    logits, labels = _flatten_binary_inputs(logits, labels)
    class_mask = labels >= 0.5 if select_fake else labels < 0.5
    class_indices = torch.nonzero(class_mask, as_tuple=False).flatten()
    class_count = class_indices.numel()
    selected_count = int(fraction * class_count)
    diagnostic_prefix = f'hard_{class_name}'
    zero = logits.sum() * 0.0
    diagnostics = {
        f'{diagnostic_prefix}_selected': logits.new_tensor(
            float(selected_count)),
        f'{diagnostic_prefix}_total': logits.new_tensor(float(class_count)),
        f'{diagnostic_prefix}_logit_mean': logits.new_zeros(()),
    }
    if selected_count == 0:
        return zero, diagnostics

    class_logits = logits[class_indices]
    selected_within_class = torch.topk(
        class_logits.detach(),
        k=selected_count,
        largest=not select_fake,
        sorted=False,
    ).indices
    selected_indices = class_indices[selected_within_class]
    per_sample = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction='none',
    )
    loss = per_sample[selected_indices].sum() / logits.numel()
    diagnostics[f'{diagnostic_prefix}_logit_mean'] = (
        logits.detach()[selected_indices].mean())
    return loss, diagnostics


def fake_reweighting_loss(
    logits,
    labels,
    fraction=0.25,
    mode='hard',
    generator=None,
):
    """Apply a count-budget-matched fake-only auxiliary BCE objective."""
    if mode not in FAKE_REWEIGHTING_MODES:
        raise ValueError(
            f'fake reweighting mode must be one of '
            f'{FAKE_REWEIGHTING_MODES}, got {mode!r}')
    if not 0 < fraction < 1:
        raise ValueError(
            f'fake reweighting fraction must be in (0, 1), got {fraction}')

    logits, labels = _flatten_binary_inputs(logits, labels)
    fake_indices = torch.nonzero(labels >= 0.5, as_tuple=False).flatten()
    fake_count = fake_indices.numel()
    effective_count = int(fraction * fake_count)
    zero = logits.sum() * 0.0
    diagnostics = {
        'hard_fake_selected': logits.new_zeros(()),
        'hard_fake_effective': logits.new_tensor(float(effective_count)),
        'hard_fake_total': logits.new_tensor(float(fake_count)),
        'hard_fake_logit_mean': logits.new_zeros(()),
    }
    if effective_count == 0:
        return zero, diagnostics

    fake_logits = logits[fake_indices]
    if mode == 'hard':
        selected_within_fake = torch.topk(
            fake_logits.detach(),
            k=effective_count,
            largest=False,
            sorted=False,
        ).indices
        selected_indices = fake_indices[selected_within_fake]
        selection_scale = 1.0
    elif mode == 'random':
        selected_within_fake = torch.randperm(
            fake_count,
            device=fake_indices.device,
            generator=generator,
        )[:effective_count]
        selected_indices = fake_indices[selected_within_fake]
        selection_scale = 1.0
    else:
        selected_indices = fake_indices
        selection_scale = effective_count / fake_count

    per_sample = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction='none',
    )
    loss = (
        selection_scale
        * per_sample[selected_indices].sum()
        / logits.numel()
    )
    diagnostics['hard_fake_selected'] = logits.new_tensor(
        float(selected_indices.numel()))
    diagnostics['hard_fake_logit_mean'] = (
        logits.detach()[selected_indices].mean())
    return loss, diagnostics


def hard_fake_reweighting_loss(logits, labels, fraction=0.25):
    """Add BCE for the lowest-logit fake samples in the global batch."""
    return fake_reweighting_loss(
        logits,
        labels,
        fraction=fraction,
        mode='hard',
    )


def hard_real_reweighting_loss(logits, labels, fraction=0.25):
    """Add BCE for the highest-logit real samples in the global batch."""
    return _hard_class_reweighting_loss(
        logits,
        labels,
        fraction=fraction,
        select_fake=False,
    )


def symmetric_logit_anchor_loss(logits, labels, anchor=3.0):
    """Keep real/fake logits near fixed symmetric targets around zero."""
    if anchor <= 0:
        raise ValueError(f'anchor must be positive, got {anchor}')
    logits, labels = _flatten_binary_inputs(logits, labels)
    targets = labels.mul(2.0).sub(1.0).mul(anchor)
    return F.smooth_l1_loss(logits, targets)


def symmetric_logit_anchor_diagnostics(logits, labels, anchor=3.0):
    """Return per-class logit means and deviations from their anchors."""
    if anchor <= 0:
        raise ValueError(f'anchor must be positive, got {anchor}')
    logits, labels = _flatten_binary_inputs(logits.detach(), labels)
    targets = labels.mul(2.0).sub(1.0).mul(anchor)
    deviations = (logits - targets).abs()

    def masked_mean(values, mask):
        weights = mask.to(dtype=values.dtype)
        count = weights.sum()
        mean = (values * weights).sum() / count.clamp_min(1.0)
        return torch.where(
            count > 0,
            mean,
            values.new_tensor(float('nan')),
        )

    real_mask = labels < 0.5
    fake_mask = ~real_mask
    return {
        'real_logit_mean': masked_mean(logits, real_mask),
        'fake_logit_mean': masked_mean(logits, fake_mask),
        'real_anchor_deviation': masked_mean(deviations, real_mask),
        'fake_anchor_deviation': masked_mean(deviations, fake_mask),
    }
