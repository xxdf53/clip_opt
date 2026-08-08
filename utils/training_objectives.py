import torch
import torch.nn.functional as F


def _validate_feature_pair(first, second, names):
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError(f'{names} must both have shape [batch, features]')
    if first.shape != second.shape:
        raise ValueError(f'{names} must have identical shapes')


def counterfactual_prompt_components(
    real_text_embeddings,
    fake_text_embeddings,
):
    """Split paired prompts into authenticity direction and content center."""
    _validate_feature_pair(
        real_text_embeddings,
        fake_text_embeddings,
        'real/fake text embeddings',
    )
    real_text_embeddings = F.normalize(
        real_text_embeddings, p=2, dim=-1)
    fake_text_embeddings = F.normalize(
        fake_text_embeddings, p=2, dim=-1)
    authenticity_direction = F.normalize(
        fake_text_embeddings - real_text_embeddings,
        p=2,
        dim=-1,
    )
    content_center = F.normalize(
        0.5 * (fake_text_embeddings + real_text_embeddings),
        p=2,
        dim=-1,
    )
    return authenticity_direction, content_center


def cpd_direction_loss(
    image_residual,
    authenticity_direction,
    labels,
    margin=0.1,
):
    """Push the LoRA residual along the label-conditioned prompt direction."""
    if margin < 0:
        raise ValueError(f'CPD direction margin cannot be negative: {margin}')
    _validate_feature_pair(
        image_residual,
        authenticity_direction,
        'image residual/authenticity direction',
    )
    labels = labels.flatten().to(dtype=image_residual.dtype)
    if labels.numel() != image_residual.shape[0]:
        raise ValueError('labels must contain one value per image residual')
    label_sign = labels.mul(2.0).sub(1.0)
    projection = (
        image_residual * authenticity_direction.detach()
    ).sum(dim=-1)
    signed_projection = label_sign * projection
    return F.softplus(margin - signed_projection).mean()


def cpd_content_rejection_loss(image_residual, content_center):
    """Discourage the task-specific LoRA residual from following content."""
    _validate_feature_pair(
        image_residual,
        content_center,
        'image residual/content center',
    )
    alignment = (
        image_residual * content_center.detach()
    ).sum(dim=-1)
    return alignment.square().mean()


def cpd_diagnostics(
    image_residual,
    authenticity_direction,
    content_center,
    labels,
    prompt_gap,
):
    """Return detached observables needed to falsify CPD's mechanism."""
    labels = labels.flatten().to(dtype=image_residual.dtype)
    label_sign = labels.mul(2.0).sub(1.0)
    projection = (
        image_residual.detach() * authenticity_direction.detach()
    ).sum(dim=-1)
    content_alignment = (
        image_residual.detach() * content_center.detach()
    ).sum(dim=-1).abs()
    return {
        'cpd_signed_projection': (
            label_sign * projection
        ).mean(),
        'cpd_content_alignment': content_alignment.mean(),
        'cpd_prompt_gap': prompt_gap.detach().flatten().mean(),
    }


def _flatten_binary_inputs(logits, labels):
    logits = logits.flatten()
    labels = labels.flatten().to(dtype=logits.dtype)
    if logits.numel() != labels.numel():
        raise ValueError('logits and labels must contain the same samples')
    return logits, labels


def symmetric_logit_anchor_loss(logits, labels, anchor=3.0):
    """Keep real/fake logits near fixed symmetric targets around zero."""
    if anchor <= 0:
        raise ValueError(f'anchor must be positive, got {anchor}')

    logits, labels = _flatten_binary_inputs(logits, labels)
    targets = labels.mul(2.0).sub(1.0).mul(anchor)
    return F.smooth_l1_loss(logits, targets)


def symmetric_logit_center_loss(logits, labels):
    """Center the real/fake class means around the zero decision boundary."""
    logits, labels = _flatten_binary_inputs(logits, labels)
    real_logits = logits[labels < 0.5]
    fake_logits = logits[labels >= 0.5]
    if real_logits.numel() == 0 or fake_logits.numel() == 0:
        return logits.sum() * 0.0

    midpoint = 0.5 * (real_logits.mean() + fake_logits.mean())
    return F.smooth_l1_loss(midpoint, torch.zeros_like(midpoint))


def worst_group_bce_loss(logits, labels, groups, group_count=3):
    """Return the largest present-group BCE and every group mean."""
    if group_count <= 0:
        raise ValueError('group_count must be positive')
    logits, labels = _flatten_binary_inputs(logits, labels)
    groups = groups.flatten().to(device=logits.device, dtype=torch.long)
    if groups.numel() != logits.numel():
        raise ValueError('groups must contain one value per logit')
    if torch.any((groups < 0) | (groups >= group_count)):
        raise ValueError('groups contain an out-of-range index')

    sample_losses = F.binary_cross_entropy_with_logits(
        logits, labels, reduction='none')
    group_losses = []
    present_losses = []
    for group in range(group_count):
        mask = groups == group
        if mask.any():
            group_loss = sample_losses[mask].mean()
            present_losses.append(group_loss)
        else:
            group_loss = logits.new_tensor(float('nan'))
        group_losses.append(group_loss)

    return torch.stack(present_losses).max(), torch.stack(group_losses)


def symmetric_logit_anchor_diagnostics(logits, labels, anchor=3.0):
    """Return detached real/fake means and deviations from their anchors."""
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
