"""Training-only objectives retained after experimental cleanup."""

import torch
import torch.nn.functional as F


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


def hard_fake_reweighting_loss(
    logits,
    labels,
    fraction=0.25,
):
    """Return a bias-neutral loss for selected hard fake samples.

    The forward value is the selected fake BCE normalized by the full batch
    size. During backward, the global logit mean is removed and restored as a
    detached value. This keeps the auxiliary gradient's common mode at zero,
    so HFR cannot directly update the classifier bias while still raising hard
    fake logits relative to the rest of the batch.
    """
    if not 0 < fraction < 1:
        raise ValueError(f'hard-fake fraction must be in (0, 1), got {fraction}')

    logits, labels = _flatten_binary_inputs(logits, labels)
    fake_indices = torch.nonzero(labels >= 0.5, as_tuple=False).flatten()
    fake_count = fake_indices.numel()
    selected_count = int(fraction * fake_count)
    zero = logits.sum() * 0.0
    diagnostics = {
        'hard_fake_selected': logits.new_tensor(float(selected_count)),
        'hard_fake_total': logits.new_tensor(float(fake_count)),
        'hard_fake_logit_mean': logits.new_zeros(()),
    }
    if selected_count == 0:
        return zero, diagnostics

    fake_logits = logits[fake_indices]
    selected_within_fake = torch.topk(
        fake_logits.detach(),
        k=selected_count,
        largest=False,
        sorted=False,
    ).indices
    selected_indices = fake_indices[selected_within_fake]
    mean_logit = logits.mean()
    bias_neutral_logits = logits - mean_logit + mean_logit.detach()
    per_sample = F.binary_cross_entropy_with_logits(
        bias_neutral_logits,
        labels,
        reduction='none',
    )
    loss = per_sample[selected_indices].sum() / logits.numel()
    diagnostics['hard_fake_logit_mean'] = (
        logits.detach()[selected_indices].mean())
    return loss, diagnostics


def gradient_conflict_diagnostics(first_loss, second_loss, parameters):
    """Measure gradient alignment without modifying parameter gradients."""
    parameters = tuple(
        parameter for parameter in parameters if parameter.requires_grad)
    first_gradients = torch.autograd.grad(
        first_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    second_gradients = torch.autograd.grad(
        second_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )

    dot_product = first_loss.detach().new_zeros((), dtype=torch.float32)
    first_squared_norm = dot_product.clone()
    second_squared_norm = dot_product.clone()
    shared_numel = 0
    for first_gradient, second_gradient in zip(
        first_gradients, second_gradients
    ):
        if first_gradient is None or second_gradient is None:
            continue
        first_gradient = first_gradient.detach().float()
        second_gradient = second_gradient.detach().float()
        dot_product = dot_product + (
            first_gradient * second_gradient).sum()
        first_squared_norm = first_squared_norm + first_gradient.square().sum()
        second_squared_norm = (
            second_squared_norm + second_gradient.square().sum())
        shared_numel += first_gradient.numel()

    if shared_numel == 0:
        raise ValueError('diagnostic losses have no shared trainable parameters')

    first_norm = first_squared_norm.sqrt()
    second_norm = second_squared_norm.sqrt()
    denominator = (first_norm * second_norm).clamp_min(
        torch.finfo(dot_product.dtype).eps)
    cosine = dot_product / denominator
    return {
        'gradient_cosine': cosine,
        'gradient_conflict': (dot_product < 0).to(dot_product.dtype),
        'gradient_contrastive_norm': first_norm,
        'gradient_classification_norm': second_norm,
        'gradient_shared_numel': dot_product.new_tensor(float(shared_numel)),
    }


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
