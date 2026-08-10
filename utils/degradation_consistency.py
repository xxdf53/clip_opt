"""Training-only consistency across clean and resampled image views."""

import torch
import torch.nn.functional as F


def resize_degraded_view(images, scale=0.75):
    """Downsample and restore a tensor batch without changing its shape."""
    if images.ndim != 4:
        raise ValueError('images must have shape [batch, channels, height, width]')
    if not 0.0 < scale <= 1.0:
        raise ValueError('degradation scale must be in (0, 1]')

    height, width = images.shape[-2:]
    reduced_size = (
        max(1, round(height * scale)),
        max(1, round(width * scale)),
    )
    reduced = F.interpolate(
        images,
        size=reduced_size,
        mode='bilinear',
        align_corners=False,
        antialias=True,
    )
    return F.interpolate(
        reduced,
        size=(height, width),
        mode='bilinear',
        align_corners=False,
        antialias=True,
    )


def degradation_consistency_loss(student_logits, teacher_logits):
    """Keep degraded-view logits close to detached clean-view logits."""
    student_logits = student_logits.flatten()
    teacher_logits = teacher_logits.detach().flatten()
    if student_logits.shape != teacher_logits.shape:
        raise ValueError('student and teacher logits must have the same shape')
    return F.smooth_l1_loss(student_logits, teacher_logits)
