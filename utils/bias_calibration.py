"""Training-side scalar bias calibration for binary logits."""

import numpy as np
import torch


def _validate_binary_logits(logits, labels):
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if logits.size == 0 or logits.size != labels.size:
        raise ValueError('logits and labels must be non-empty and equal length')
    if not np.all(np.isfinite(logits)):
        raise ValueError('logits must be finite')
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('both real and fake calibration samples are required')
    return logits, labels.astype(np.int64, copy=False)


def balanced_error_rate(logits, labels, threshold=0.0):
    """Return equally weighted real and fake error rates at one threshold."""
    logits, labels = _validate_binary_logits(logits, labels)
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError('threshold must be finite')
    predictions = logits > threshold
    real_mask = labels == 0
    fake_mask = labels == 1
    real_error = predictions[real_mask].mean()
    fake_error = (~predictions[fake_mask]).mean()
    return 0.5 * float(real_error + fake_error)


def find_balanced_error_threshold(logits, labels):
    """Find the exact empirical threshold minimizing balanced error.

    Candidate partitions are evaluated after every group of tied logits. When
    several partitions are optimal, the candidate midpoint closest to zero is
    used so calibration makes the smallest unnecessary bias change.
    """
    logits, labels = _validate_binary_logits(logits, labels)
    order = np.argsort(logits, kind='stable')
    sorted_logits = logits[order]
    sorted_labels = labels[order]

    group_ends = np.flatnonzero(np.r_[np.diff(sorted_logits) != 0, True])
    split_counts = np.r_[0, group_ends + 1]
    cumulative_real = np.r_[0, np.cumsum(sorted_labels == 0)]
    cumulative_fake = np.r_[0, np.cumsum(sorted_labels == 1)]
    real_count = int((labels == 0).sum())
    fake_count = int((labels == 1).sum())

    false_negative_fake = cumulative_fake[split_counts]
    false_positive_real = real_count - cumulative_real[split_counts]
    errors = 0.5 * (
        false_negative_fake / fake_count
        + false_positive_real / real_count
    )

    thresholds = np.empty(split_counts.size, dtype=np.float64)
    thresholds[0] = np.nextafter(sorted_logits[0], -np.inf)
    thresholds[-1] = np.nextafter(sorted_logits[-1], np.inf)
    for index, split in enumerate(split_counts[1:-1], start=1):
        lower = sorted_logits[split - 1]
        upper = sorted_logits[split]
        thresholds[index] = lower + 0.5 * (upper - lower)

    minimum_error = errors.min()
    optimal = np.flatnonzero(
        np.isclose(errors, minimum_error, rtol=0.0, atol=1e-15))
    selected = optimal[np.argmin(np.abs(thresholds[optimal]))]
    return float(thresholds[selected])


def fold_threshold_into_linear_bias(linear, threshold):
    """Shift a binary linear head so its inference threshold remains zero."""
    if not isinstance(linear, torch.nn.Linear) or linear.out_features != 1:
        raise TypeError('calibration requires a one-output torch.nn.Linear')
    if linear.bias is None:
        raise ValueError('calibration requires a classifier bias')
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError('calibration threshold must be finite')
    with torch.no_grad():
        linear.bias.sub_(linear.bias.new_tensor(threshold))
