import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def compute_binary_metrics(labels, scores, ece_bins=15):
    """Compute binary detection metrics as percentages at threshold 0.5."""
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if labels.size != scores.size or labels.size == 0:
        raise ValueError('labels and scores must be non-empty and equal length')

    unique_labels = set(labels.tolist())
    if unique_labels != {0, 1}:
        raise ValueError('both real and fake samples are required')
    if ece_bins <= 0:
        raise ValueError('ece_bins must be positive')
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError('scores must be probabilities in [0, 1]')

    predictions = scores > 0.5
    real_mask = labels == 0
    fake_mask = labels == 1
    bin_edges = np.linspace(0.0, 1.0, ece_bins + 1)
    calibration_error = 0.0
    for index, (lower, upper) in enumerate(
            zip(bin_edges[:-1], bin_edges[1:])):
        if index == ece_bins - 1:
            bin_mask = (scores >= lower) & (scores <= upper)
        else:
            bin_mask = (scores >= lower) & (scores < upper)
        if not np.any(bin_mask):
            continue
        calibration_error += (
            np.mean(bin_mask)
            * abs(float(labels[bin_mask].mean())
                  - float(scores[bin_mask].mean()))
        )

    return {
        'n': int(labels.size),
        'acc': accuracy_score(labels, predictions) * 100.0,
        'real_acc': accuracy_score(
            labels[real_mask], predictions[real_mask]) * 100.0,
        'fake_acc': accuracy_score(
            labels[fake_mask], predictions[fake_mask]) * 100.0,
        'ap': average_precision_score(labels, scores) * 100.0,
        'roc_auc': roc_auc_score(labels, scores) * 100.0,
        'ece': calibration_error * 100.0,
        'brier': brier_score_loss(labels, scores),
    }
