import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score


def compute_binary_metrics(labels, scores):
    """Compute binary detection metrics as percentages at threshold 0.5."""
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if labels.size != scores.size or labels.size == 0:
        raise ValueError('labels and scores must be non-empty and equal length')

    unique_labels = set(labels.tolist())
    if unique_labels != {0, 1}:
        raise ValueError('both real and fake samples are required')

    predictions = scores > 0.5
    real_mask = labels == 0
    fake_mask = labels == 1
    return {
        'n': int(labels.size),
        'acc': accuracy_score(labels, predictions) * 100.0,
        'real_acc': accuracy_score(
            labels[real_mask], predictions[real_mask]) * 100.0,
        'fake_acc': accuracy_score(
            labels[fake_mask], predictions[fake_mask]) * 100.0,
        'ap': average_precision_score(labels, scores) * 100.0,
    }
