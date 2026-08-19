"""Validation-only threshold selection and binary logit calibration."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from utils.binary_metrics import compute_binary_metrics


CALIBRATION_METHOD = 'balanced-threshold-temperature-v1'


def stable_sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    scores = np.empty_like(values)
    nonnegative = values >= 0
    scores[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    scores[~nonnegative] = exp_values / (1.0 + exp_values)
    return scores


def file_sha256(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open('rb') as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_prediction_csv(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'prediction CSV not found: {path}')

    required = {'generator', 'path', 'label', 'raw_logit'}
    with path.open(newline='', encoding='utf-8') as input_file:
        reader = csv.DictReader(input_file)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f'prediction CSV is missing fields: {sorted(missing)}')
        records = []
        for row_index, row in enumerate(reader, start=2):
            try:
                label = int(row['label'])
                raw_logit = float(row['raw_logit'])
            except ValueError as error:
                raise ValueError(
                    f'invalid label or raw_logit at CSV row {row_index}') \
                    from error
            if label not in (0, 1):
                raise ValueError(
                    f'label must be 0 or 1 at CSV row {row_index}')
            if not np.isfinite(raw_logit):
                raise ValueError(
                    f'raw_logit must be finite at CSV row {row_index}')
            records.append({
                'generator': row['generator'],
                'path': row['path'],
                'label': label,
                'raw_logit': raw_logit,
            })
    if not records:
        raise ValueError('prediction CSV contains no records')
    if {record['label'] for record in records} != {0, 1}:
        raise ValueError('prediction CSV must contain both real and fake labels')
    return path, records


def arrays_from_records(records):
    labels = np.asarray([record['label'] for record in records], dtype=np.int64)
    logits = np.asarray(
        [record['raw_logit'] for record in records], dtype=np.float64)
    return labels, logits


def balanced_accuracy(labels, predictions):
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    real_mask = labels == 0
    fake_mask = labels == 1
    if not np.any(real_mask) or not np.any(fake_mask):
        raise ValueError('balanced accuracy requires both classes')
    real_accuracy = np.mean(predictions[real_mask] == 0)
    fake_accuracy = np.mean(predictions[fake_mask] == 1)
    return 0.5 * (real_accuracy + fake_accuracy)


def fit_balanced_threshold(labels, logits):
    """Choose logit threshold maximizing validation balanced accuracy."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if labels.size != logits.size or labels.size == 0:
        raise ValueError('labels and logits must be non-empty and equal length')
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('threshold fitting requires both classes')

    order = np.argsort(logits, kind='mergesort')
    sorted_logits = logits[order]
    sorted_labels = labels[order]
    unique_logits, starts = np.unique(sorted_logits, return_index=True)
    real_total = float(np.sum(labels == 0))
    fake_total = float(np.sum(labels == 1))

    true_negative = 0.0
    true_positive = fake_total
    first_threshold = np.nextafter(unique_logits[0], -np.inf)
    candidates = [(0.5, float(first_threshold))]
    for index, threshold in enumerate(unique_logits):
        start = starts[index]
        stop = starts[index + 1] if index + 1 < starts.size else labels.size
        group_labels = sorted_labels[start:stop]
        true_negative += float(np.sum(group_labels == 0))
        true_positive -= float(np.sum(group_labels == 1))
        score = 0.5 * (
            true_negative / real_total + true_positive / fake_total)
        candidates.append((score, float(threshold)))

    best_score = max(score for score, _ in candidates)
    tied_thresholds = [
        threshold
        for score, threshold in candidates
        if np.isclose(score, best_score, rtol=0.0, atol=1e-12)
    ]
    threshold = min(tied_thresholds, key=lambda value: (abs(value), value))
    return threshold, best_score


def binary_nll(labels, logits):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def fit_temperature(labels, logits, threshold, bounds=(0.01, 100.0)):
    """Fit positive temperature by validation binary NLL."""
    lower, upper = bounds
    if lower <= 0 or upper <= lower:
        raise ValueError('temperature bounds must satisfy 0 < lower < upper')
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)

    def objective(log_temperature):
        temperature = float(np.exp(log_temperature))
        calibrated_logits = (logits - threshold) / temperature
        return binary_nll(labels, calibrated_logits)

    result = minimize_scalar(
        objective,
        bounds=(float(np.log(lower)), float(np.log(upper))),
        method='bounded',
        options={'xatol': 1e-10},
    )
    if not result.success:
        raise RuntimeError(f'temperature fitting failed: {result.message}')
    return float(np.exp(result.x)), float(result.fun)


def apply_parameters(logits, threshold, temperature):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    calibrated_logits = (
        np.asarray(logits, dtype=np.float64) - float(threshold)
    ) / float(temperature)
    return calibrated_logits, stable_sigmoid(calibrated_logits)


def fit_calibration(records, validation_csv, temperature_bounds=(0.01, 100.0)):
    labels, logits = arrays_from_records(records)
    threshold, threshold_balanced_accuracy = fit_balanced_threshold(
        labels, logits)
    temperature, calibrated_nll = fit_temperature(
        labels,
        logits,
        threshold=threshold,
        bounds=temperature_bounds,
    )
    _, calibrated_scores = apply_parameters(logits, threshold, temperature)
    raw_scores = stable_sigmoid(logits)
    return {
        'schema_version': 1,
        'method': CALIBRATION_METHOD,
        'threshold': threshold,
        'temperature': temperature,
        'threshold_objective': 'validation_balanced_accuracy',
        'temperature_objective': 'validation_binary_nll',
        'validation_csv': str(Path(validation_csv).expanduser().resolve()),
        'validation_csv_sha256': file_sha256(validation_csv),
        'validation_samples': int(labels.size),
        'validation_real_samples': int(np.sum(labels == 0)),
        'validation_fake_samples': int(np.sum(labels == 1)),
        'validation_balanced_accuracy': threshold_balanced_accuracy * 100.0,
        'validation_calibrated_nll': calibrated_nll,
        'validation_metrics_raw': compute_binary_metrics(labels, raw_scores),
        'validation_metrics_calibrated': compute_binary_metrics(
            labels, calibrated_scores),
    }


def save_calibration(parameters, path):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as output_file:
        json.dump(parameters, output_file, indent=2, sort_keys=True)
        output_file.write('\n')
    return path


def load_calibration(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'calibration JSON not found: {path}')
    with path.open(encoding='utf-8') as input_file:
        parameters = json.load(input_file)
    if parameters.get('method') != CALIBRATION_METHOD:
        raise ValueError(
            f'unsupported calibration method: {parameters.get("method")}')
    threshold = float(parameters['threshold'])
    temperature = float(parameters['temperature'])
    if not np.isfinite(threshold) or not np.isfinite(temperature):
        raise ValueError('calibration parameters must be finite')
    if temperature <= 0:
        raise ValueError('calibration temperature must be positive')
    return path, parameters


def calibrated_records(records, parameters):
    _, logits = arrays_from_records(records)
    calibrated_logits, scores = apply_parameters(
        logits,
        threshold=parameters['threshold'],
        temperature=parameters['temperature'],
    )
    output = []
    for record, calibrated_logit, score in zip(
        records, calibrated_logits.tolist(), scores.tolist()
    ):
        output.append({
            **record,
            'raw_score': float(stable_sigmoid([record['raw_logit']])[0]),
            'calibrated_logit': float(calibrated_logit),
            'calibrated_score': float(score),
            'prediction': int(score > 0.5),
        })
    return output


def summarize_calibrated_records(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record['generator'], []).append(record)

    group_metrics = {}
    for generator, group_records in grouped.items():
        group_metrics[generator] = compute_binary_metrics(
            [record['label'] for record in group_records],
            [record['calibrated_score'] for record in group_records],
        )
    metric_names = ('acc', 'real_acc', 'fake_acc', 'ap', 'roc_auc', 'ece', 'brier')
    macro_metrics = {
        name: float(np.mean([
            metrics[name] for metrics in group_metrics.values()
        ]))
        for name in metric_names
    }
    macro_metrics['n'] = len(records)
    overall_metrics = compute_binary_metrics(
        [record['label'] for record in records],
        [record['calibrated_score'] for record in records],
    )
    return {
        'group_metrics': group_metrics,
        'macro_metrics': macro_metrics,
        'overall_metrics': overall_metrics,
    }


def write_calibrated_csv(records, path):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        'generator',
        'path',
        'label',
        'raw_logit',
        'raw_score',
        'calibrated_logit',
        'calibrated_score',
        'prediction',
    )
    with path.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return path
