import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.binary_calibration import (
    apply_parameters,
    calibrated_records,
    fit_balanced_threshold,
    fit_calibration,
    load_calibration,
    load_prediction_csv,
    save_calibration,
    summarize_calibrated_records,
)


def write_predictions(path, rows):
    with path.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=('generator', 'path', 'label', 'raw_logit', 'score'),
        )
        writer.writeheader()
        writer.writerows(rows)


class BinaryCalibrationTests(unittest.TestCase):
    def test_balanced_threshold_corrects_shifted_logits(self):
        labels = np.asarray([0, 0, 1, 1])
        logits = np.asarray([0.1, 0.2, 0.3, 0.4])

        threshold, score = fit_balanced_threshold(labels, logits)

        self.assertEqual(threshold, 0.2)
        self.assertEqual(score, 1.0)

    def test_fit_and_apply_improves_validation_nll(self):
        rows = [
            {'generator': 'g', 'path': '/r1', 'label': 0, 'raw_logit': -0.2},
            {'generator': 'g', 'path': '/r2', 'label': 0, 'raw_logit': -0.1},
            {'generator': 'g', 'path': '/f1', 'label': 1, 'raw_logit': 0.1},
            {'generator': 'g', 'path': '/f2', 'label': 1, 'raw_logit': 0.2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            validation_csv = Path(directory) / 'validation.csv'
            write_predictions(validation_csv, [
                {**row, 'score': 0.5} for row in rows
            ])
            _, records = load_prediction_csv(validation_csv)
            parameters = fit_calibration(records, validation_csv)

        self.assertGreater(parameters['temperature'], 0.0)
        self.assertLessEqual(
            parameters['validation_metrics_calibrated']['brier'],
            parameters['validation_metrics_raw']['brier'],
        )

    def test_calibration_json_round_trip(self):
        parameters = {
            'schema_version': 1,
            'method': 'balanced-threshold-temperature-v1',
            'threshold': 0.25,
            'temperature': 0.5,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = save_calibration(
                parameters, Path(directory) / 'calibration.json')
            _, loaded = load_calibration(path)

        self.assertEqual(loaded['threshold'], 0.25)
        self.assertEqual(loaded['temperature'], 0.5)

    def test_apply_uses_frozen_parameters_and_preserves_ranking(self):
        records = [
            {'generator': 'g', 'path': '/r', 'label': 0, 'raw_logit': -0.1},
            {'generator': 'g', 'path': '/f', 'label': 1, 'raw_logit': 0.2},
        ]
        parameters = {'threshold': 0.05, 'temperature': 0.5}

        output = calibrated_records(records, parameters)
        summary = summarize_calibrated_records(output)

        self.assertLess(
            output[0]['calibrated_score'], output[1]['calibrated_score'])
        self.assertEqual(summary['overall_metrics']['acc'], 100.0)

    def test_temperature_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            apply_parameters([0.0], threshold=0.0, temperature=0.0)

    def test_rejects_calibration_with_unknown_method(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps({
                'method': 'unknown',
                'threshold': 0.0,
                'temperature': 1.0,
            }), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                load_calibration(path)


if __name__ == '__main__':
    unittest.main()
