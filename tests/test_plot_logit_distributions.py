import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_logit_distributions import (
    load_prediction_csv,
    main,
    validate_alignment,
)
from utils.logit_distribution import build_shared_bin_edges


FIELDS = ('generator', 'path', 'label', 'raw_logit', 'score')


def write_predictions(path, rows, fields=FIELDS):
    with path.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fields,
            extrasaction='ignore',
        )
        writer.writeheader()
        writer.writerows(rows)


def toy_rows(logit_shift=0.0):
    rows = [
        ('g1', '/real-1.png', 0, -0.8),
        ('g1', '/real-2.png', 0, -0.3),
        ('g2', '/fake-1.png', 1, 0.2),
        ('g2', '/fake-2.png', 1, 0.9),
    ]
    return [
        {
            'generator': generator,
            'path': path,
            'label': label,
            'raw_logit': raw_logit + logit_shift,
            'score': 0.25 if label == 0 else 0.75,
        }
        for generator, path, label, raw_logit in rows
    ]


class PlotLogitDistributionsTests(unittest.TestCase):
    def test_rejects_missing_required_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing.csv'
            fields = tuple(field for field in FIELDS if field != 'score')
            write_predictions(path, toy_rows(), fields=fields)

            with self.assertRaisesRegex(ValueError, 'missing fields'):
                load_prediction_csv(path)

    def test_rejects_duplicate_path_and_nonfinite_logit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = toy_rows()
            duplicate[1]['path'] = duplicate[0]['path']
            duplicate_path = root / 'duplicate.csv'
            write_predictions(duplicate_path, duplicate)
            with self.assertRaisesRegex(ValueError, 'duplicate path'):
                load_prediction_csv(duplicate_path)

            nonfinite = toy_rows()
            nonfinite[0]['raw_logit'] = 'nan'
            nonfinite_path = root / 'nonfinite.csv'
            write_predictions(nonfinite_path, nonfinite)
            with self.assertRaisesRegex(ValueError, 'must be finite'):
                load_prediction_csv(nonfinite_path)

    def test_requires_identical_set_and_order(self):
        baseline = toy_rows()
        reordered = [baseline[1], baseline[0], *baseline[2:]]
        with self.assertRaisesRegex(ValueError, 'different image order'):
            validate_alignment(baseline, reordered)

        changed = [dict(row) for row in baseline]
        changed[0]['path'] = '/different.png'
        with self.assertRaisesRegex(ValueError, 'different image sets'):
            validate_alignment(baseline, changed)

    def test_cli_writes_shared_bin_multi_format_outputs_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_csv = root / 'baseline.csv'
            car_csv = root / 'car.csv'
            write_predictions(baseline_csv, toy_rows())
            write_predictions(car_csv, toy_rows(logit_shift=0.1))
            output_prefix = root / 'figure' / 'logits.v1'

            summary = main([
                '--baseline_csv', str(baseline_csv),
                '--car_csv', str(car_csv),
                '--protocol_label', 'Toy protocol',
                '--output_prefix', str(output_prefix),
                '--bins', '4',
                '--density_scale', 'log',
            ])

            for suffix in ('.svg', '.pdf', '.png', '.summary.json'):
                self.assertTrue(Path(f'{output_prefix}{suffix}').is_file())
            with Path(f'{output_prefix}.summary.json').open(
                encoding='utf-8'
            ) as input_file:
                saved = json.load(input_file)
            expected_edges = build_shared_bin_edges(
                [
                    [-0.8, -0.3],
                    [0.2, 0.9],
                    [-0.7, -0.2],
                    [0.3, 1.0],
                ],
                bins=4,
            )
            self.assertEqual(
                saved['histogram']['shared_bin_edges'], expected_edges)
            self.assertEqual(saved['histogram']['density_scale'], 'log')
            self.assertEqual(saved['inputs']['baseline']['samples'], 4)
            self.assertEqual(saved['inputs']['baseline']['generators'], 2)
            self.assertEqual(summary['alignment']['same_set_and_order'], True)

    def test_rejects_nonpositive_bins(self):
        with self.assertRaises(SystemExit):
            main([
                '--baseline_csv', 'baseline.csv',
                '--car_csv', 'car.csv',
                '--protocol_label', 'Protocol',
                '--output_prefix', 'output',
                '--bins', '0',
            ])


if __name__ == '__main__':
    unittest.main()
