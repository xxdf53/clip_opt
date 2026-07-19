import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from utils.binary_evaluation import (
    build_group_dataset,
    build_transform,
    evaluate_groups,
    format_diagnostics,
    format_metrics,
    write_predictions_csv,
)


def write_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (32, 24), color=color).save(path)


def make_binary_leaf(path, stem):
    write_image(path / '0_real' / f'{stem}_real.png', (10, 20, 30))
    write_image(path / '1_fake' / f'{stem}_fake.png', (200, 210, 220))


class BinaryEvaluationTests(unittest.TestCase):
    def test_shared_metric_format_includes_calibration_and_logit_stats(self):
        metrics = {
            'n': 4,
            'acc': 75.0,
            'real_acc': 50.0,
            'fake_acc': 100.0,
            'ap': 98.5,
            'roc_auc': 97.5,
            'ece': 12.25,
            'brier': 0.125,
        }
        logit_stats = {
            'real_mean': -1.0,
            'real_std': 0.5,
            'fake_mean': 2.0,
            'fake_std': 0.75,
            'separation': 4.8,
        }

        self.assertIn('ACC= 75.00%', format_metrics('group', metrics))
        diagnostics = format_diagnostics(metrics, logit_stats)
        self.assertIn('AUROC= 97.50%', diagnostics)
        self.assertIn('ECE= 12.25%', diagnostics)
        self.assertIn('R=-1.000±0.500', diagnostics)

    def test_group_dataset_returns_deterministic_paths_and_binary_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_binary_leaf(root / 'horse', 'horse')
            make_binary_leaf(root / 'apple', 'apple')

            dataset = build_group_dataset(
                [root / 'horse', root / 'apple'],
                build_transform(),
            )

            samples = [dataset[index] for index in range(len(dataset))]
            paths = [Path(sample[2]).name for sample in samples]
            labels = [sample[1] for sample in samples]

            self.assertEqual(paths, [
                'apple_real.png',
                'apple_fake.png',
                'horse_real.png',
                'horse_fake.png',
            ])
            self.assertEqual(labels, [0, 1, 0, 1])
            self.assertEqual(tuple(samples[0][0].shape), (3, 224, 224))

    def test_evaluate_groups_keeps_partial_batch_and_raw_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_binary_leaf(root / 'generator_a', 'a')
            write_image(
                root / 'generator_a' / '1_fake' / 'a_fake_2.png',
                (230, 220, 210),
            )

            calls = []
            logits = torch.tensor([[-2.0], [2.0], [3.0]])

            def forward_logits(images):
                start = sum(calls)
                calls.append(images.shape[0])
                return logits[start:start + images.shape[0]]

            summary = evaluate_groups(
                {'generator_a': [root / 'generator_a']},
                forward_logits=forward_logits,
                device=torch.device('cpu'),
                batch_size=2,
                num_workers=0,
            )

            self.assertEqual(calls, [2, 1])
            self.assertEqual(len(summary['predictions']), 3)
            self.assertEqual(
                [record['raw_logit'] for record in summary['predictions']],
                [-2.0, 2.0, 3.0],
            )
            self.assertEqual(
                {record['generator'] for record in summary['predictions']},
                {'generator_a'},
            )
            self.assertEqual(summary['group_metrics']['generator_a']['n'], 3)
            self.assertEqual(summary['overall_metrics']['n'], 3)
            self.assertEqual(
                summary['group_logit_stats']['generator_a']['real_mean'],
                -2.0,
            )
            self.assertEqual(
                summary['group_logit_stats']['generator_a']['fake_mean'],
                2.5,
            )
            self.assertEqual(summary['overall_logit_stats']['fake_std'], 0.5)

    def test_write_predictions_csv_preserves_expected_fields(self):
        predictions = [{
            'generator': 'biggan',
            'path': '/dataset/biggan/0_real/example.png',
            'label': 0,
            'raw_logit': -1.25,
            'score': 0.222700,
        }]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'nested' / 'predictions.csv'
            write_predictions_csv(predictions, output)

            with output.open(newline='', encoding='utf-8') as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(
                list(rows[0]),
                ['generator', 'path', 'label', 'raw_logit', 'score'],
            )
            self.assertEqual(rows[0]['generator'], 'biggan')
            self.assertEqual(rows[0]['label'], '0')
            self.assertEqual(rows[0]['raw_logit'], '-1.25')

    def test_component_outputs_are_exported_without_changing_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_binary_leaf(root / 'generator_a', 'a')

            def forward_components(images):
                batch_size = images.shape[0]
                return {
                    'final_logits': torch.tensor([[-2.0], [2.0]])[:batch_size],
                    'global_logits': torch.tensor([[-1.5], [1.5]])[:batch_size],
                    'local_logits': torch.tensor([[-1.0], [1.0]])[:batch_size],
                    'gate': torch.full((batch_size, 1), 0.5),
                    'learned_gate': torch.full((batch_size, 1), 0.25),
                }

            summary = evaluate_groups(
                {'generator_a': [root / 'generator_a']},
                forward_logits=forward_components,
                device=torch.device('cpu'),
                batch_size=2,
                num_workers=0,
            )
            output = Path(directory) / 'components.csv'
            write_predictions_csv(summary['predictions'], output)

            with output.open(newline='', encoding='utf-8') as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(summary['overall_metrics']['acc'], 100.0)
            self.assertEqual(rows[0]['global_logit'], '-1.5')
            self.assertEqual(rows[0]['local_logit'], '-1.0')
            self.assertEqual(rows[0]['gate'], '0.5')
            self.assertEqual(rows[0]['learned_gate'], '0.25')


if __name__ == '__main__':
    unittest.main()
