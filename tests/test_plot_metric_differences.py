import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_metric_differences import main


class PlotMetricDifferencesTests(unittest.TestCase):
    def test_writes_outputs_and_exact_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            output_prefix = Path(directory) / 'metric.gap'
            summary = main([
                '--metrics', 'Macro ACC', 'Macro AP', 'AUROC',
                '--baseline', '88.75', '98.52', '98.14',
                '--car', '90.42', '98.77', '98.41',
                '--protocol_label', 'GAN | validation-fixed | seed 123',
                '--output_prefix', str(output_prefix),
            ])

            for suffix in ('.svg', '.pdf', '.png', '.summary.json'):
                self.assertTrue(Path(f'{output_prefix}{suffix}').is_file())
            with Path(f'{output_prefix}.summary.json').open(
                encoding='utf-8'
            ) as input_file:
                saved = json.load(input_file)
            self.assertAlmostEqual(
                saved['metrics'][0]['difference_pp'], 1.67)
            self.assertAlmostEqual(
                saved['metrics'][1]['difference_pp'], 0.25)
            self.assertEqual(summary['metrics'][2]['name'], 'AUROC')

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(SystemExit):
            main([
                '--metrics', 'ACC', 'AP',
                '--baseline', '90',
                '--car', '91', '92',
                '--protocol_label', 'Protocol',
                '--output_prefix', 'output',
            ])

    def test_rejects_nonfinite_values(self):
        with self.assertRaises(SystemExit):
            main([
                '--metrics', 'ACC',
                '--baseline', 'nan',
                '--car', '91',
                '--protocol_label', 'Protocol',
                '--output_prefix', 'output',
            ])


if __name__ == '__main__':
    unittest.main()
