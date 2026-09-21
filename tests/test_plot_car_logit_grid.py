import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_car_logit_grid import (
    load_prediction_csvs,
    main,
    parse_source_specs,
    validate_alignment,
)


FIELDS = ('generator', 'path', 'label', 'raw_logit', 'score')


def write_predictions(path, rows):
    with path.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def source_rows(source, offset=0.0):
    stem = source.lower()
    values = (
        (0, -1.2 + offset),
        (0, -0.6 + offset),
        (1, 0.3 + offset),
        (1, 1.1 + offset),
    )
    return [
        {
            'generator': source,
            'path': f'/{stem}/{label}-{index}.png',
            'label': label,
            'raw_logit': raw_logit,
            'score': 0.2 if label == 0 else 0.8,
        }
        for index, (label, raw_logit) in enumerate(values)
    ]


class PlotCarLogitGridTests(unittest.TestCase):
    def test_parses_explicit_source_display_names(self):
        specs = parse_source_specs(['seeingdark=SITD', 'crn=CRN'])
        self.assertEqual(specs[0]['csv_name'], 'seeingdark')
        self.assertEqual(specs[0]['display_name'], 'SITD')
        self.assertEqual(specs[1]['normalized_name'], 'crn')

    def test_rejects_duplicate_identity_across_input_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / 'first.csv'
            second = root / 'second.csv'
            rows = source_rows('crn')
            write_predictions(first, rows)
            write_predictions(second, rows)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_prediction_csvs([first, second])

    def test_requires_identical_set_and_order(self):
        baseline = source_rows('crn')
        reordered = [baseline[1], baseline[0], *baseline[2:]]
        with self.assertRaisesRegex(ValueError, 'different image order'):
            validate_alignment(baseline, reordered, 'GAN')

        changed = [dict(row) for row in baseline]
        changed[0]['path'] = '/different.png'
        with self.assertRaisesRegex(ValueError, 'different image sets'):
            validate_alignment(baseline, changed, 'GAN')

    def test_cli_writes_two_by_four_grid_and_audit_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gan_baseline = root / 'gan_baseline.csv'
            gan_car = root / 'gan_car.csv'
            diffusion_baseline = root / 'diffusion_baseline.csv'
            diffusion_car = root / 'diffusion_car.csv'

            gan_sources = ('deepfake', 'crn')
            diffusion_sources = ('adm', 'vqdm')
            write_predictions(
                gan_baseline,
                [row for source in gan_sources for row in source_rows(source)],
            )
            write_predictions(
                gan_car,
                [
                    row
                    for source in gan_sources
                    for row in source_rows(source, offset=0.15)
                ],
            )
            write_predictions(
                diffusion_baseline,
                [
                    row
                    for source in diffusion_sources
                    for row in source_rows(source)
                ],
            )
            write_predictions(
                diffusion_car,
                [
                    row
                    for source in diffusion_sources
                    for row in source_rows(source, offset=0.2)
                ],
            )
            output_prefix = root / 'output' / 'car_logit_grid'

            summary = main([
                '--gan_baseline_csv', str(gan_baseline),
                '--gan_car_csv', str(gan_car),
                '--diffusion_baseline_csv', str(diffusion_baseline),
                '--diffusion_car_csv', str(diffusion_car),
                '--output_prefix', str(output_prefix),
                '--formats', 'png',
                '--bins', '8',
                '--dpi', '120',
            ])

            self.assertTrue(Path(f'{output_prefix}.png').is_file())
            summary_path = Path(f'{output_prefix}.summary.json')
            self.assertTrue(summary_path.is_file())
            with summary_path.open(encoding='utf-8') as input_file:
                saved = json.load(input_file)
            self.assertEqual(
                [source['display_name']
                 for source in saved['protocols']['gan']['sources']],
                ['Deepfakes', 'CRN'],
            )
            self.assertEqual(
                [source['display_name']
                 for source in saved['protocols']['diffusion']['sources']],
                ['ADM', 'VQDM'],
            )
            self.assertEqual(
                saved['figure_contract']['all_matching_samples_included'],
                True,
            )
            self.assertEqual(summary['alignment']['gan_same_set_and_order'], True)
            self.assertEqual(summary['plot']['gan_bins'], 8)
            self.assertEqual(summary['plot']['diffusion_bins'], 8)
            self.assertIn(
                'within each source and class',
                summary['figure_contract']['axis_comparability'],
            )
            self.assertIn(
                'rows encode Real/Generated classes',
                summary['figure_contract']['layout'],
            )


if __name__ == '__main__':
    unittest.main()
