import argparse
import unittest

from options.base_options import (
    BaseOptions,
    build_experiment_name,
    validate_experiment_configuration,
)


class TrainingOptionTests(unittest.TestCase):
    def parse(self, argv):
        parser = BaseOptions().initialize(argparse.ArgumentParser())
        return parser.parse_args(argv)

    def test_retained_gan_objectives_default_to_disabled(self):
        args = self.parse([])

        self.assertEqual(args.anchor_loss_weight, 0.0)
        self.assertEqual(args.cpd_direction_weight, 0.0)
        self.assertFalse(args.paired_authenticity_prompt_classification)
        self.assertFalse(args.pld_lora_initialization)
        self.assertEqual(args.pld_lora_microbatch_size, 8)
        self.assertEqual(args.hard_fake_loss_weight, 0.0)
        self.assertEqual(args.hard_fake_fraction, 0.25)

    def test_data_seed_and_manifest_default_to_disabled(self):
        args = self.parse([])

        self.assertIsNone(args.data_seed)
        self.assertEqual(args.train_manifest, '')

    def test_compact_name_distinguishes_data_and_model_seeds(self):
        args = self.parse([
            '--name', 'paired',
            '--seed', '42',
            '--data_seed', '314159',
        ])

        name = build_experiment_name(args, timestamp='20260808-120000')

        self.assertIn('__ds314159__ms42__', name)

    def test_compact_name_records_retained_gan_objectives(self):
        args = self.parse([
            '--name', 'gan_objectives',
            '--anchor_loss_weight', '0.5',
            '--logit_anchor', '3',
            '--cpd_direction_weight', '0.5',
            '--cpd_start_step', '400',
            '--cpd_warmup_steps', '400',
        ])

        name = build_experiment_name(args, timestamp='20260809-150000')

        self.assertIn('__anchor-w0.5-t3.0__', name)
        self.assertIn('__cpd-d0.5-c0.0-m0.1-s400-w400', name)

    def test_compact_name_records_pld_lora_initialization(self):
        args = self.parse([
            '--name', 'pld',
            '--pld_lora_initialization',
        ])

        name = build_experiment_name(args, timestamp='20260818-120000')

        self.assertTrue(name.endswith('__pld-lora'))

    def test_compact_name_records_hard_fake_reweighting(self):
        args = self.parse([
            '--name', 'hfr',
            '--hard_fake_loss_weight', '1.0',
            '--hard_fake_fraction', '0.25',
        ])

        name = build_experiment_name(args, timestamp='20260819-120000')

        self.assertTrue(name.endswith('__hfr-w1.0-q0.25'))

    def test_pld_requires_manifest_and_standalone_training(self):
        missing_manifest = self.parse(['--pld_lora_initialization'])
        with self.assertRaisesRegex(ValueError, 'requires --train_manifest'):
            validate_experiment_configuration(missing_manifest)

        stacked = self.parse([
            '--pld_lora_initialization',
            '--train_manifest', 'fixed.txt',
            '--anchor_loss_weight', '0.5',
        ])
        with self.assertRaisesRegex(ValueError, 'must be tested alone'):
            validate_experiment_configuration(stacked)

    def test_hfr_requires_standalone_training(self):
        stacked = self.parse([
            '--hard_fake_loss_weight', '1.0',
            '--anchor_loss_weight', '0.5',
        ])
        with self.assertRaisesRegex(ValueError, 'must be tested alone'):
            validate_experiment_configuration(stacked)

    def test_compact_name_records_paired_authenticity_classification(self):
        args = self.parse([
            '--name', 'paired_authenticity',
            '--paired_authenticity_prompt_classification',
        ])

        name = build_experiment_name(args, timestamp='20260817-120000')

        self.assertTrue(name.endswith('__papc'))

    def test_name_is_truncated_by_utf8_bytes(self):
        args = self.parse(['--name', 'long_experiment_name_' * 100])

        name = build_experiment_name(args, timestamp='20260727-120000')

        self.assertLessEqual(len(name.encode('utf-8')), 180)


if __name__ == '__main__':
    unittest.main()
