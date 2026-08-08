import argparse
import unittest

from options.base_options import BaseOptions, build_experiment_name


class TrainingOptionTests(unittest.TestCase):
    def parse(self, argv):
        parser = BaseOptions().initialize(argparse.ArgumentParser())
        return parser.parse_args(argv)

    def test_anchor_defaults_to_disabled_with_symmetric_target_three(self):
        args = self.parse([])

        self.assertEqual(args.anchor_loss_weight, 0.0)
        self.assertEqual(args.logit_anchor, 3.0)
        self.assertEqual(args.logit_center_loss_weight, 0.0)

    def test_accepts_logit_anchor_options(self):
        args = self.parse([
            '--anchor_loss_weight', '0.5',
            '--logit_anchor', '3.5',
        ])

        self.assertEqual(args.anchor_loss_weight, 0.5)
        self.assertEqual(args.logit_anchor, 3.5)

    def test_cpd_defaults_to_disabled(self):
        args = self.parse([])

        self.assertEqual(args.cpd_direction_weight, 0.0)
        self.assertEqual(args.cpd_content_weight, 0.0)
        self.assertEqual(args.cpd_direction_margin, 0.1)
        self.assertEqual(args.cpd_start_step, 0)
        self.assertEqual(args.cpd_warmup_steps, 0)

    def test_accepts_cpd_options(self):
        args = self.parse([
            '--cpd_direction_weight', '1.0',
            '--cpd_content_weight', '0.1',
            '--cpd_direction_margin', '0.2',
            '--cpd_start_step', '400',
            '--cpd_warmup_steps', '400',
        ])

        self.assertEqual(args.cpd_direction_weight, 1.0)
        self.assertEqual(args.cpd_content_weight, 0.1)
        self.assertEqual(args.cpd_direction_margin, 0.2)
        self.assertEqual(args.cpd_start_step, 400)
        self.assertEqual(args.cpd_warmup_steps, 400)

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

    def test_compact_name_records_active_anchor_configuration(self):
        args = self.parse([
            '--name', 'c2p_anchor',
            '--seed', '123',
            '--lora_r', '6',
            '--lora_alpha', '6',
            '--lora_dropout', '0.8',
            '--lr', '0.0002',
            '--claloss', '8.0',
            '--anchor_loss_weight', '0.5',
            '--logit_anchor', '3.0',
        ])

        name = build_experiment_name(args, timestamp='20260727-120000')

        self.assertEqual(
            name,
            'c2p_anchor__20260727-120000__s123__'
            'r6a6d0.8__lr0.0002__c8.0__anchor-w0.5-t3.0',
        )

    def test_compact_name_omits_disabled_anchor(self):
        args = self.parse(['--name', 'c2p_baseline'])

        name = build_experiment_name(args, timestamp='20260727-120000')

        self.assertNotIn('anchor-', name)

    def test_compact_name_records_active_logit_centering(self):
        args = self.parse([
            '--name', 'c2p_center',
            '--logit_center_loss_weight', '1.0',
        ])

        name = build_experiment_name(args, timestamp='20260809-120000')

        self.assertTrue(name.endswith('__center-w1.0'))

    def test_compact_name_records_active_cpd_configuration(self):
        args = self.parse([
            '--name', 'c2p_cpd',
            '--cpd_direction_weight', '1.0',
            '--cpd_content_weight', '0.1',
            '--cpd_direction_margin', '0.2',
        ])

        name = build_experiment_name(args, timestamp='20260728-120000')

        self.assertTrue(name.endswith('__cpd-d1.0-c0.1-m0.2'))

    def test_compact_name_records_active_cpd_schedule(self):
        args = self.parse([
            '--name', 'c2p_cpd_warmup',
            '--cpd_direction_weight', '0.5',
            '--cpd_start_step', '400',
            '--cpd_warmup_steps', '400',
        ])

        name = build_experiment_name(args, timestamp='20260728-130000')

        self.assertTrue(
            name.endswith('__cpd-d0.5-c0.0-m0.1-s400-w400'))

    def test_name_is_truncated_by_utf8_bytes(self):
        args = self.parse([
            '--name', 'long_experiment_name_' * 100,
            '--anchor_loss_weight', '0.5',
        ])

        name = build_experiment_name(args, timestamp='20260727-120000')

        self.assertLessEqual(len(name.encode('utf-8')), 180)
        self.assertTrue(name.endswith('__anchor-w0.5-t3.0'))


if __name__ == '__main__':
    unittest.main()
