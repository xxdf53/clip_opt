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

    def test_accepts_logit_anchor_options(self):
        args = self.parse([
            '--anchor_loss_weight', '0.5',
            '--logit_anchor', '3.5',
        ])

        self.assertEqual(args.anchor_loss_weight, 0.5)
        self.assertEqual(args.logit_anchor, 3.5)

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
