import argparse
import unittest

from options.base_options import BaseOptions, build_experiment_name


class TrainingOptionTests(unittest.TestCase):
    def parse(self, argv):
        parser = BaseOptions().initialize(argparse.ArgumentParser())
        return parser.parse_args(argv)

    def test_new_local_training_defaults_to_adaptive_residual(self):
        args = self.parse([])

        self.assertEqual(args.local_fusion, 'adaptive_residual')
        self.assertEqual(args.local_gate_init, 0.01)
        self.assertEqual(args.rank_loss_weight, 0.0)

    def test_accepts_protected_global_and_auxiliary_losses(self):
        args = self.parse([
            '--init_baseline_checkpoint', 'baseline.pth',
            '--freeze_global_branch',
            '--rank_loss_weight', '1.0',
            '--preserve_loss_weight', '0.1',
            '--gate_loss_weight', '0.01',
        ])

        self.assertEqual(args.init_baseline_checkpoint, 'baseline.pth')
        self.assertTrue(args.freeze_global_branch)
        self.assertEqual(args.rank_loss_weight, 1.0)
        self.assertEqual(args.preserve_loss_weight, 0.1)
        self.assertEqual(args.gate_loss_weight, 0.01)

    def test_compact_name_keeps_only_identifying_architecture_fields(self):
        args = self.parse([
            '--name', 'c2p_local_adaptive_residual',
            '--seed', '123',
            '--lora_r', '6',
            '--lora_alpha', '6',
            '--lora_dropout', '0.8',
            '--lr', '0.0002',
            '--claloss', '8.0',
            '--use_local_features',
            '--local_layer', '12',
            '--local_pool', 'mean_std',
            '--local_dim', '256',
            '--local_fusion', 'adaptive_residual',
            '--freeze_global_branch',
        ])

        name = build_experiment_name(args, timestamp='20260719-203208')

        self.assertEqual(
            name,
            'c2p_local_adaptive_residual__20260719-203208__s123__'
            'r6a6d0.8__lr0.0002__c8.0__L12-ms-d256-ar__fg',
        )

    def test_name_is_truncated_by_utf8_bytes(self):
        args = self.parse([
            '--name', '超长实验名称' * 100,
            '--use_local_features',
        ])

        name = build_experiment_name(args, timestamp='20260719-203208')

        self.assertLessEqual(len(name.encode('utf-8')), 180)


if __name__ == '__main__':
    unittest.main()
