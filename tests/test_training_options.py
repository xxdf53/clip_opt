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
        self.assertEqual(args.margin_loss_weight, 0.0)
        self.assertEqual(args.logit_margin, 1.0)
        self.assertEqual(args.anchor_loss_weight, 0.0)
        self.assertEqual(args.logit_anchor, 3.0)
        self.assertEqual(args.local_candidate_loss_weight, 0.0)
        self.assertEqual(args.gate_supervision_weight, 0.0)
        self.assertEqual(args.gate_target_margin, 0.1)
        self.assertEqual(args.residual_alpha, 1.0)
        self.assertEqual(args.residual_scale, 4.0)

    def test_accepts_protected_global_and_auxiliary_losses(self):
        args = self.parse([
            '--init_baseline_checkpoint', 'baseline.pth',
            '--freeze_global_branch',
            '--rank_loss_weight', '1.0',
            '--margin_loss_weight', '0.5',
            '--logit_margin', '1.5',
            '--anchor_loss_weight', '0.25',
            '--logit_anchor', '3.5',
            '--preserve_loss_weight', '0.1',
            '--gate_loss_weight', '0.01',
            '--local_candidate_loss_weight', '1.0',
            '--gate_supervision_weight', '1.0',
            '--gate_target_margin', '0.2',
        ])

        self.assertEqual(args.init_baseline_checkpoint, 'baseline.pth')
        self.assertTrue(args.freeze_global_branch)
        self.assertEqual(args.rank_loss_weight, 1.0)
        self.assertEqual(args.margin_loss_weight, 0.5)
        self.assertEqual(args.logit_margin, 1.5)
        self.assertEqual(args.anchor_loss_weight, 0.25)
        self.assertEqual(args.logit_anchor, 3.5)
        self.assertEqual(args.preserve_loss_weight, 0.1)
        self.assertEqual(args.gate_loss_weight, 0.01)
        self.assertEqual(args.local_candidate_loss_weight, 1.0)
        self.assertEqual(args.gate_supervision_weight, 1.0)
        self.assertEqual(args.gate_target_margin, 0.2)

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

    def test_compact_name_identifies_bounded_residual(self):
        args = self.parse([
            '--name', 'c2p_local_bounded_residual',
            '--use_local_features',
            '--local_fusion', 'bounded_residual',
            '--residual_alpha', '1.0',
            '--residual_scale', '4.0',
            '--freeze_global_branch',
        ])

        name = build_experiment_name(args, timestamp='20260721-120000')

        self.assertIn('__L12-ms-d256-br-a1.0-s4.0__fg', name)

    def test_name_is_truncated_by_utf8_bytes(self):
        args = self.parse([
            '--name', '超长实验名称' * 100,
            '--use_local_features',
        ])

        name = build_experiment_name(args, timestamp='20260719-203208')

        self.assertLessEqual(len(name.encode('utf-8')), 180)


if __name__ == '__main__':
    unittest.main()
