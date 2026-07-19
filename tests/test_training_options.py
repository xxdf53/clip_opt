import argparse
import unittest

from options.base_options import BaseOptions


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


if __name__ == '__main__':
    unittest.main()
