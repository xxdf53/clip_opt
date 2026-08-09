import argparse
import unittest

from options.base_options import BaseOptions, build_experiment_name


class TrainingOptionTests(unittest.TestCase):
    def parse(self, argv):
        parser = BaseOptions().initialize(argparse.ArgumentParser())
        return parser.parse_args(argv)

    def test_sph_defaults_to_disabled(self):
        args = self.parse([])

        self.assertFalse(args.symmetric_prototype_head)

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

    def test_compact_name_records_sph(self):
        args = self.parse([
            '--name', 'c2p_sph',
            '--symmetric_prototype_head',
        ])

        name = build_experiment_name(args, timestamp='20260809-150000')

        self.assertTrue(name.endswith('__sph'))

    def test_name_is_truncated_by_utf8_bytes(self):
        args = self.parse(['--name', 'long_experiment_name_' * 100])

        name = build_experiment_name(args, timestamp='20260727-120000')

        self.assertLessEqual(len(name.encode('utf-8')), 180)


if __name__ == '__main__':
    unittest.main()
