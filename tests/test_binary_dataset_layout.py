import tempfile
import unittest
from pathlib import Path

from utils.binary_dataset_layout import discover_binary_groups


def make_binary_leaf(path):
    (path / '0_real').mkdir(parents=True)
    (path / '1_fake').mkdir()


class BinaryDatasetLayoutTests(unittest.TestCase):
    def test_direct_binary_root_is_one_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'my_first_test'
            make_binary_leaf(root)

            groups = discover_binary_groups(root)

            self.assertEqual(groups, {'my_first_test': [root.resolve()]})

    def test_groups_direct_and_nested_leaves_by_top_level_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'CNN_synth_testset'
            make_binary_leaf(root / 'biggan')
            make_binary_leaf(root / 'cyclegan' / 'horse')
            make_binary_leaf(root / 'cyclegan' / 'apple')
            make_binary_leaf(root / 'stylegan' / 'cat')

            groups = discover_binary_groups(root)

            self.assertEqual(list(groups), ['biggan', 'cyclegan', 'stylegan'])
            self.assertEqual(groups['biggan'], [(root / 'biggan').resolve()])
            self.assertEqual(groups['cyclegan'], [
                (root / 'cyclegan' / 'apple').resolve(),
                (root / 'cyclegan' / 'horse').resolve(),
            ])

    def test_rejects_an_unpaired_label_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'dataset'
            (root / 'broken' / '0_real').mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, 'missing 1_fake'):
                discover_binary_groups(root)

    def test_rejects_root_without_binary_leaves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'dataset'
            (root / 'empty').mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, 'no paired 0_real/1_fake'):
                discover_binary_groups(root)

    def test_rejects_missing_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'missing'

            with self.assertRaisesRegex(FileNotFoundError, 'dataset root not found'):
                discover_binary_groups(root)


if __name__ == '__main__':
    unittest.main()
