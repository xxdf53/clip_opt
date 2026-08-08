import tempfile
import unittest
from pathlib import Path

from data import _read_training_manifest
from scripts.create_training_manifest import discover_images


class SampleDataset:
    def __init__(self, paths):
        self.samples = [(str(path), index % 2) for index, path in enumerate(paths)]


class TrainingManifestTests(unittest.TestCase):
    def test_manifest_preserves_requested_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / 'train'
            real = root / '0_real' / 'real.png'
            fake = root / '1_fake' / 'fake.png'
            real.parent.mkdir(parents=True)
            fake.parent.mkdir(parents=True)
            real.touch()
            fake.touch()
            manifest = Path(temporary_directory) / 'manifest.txt'
            manifest.write_text(
                '# metadata\n1_fake/fake.png\n0_real/real.png\n',
                encoding='utf-8',
            )

            indices = _read_training_manifest(
                SampleDataset([real, fake]), root, manifest)

            self.assertEqual(indices, [1, 0])

    def test_manifest_rejects_duplicate_entries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / 'train'
            image = root / '0_real' / 'real.png'
            image.parent.mkdir(parents=True)
            image.touch()
            manifest = Path(temporary_directory) / 'manifest.txt'
            manifest.write_text(
                '0_real/real.png\n0_real/real.png\n', encoding='utf-8')

            with self.assertRaisesRegex(ValueError, 'duplicate'):
                _read_training_manifest(
                    SampleDataset([image]), root, manifest)

    def test_discovery_uses_portable_sorted_relative_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / '1_fake').mkdir()
            (root / '0_real').mkdir()
            (root / '1_fake' / 'b.JPEG').touch()
            (root / '0_real' / 'a.png').touch()
            (root / '0_real' / 'ignore.txt').touch()

            self.assertEqual(
                discover_images(root),
                ['0_real/a.png', '1_fake/b.JPEG'],
            )


if __name__ == '__main__':
    unittest.main()
