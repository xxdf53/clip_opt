import tempfile
import unittest
from pathlib import Path

from data import _read_training_manifest
from scripts.create_g10_stratified_manifest import (
    select_batch_balanced_paths,
    validate_selected_captions,
    write_manifest_bundle,
)
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

    def test_g10_selection_balances_every_batch(self):
        strata = {}
        for category in ('car', 'cat'):
            for label in ('0_real', '1_fake'):
                stratum = f'{category}/{label}'
                strata[stratum] = [
                    f'{stratum}/{index}.png' for index in range(8)
                ]

        ordered, selected, per_batch, total_steps = (
            select_batch_balanced_paths(
                strata,
                per_stratum=4,
                batch_size=4,
                data_seed=271828,
            )
        )

        self.assertEqual(per_batch, 1)
        self.assertEqual(total_steps, 4)
        self.assertEqual(len(ordered), 16)
        self.assertEqual(len(set(ordered)), 16)
        self.assertTrue(all(len(paths) == 4 for paths in selected.values()))
        for start in range(0, len(ordered), 4):
            batch = ordered[start:start + 4]
            self.assertEqual(
                {Path(path).parent.as_posix() for path in batch},
                set(strata),
            )

    def test_g10_selection_is_deterministic(self):
        strata = {
            'car/0_real': [f'car/0_real/{index}.png' for index in range(8)],
            'car/1_fake': [f'car/1_fake/{index}.png' for index in range(8)],
        }

        first = select_batch_balanced_paths(
            strata, per_stratum=4, batch_size=2, data_seed=123)[0]
        second = select_batch_balanced_paths(
            strata, per_stratum=4, batch_size=2, data_seed=123)[0]
        different_seed = select_batch_balanced_paths(
            strata, per_stratum=4, batch_size=2, data_seed=42)[0]

        self.assertEqual(first, second)
        self.assertNotEqual(first, different_seed)

    def test_g10_caption_validation_rejects_missing_caption(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            textroot = Path(temporary_directory)
            caption = textroot / 'car' / '0_real' / 'present.txt'
            caption.parent.mkdir(parents=True)
            caption.write_text('A car.', encoding='utf-8')

            with self.assertRaisesRegex(ValueError, 'missing=1'):
                validate_selected_captions(
                    [
                        'car/0_real/present.png',
                        'car/0_real/missing.png',
                    ],
                    textroot,
                )

    def test_g10_manifest_bundle_records_balanced_audit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataroot = root / 'images'
            textroot = root / 'captions'
            for category in ('car', 'cat'):
                for label in ('0_real', '1_fake'):
                    for index in range(2):
                        relative = Path(category) / label / f'{index}.png'
                        image = dataroot / relative
                        caption = (textroot / relative).with_suffix('.txt')
                        image.parent.mkdir(parents=True, exist_ok=True)
                        caption.parent.mkdir(parents=True, exist_ok=True)
                        image.touch()
                        caption.write_text('Caption.', encoding='utf-8')
            manifest = root / 'manifest.txt'
            metadata_path = root / 'manifest.json'

            metadata, manifest_sha256, metadata_sha256 = (
                write_manifest_bundle(
                    dataroot=dataroot,
                    textroot=textroot,
                    output=manifest,
                    metadata_output=metadata_path,
                    categories=('car', 'cat'),
                    labels=('0_real', '1_fake'),
                    per_stratum=2,
                    batch_size=4,
                    data_seed=271828,
                )
            )

            self.assertEqual(metadata['selected_count'], 8)
            self.assertEqual(metadata['total_steps'], 2)
            self.assertEqual(metadata['per_stratum_per_batch'], 1)
            self.assertEqual(metadata['missing_caption_count'], 0)
            self.assertEqual(len(manifest_sha256), 64)
            self.assertEqual(len(metadata_sha256), 64)
            self.assertTrue(manifest.is_file())
            self.assertTrue(metadata_path.is_file())


if __name__ == '__main__':
    unittest.main()
