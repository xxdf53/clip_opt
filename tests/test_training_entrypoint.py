import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.train import (
    discover_evaluation_sets,
    format_training_losses,
    reject_retired_training_flags,
)


class TrainingEntrypointTests(unittest.TestCase):
    def test_discovers_only_sorted_evaluation_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'stylegan').mkdir()
            (root / 'biggan').mkdir()
            (root / 'notes.txt').write_text('not a dataset')

            self.assertEqual(
                discover_evaluation_sets(root),
                ['biggan', 'stylegan'],
            )

    def test_rejects_empty_evaluation_root(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'no evaluation subsets'):
                discover_evaluation_sets(directory)

    def test_formats_active_training_losses(self):
        value = SimpleNamespace(item=lambda: 1.25)
        model = SimpleNamespace(
            loss=value,
            loss_contrastive=value,
            loss_classification=value,
            real_logit_mean=value,
            fake_logit_mean=value,
        )

        text = format_training_losses(model)

        self.assertIn('contrastive=1.250000', text)
        self.assertIn('classification=1.250000', text)
        self.assertIn('logit_real=1.250000', text)
        self.assertIn('logit_fake=1.250000', text)

    def test_formats_spectral_band_dropout_diagnostics(self):
        value = SimpleNamespace(item=lambda: 1.25)
        model = SimpleNamespace(
            loss=value,
            loss_contrastive=value,
            loss_classification=value,
            real_logit_mean=value,
            fake_logit_mean=value,
            spectral_band_dropout=0.25,
            spectral_band_applied=value,
            spectral_band_mask_fraction=value,
        )

        text = format_training_losses(model)

        self.assertIn('sbd_probability=0.25', text)
        self.assertIn('sbd_applied=1.250000', text)

    def test_rejects_retired_training_flags(self):
        with self.assertRaisesRegex(ValueError, 'retired training options'):
            reject_retired_training_flags([
                '--patch_residual_head',
                '--augmentation_dro_weight=1',
                '--degradation_consistency_weight=1',
                '--degradation_scale=0.75',
                '--ema_decay=0.99',
                '--residual_vib',
                '--residual_trust_weight=1',
                '--symmetric_prototype_head',
                '--global_contrastive_weight=0.1',
                '--boundary_center_weight=0.5',
                '--semantic_residual_weight=0.1',
            ])


if __name__ == '__main__':
    unittest.main()
