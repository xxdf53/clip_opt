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

    def test_formats_paired_authenticity_loss(self):
        value = SimpleNamespace(item=lambda: 1.25)
        model = SimpleNamespace(
            loss=value,
            loss_contrastive=value,
            loss_classification=value,
            real_logit_mean=value,
            fake_logit_mean=value,
            paired_authenticity_enabled=True,
            papc_margin_real=value,
            papc_margin_fake=value,
            papc_margin_std_real=value,
            papc_margin_std_fake=value,
            papc_direction_norm=value,
        )

        text = format_training_losses(model)

        self.assertIn('paired_authenticity=1.250000', text)
        self.assertNotIn('contrastive=', text)
        self.assertIn('papc_margin_real=1.250000', text)
        self.assertIn('papc_margin_fake=1.250000', text)
        self.assertIn('papc_margin_std_fake=1.250000', text)
        self.assertIn('papc_direction_norm=1.250000', text)

    def test_formats_hard_fake_diagnostics(self):
        value = SimpleNamespace(item=lambda: 1.25)
        model = SimpleNamespace(
            loss=value,
            loss_contrastive=value,
            loss_classification=value,
            real_logit_mean=value,
            fake_logit_mean=value,
            hard_fake_enabled=True,
            fake_reweighting_mode='uniform',
            loss_hard_fake=value,
            hard_fake_selected=value,
            hard_fake_effective=value,
            hard_fake_total=value,
            hard_fake_logit_mean=value,
        )

        text = format_training_losses(model)

        self.assertIn('fake_reweight_mode=uniform', text)
        self.assertIn('hard_fake=1.250000', text)
        self.assertIn('hard_fake_selected=1', text)
        self.assertIn('hard_fake_effective=1', text)
        self.assertIn('hard_fake_total=1', text)

    def test_formats_hard_real_diagnostics(self):
        value = SimpleNamespace(item=lambda: 1.25)
        model = SimpleNamespace(
            loss=value,
            loss_contrastive=value,
            loss_classification=value,
            real_logit_mean=value,
            fake_logit_mean=value,
            hard_real_enabled=True,
            loss_hard_real=value,
            hard_real_selected=value,
            hard_real_total=value,
            hard_real_logit_mean=value,
        )

        text = format_training_losses(model)

        self.assertIn('hard_real=1.250000', text)
        self.assertIn('hard_real_selected=1', text)
        self.assertIn('hard_real_total=1', text)

    def test_formats_adaptive_hard_diagnostics(self):
        value = SimpleNamespace(item=lambda: 1.25)
        model = SimpleNamespace(
            loss=value,
            loss_contrastive=value,
            loss_classification=value,
            real_logit_mean=value,
            fake_logit_mean=value,
            adaptive_hard_enabled=True,
            loss_adaptive_hard=value,
            adaptive_hard_loss_weight=1.0,
            adaptive_hard_temperature=0.5,
            adaptive_hard_ema_decay=0.9,
            adaptive_hard_warmup_steps=20,
            adaptive_hard_fake_share=value,
            adaptive_hard_real_share=value,
            adaptive_hard_fake_stat=value,
            adaptive_hard_real_stat=value,
            all_fake_bce_mean=value,
            all_real_bce_mean=value,
            relative_fake_score=value,
            relative_real_score=value,
            adaptive_hard_in_warmup=value,
            hard_fake_selected=value,
            hard_real_selected=value,
        )

        text = format_training_losses(model)

        self.assertIn('adaptive_hard=1.250000', text)
        self.assertIn('adaptive_hard_weight=1.000000', text)
        self.assertIn('adaptive_hard_temperature=0.500000', text)
        self.assertIn('adaptive_hard_ema_decay=0.900000', text)
        self.assertIn('adaptive_hard_share_fake=1.250000', text)
        self.assertIn('adaptive_hard_share_real=1.250000', text)
        self.assertIn('adaptive_hard_stat_fake=1.250000', text)
        self.assertIn('all_fake_bce_mean=1.250000', text)
        self.assertIn('all_real_bce_mean=1.250000', text)
        self.assertIn('relative_fake_score=1.250000', text)
        self.assertIn('relative_real_score=1.250000', text)
        self.assertIn('adaptive_hard_fake_selected=1', text)

    def test_rejects_retired_training_flags(self):
        with self.assertRaisesRegex(ValueError, 'retired training options'):
            reject_retired_training_flags([
                '--patch_residual_head',
                '--paired_authenticity_normalize_direction',
                '--augmentation_dro_weight=1',
                '--balanced_bias_calibration',
                '--degradation_consistency_weight=1',
                '--degradation_scale=0.75',
                '--ema_decay=0.99',
                '--classification_referenced_gradient_cap',
                '--gradient_conflict_diagnostics',
                '--hard_fake_semantic_coverage',
                '--gradient_conflict_projection',
                '--paired_authenticity_head_initialization',
                '--residual_vib',
                '--residual_trust_weight=1',
                '--symmetric_prototype_head',
                '--global_contrastive_weight=0.1',
                '--boundary_center_weight=0.5',
                '--semantic_residual_weight=0.1',
                '--spectral_band_dropout=0.25',
                '--rrsd_max_correction=0.5',
            ])


if __name__ == '__main__':
    unittest.main()
