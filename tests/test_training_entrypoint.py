import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.train import discover_evaluation_sets, format_training_losses


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
            loss_anchor=value,
            loss_cpd_direction=value,
            loss_cpd_content=value,
            cpd_schedule_scale=0.5,
            effective_cpd_direction_weight=0.25,
            cpd_signed_projection=value,
            cpd_content_alignment=value,
            cpd_prompt_gap=value,
            real_logit_mean=value,
            fake_logit_mean=value,
            real_anchor_deviation=value,
            fake_anchor_deviation=value,
        )

        text = format_training_losses(model)

        self.assertIn('contrastive=1.250000', text)
        self.assertIn('anchor=1.250000', text)
        self.assertIn('cpd_direction=1.250000', text)
        self.assertIn('cpd_content=1.250000', text)
        self.assertIn('cpd_scale=0.500000', text)
        self.assertIn('cpd_direction_weight=0.250000', text)
        self.assertIn('cpd_projection=1.250000', text)
        self.assertNotIn('rank=', text)


if __name__ == '__main__':
    unittest.main()
