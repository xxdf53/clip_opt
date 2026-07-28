import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from data.datasets import ImageFolder2
from utils.cpd import (
    build_counterfactual_captions,
    build_label_caption,
    cpd_is_enabled,
    cpd_schedule_scale,
)


class FakeTokenizer:
    model_max_length = 5

    def __call__(self, texts, **_):
        rows = []
        for text in texts:
            class_token = 10 if text.startswith('Camera.') else 20
            rows.append([class_token, 1, 2, 0, 0])
        input_ids = torch.tensor(rows)
        return {
            'input_ids': input_ids,
            'attention_mask': (input_ids != 0).long(),
        }


class CounterfactualPromptTextTests(unittest.TestCase):
    def test_cpd_schedule_delays_and_linearly_warms_up(self):
        self.assertEqual(cpd_schedule_scale(399, 400, 400), 0.0)
        self.assertEqual(cpd_schedule_scale(400, 400, 400), 0.0)
        self.assertEqual(cpd_schedule_scale(600, 400, 400), 0.5)
        self.assertEqual(cpd_schedule_scale(800, 400, 400), 1.0)
        self.assertEqual(cpd_schedule_scale(1200, 400, 400), 1.0)

    def test_zero_warmup_preserves_fixed_weight_behavior(self):
        self.assertEqual(cpd_schedule_scale(1, 0, 0), 1.0)
        self.assertEqual(cpd_schedule_scale(401, 400, 0), 1.0)

    def test_cpd_schedule_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            cpd_schedule_scale(-1, 0, 0)
        with self.assertRaises(ValueError):
            cpd_schedule_scale(1, -1, 0)
        with self.assertRaises(ValueError):
            cpd_schedule_scale(1, 0, -1)

    def test_builds_real_then_fake_counterfactual_pair(self):
        real_text, fake_text = build_counterfactual_captions(
            'A cat on a chair.',
            ['Deepfake', 'Camera'],
        )

        self.assertEqual(
            real_text,
            'Camera. A cat on a chair. Camera.',
        )
        self.assertEqual(
            fake_text,
            'Deepfake. A cat on a chair. Deepfake.',
        )

    def test_label_caption_matches_binary_target_order(self):
        self.assertTrue(build_label_caption(
            'A car.', ['Deepfake', 'Camera'], target=0
        ).startswith('Camera.'))
        self.assertTrue(build_label_caption(
            'A car.', ['Deepfake', 'Camera'], target=1
        ).startswith('Deepfake.'))

    def test_rejects_odd_category_prompt_configuration(self):
        with self.assertRaises(ValueError):
            build_counterfactual_captions(
                'A cat.', ['Deepfake', 'Camera', 'Extra'])

    def test_cpd_is_enabled_by_either_objective(self):
        class Options:
            cpd_direction_weight = 0.0
            cpd_content_weight = 0.0

        options = Options()
        self.assertFalse(cpd_is_enabled(options))
        options.cpd_direction_weight = 1.0
        self.assertTrue(cpd_is_enabled(options))
        options.cpd_direction_weight = 0.0
        options.cpd_content_weight = 0.1
        self.assertTrue(cpd_is_enabled(options))

    def _build_dataset(self, directory, cpd_direction_weight):
        root = Path(directory) / 'images'
        text_root = Path(directory) / 'captions'
        (root / '0_real').mkdir(parents=True)
        (root / '1_fake').mkdir()
        (text_root / '0_real').mkdir(parents=True)
        (text_root / '1_fake').mkdir()
        for label in ('0_real', '1_fake'):
            Image.new('RGB', (4, 4), 'white').save(
                root / label / 'sample.png')
            (text_root / label / 'sample.txt').write_text(
                'A white square.',
                encoding='utf-8',
            )
        options = SimpleNamespace(
            imgroot=str(root),
            textroot=str(text_root),
            isTrain=True,
            data_aug=False,
            clip='fake-clip',
            cates=['Deepfake', 'Camera'],
            cpd_direction_weight=cpd_direction_weight,
            cpd_content_weight=0.0,
        )
        return ImageFolder2(str(root), options, transform=None)

    def test_dataset_keeps_original_token_shape_when_cpd_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._build_dataset(directory, 0.0)
            with patch(
                'data.datasets._get_tokenizer',
                return_value=FakeTokenizer(),
            ):
                _, _, _, input_ids, attention_mask, _ = dataset[0]

        self.assertEqual(input_ids.shape, (5,))
        self.assertEqual(attention_mask.shape, (5,))

    def test_dataset_returns_real_fake_pair_when_cpd_is_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._build_dataset(directory, 1.0)
            with patch(
                'data.datasets._get_tokenizer',
                return_value=FakeTokenizer(),
            ):
                _, _, _, input_ids, attention_mask, _ = dataset[0]

        self.assertEqual(input_ids.shape, (2, 5))
        self.assertEqual(attention_mask.shape, (2, 5))
        self.assertEqual(input_ids[:, 0].tolist(), [10, 20])


if __name__ == '__main__':
    unittest.main()
