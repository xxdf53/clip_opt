import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from data import collate
from data.datasets import ImageFolder2


class FakeTokenizer:
    model_max_length = 4

    def __call__(self, texts, **_):
        count = len(texts)
        return {
            'input_ids': torch.ones(count, 4, dtype=torch.long),
            'attention_mask': torch.ones(count, 4, dtype=torch.long),
        }


class AugmentationDroTests(unittest.TestCase):
    def test_dataset_returns_the_selected_augmentation_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'images'
            text_root = Path(directory) / 'captions'
            (root / '0_real').mkdir(parents=True)
            (root / '1_fake').mkdir()
            (text_root / '0_real').mkdir(parents=True)
            (text_root / '1_fake').mkdir()
            for label in ('0_real', '1_fake'):
                Image.new('RGB', (8, 8), 'white').save(
                    root / label / 'sample.png')
                (text_root / label / 'sample.txt').write_text(
                    'A white square.', encoding='utf-8')

            options = SimpleNamespace(
                imgroot=str(root),
                textroot=str(text_root),
                isTrain=True,
                data_aug=False,
                augmentation_dro_weight=1.0,
                jpg_method=['pil'],
                jpg_qual=[75],
                blur_sig=[0.5],
                clip='fake-clip',
                cates=['Deepfake', 'Camera'],
                cpd_direction_weight=0.0,
                cpd_content_weight=0.0,
            )
            dataset = ImageFolder2(str(root), options, transform=None)

            with (
                patch('data.datasets.randrange', return_value=1),
                patch(
                    'data.datasets._get_tokenizer',
                    return_value=FakeTokenizer(),
                ),
            ):
                sample = dataset[0]

        self.assertEqual(len(sample), 7)
        self.assertEqual(sample[6], 1)

    def test_collate_appends_group_ids_only_when_present(self):
        base = (
            'image.png',
            torch.zeros(3, 2, 2),
            'caption',
            torch.ones(4, dtype=torch.long),
            torch.ones(4, dtype=torch.long),
            0,
        )

        self.assertEqual(len(collate([base])), 6)
        grouped = collate([base + (2,)])
        self.assertEqual(len(grouped), 7)
        self.assertEqual(grouped[6].tolist(), [2])


if __name__ == '__main__':
    unittest.main()
