import unittest

from utils.captions import build_label_caption


class CaptionTests(unittest.TestCase):
    def test_label_caption_preserves_original_binary_prompt_order(self):
        self.assertEqual(
            build_label_caption('A car.', ['Deepfake', 'Camera'], target=0),
            'Camera. A car. Camera.',
        )
        self.assertEqual(
            build_label_caption('A car.', ['Deepfake', 'Camera'], target=1),
            'Deepfake. A car. Deepfake.',
        )

    def test_rejects_invalid_binary_target(self):
        with self.assertRaisesRegex(ValueError, 'binary target'):
            build_label_caption('A car.', ['Deepfake', 'Camera'], target=2)


if __name__ == '__main__':
    unittest.main()
