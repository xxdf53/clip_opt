import unittest

import torch

from scripts import test_airplane_official
from scripts import test_checkpoint


class EvaluationCliTests(unittest.TestCase):
    def test_official_cli_accepts_prediction_export(self):
        args = test_airplane_official.parse_args([
            '--dataroot', 'dataset',
            '--model_path', 'official.pth',
            '--predictions_csv', 'official.csv',
        ])

        self.assertEqual(args.predictions_csv, 'official.csv')

    def test_lora_cli_accepts_shared_loader_options(self):
        args = test_checkpoint.parse_args([
            '--dataroot', 'dataset',
            '--checkpoint', 'model.pth',
            '--num_workers', '0',
            '--predictions_csv', 'lora.csv',
        ])

        self.assertEqual(args.num_workers, 0)
        self.assertEqual(args.predictions_csv, 'lora.csv')
        self.assertEqual(args.local_fusion, 'auto')

    def test_lora_cli_accepts_residual_gate_options(self):
        args = test_checkpoint.parse_args([
            '--dataroot', 'dataset',
            '--checkpoint', 'model.pth',
            '--use_local_features',
            '--local_fusion', 'residual_gate',
            '--local_gate_init', '0.02',
        ])

        self.assertEqual(args.local_fusion, 'residual_gate')
        self.assertAlmostEqual(args.local_gate_init, 0.02)

    def test_forward_adapters_preserve_model_specific_signatures(self):
        images = torch.zeros(2, 3, 4, 4)

        class OfficialModel:
            def __call__(self, received_images):
                self.call = (received_images,)
                return torch.ones(2, 1)

        class LoraModel:
            def __call__(self, *args, **kwargs):
                self.call = (args, kwargs)
                return torch.ones(2, 1)

        official = OfficialModel()
        lora = LoraModel()

        test_airplane_official.official_forward_logits(official, images)
        test_checkpoint.lora_forward_logits(lora, images)

        self.assertEqual(official.call, (images,))
        self.assertEqual(lora.call, ((images, None, None), {'cla': True}))


if __name__ == '__main__':
    unittest.main()
