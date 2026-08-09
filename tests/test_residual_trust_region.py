import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from networks.trainer import CLIPModel_lora
from utils.training_objectives import residual_trust_region_loss


class ResidualTrustRegionTests(unittest.TestCase):
    def test_loss_is_zero_without_feature_drift(self):
        residual = torch.zeros(4, 8)
        self.assertEqual(residual_trust_region_loss(residual).item(), 0.0)

    def test_loss_is_mean_squared_residual_norm(self):
        residual = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        self.assertEqual(residual_trust_region_loss(residual).item(), 5.0)

    def test_training_path_returns_lora_residual(self):
        model = CLIPModel_lora.__new__(CLIPModel_lora)
        nn.Module.__init__(model)
        model.model = nn.Module()
        model.model.visual_projection = nn.Identity()
        model.model.text_projection = nn.Identity()
        model.model.logit_scale = nn.Parameter(torch.tensor(0.0))
        model.model.fc = nn.Linear(4, 1)
        calls = []

        def encode_outputs(images, disable_lora=False):
            calls.append(disable_lora)
            offset = 0.5 if disable_lora else 1.0
            return SimpleNamespace(pooler_output=images + offset)

        model._encode_image_outputs = encode_outputs
        model.encode_text = lambda input_ids, attention_mask: input_ids.float()

        outputs = model(
            torch.randn(3, 4),
            input_ids=torch.randn(3, 4),
            attention_mask=torch.ones(3, 4),
            return_image_residual=True,
        )

        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0].shape, (1,))
        self.assertEqual(outputs[2]['image_residual'].shape, (3, 4))
        self.assertEqual(calls, [False, True])

    def test_image_only_inference_has_no_frozen_forward(self):
        model = CLIPModel_lora.__new__(CLIPModel_lora)
        nn.Module.__init__(model)
        model.model = nn.Module()
        model.model.visual_projection = nn.Identity()
        model.model.fc = nn.Linear(4, 1)
        calls = []

        def encode_outputs(images, disable_lora=False):
            calls.append(disable_lora)
            return SimpleNamespace(pooler_output=images)

        model._encode_image_outputs = encode_outputs
        logits = model(torch.randn(2, 4), cla=True)

        self.assertEqual(logits.shape, (2, 1))
        self.assertEqual(calls, [False])


if __name__ == '__main__':
    unittest.main()
