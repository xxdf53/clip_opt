import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from networks.trainer import (
    CLIPModel_lora,
    ResidualVariationalBottleneck,
)


class ResidualVariationalBottleneckTests(unittest.TestCase):
    def test_outputs_have_expected_shapes_and_finite_kl(self):
        bottleneck = ResidualVariationalBottleneck(8, latent_dim=3)

        logits, mu, logvar = bottleneck(torch.randn(5, 8))
        kl = bottleneck.kl_divergence(mu, logvar)

        self.assertEqual(logits.shape, (5,))
        self.assertEqual(mu.shape, (5, 3))
        self.assertEqual(logvar.shape, (5, 3))
        self.assertTrue(torch.isfinite(kl))
        self.assertGreaterEqual(kl.item(), 0.0)

    def test_training_path_uses_lora_residual(self):
        model = CLIPModel_lora.__new__(CLIPModel_lora)
        nn.Module.__init__(model)
        model.model = nn.Module()
        model.model.visual_projection = nn.Identity()
        model.model.text_projection = nn.Identity()
        model.model.logit_scale = nn.Parameter(torch.tensor(0.0))
        model.model.fc = nn.Linear(4, 1)
        model.residual_vib = ResidualVariationalBottleneck(4, latent_dim=2)
        calls = []

        def encode_outputs(images, disable_lora=False):
            calls.append(disable_lora)
            offset = 0.5 if disable_lora else 1.0
            return SimpleNamespace(pooler_output=images + offset)

        model._encode_image_outputs = encode_outputs
        model.encode_text = lambda input_ids, attention_mask: input_ids.float()
        images = torch.randn(3, 4)

        outputs = model(
            images,
            input_ids=torch.randn(3, 4),
            attention_mask=torch.ones(3, 4),
        )

        self.assertEqual(len(outputs), 3)
        auxiliary = outputs[2]
        self.assertEqual(auxiliary['vib_logits'].shape, (3,))
        self.assertEqual(auxiliary['vib_mu'].shape, (3, 2))
        self.assertEqual(auxiliary['vib_logvar'].shape, (3, 2))
        self.assertEqual(calls, [False, True])

    def test_image_only_inference_skips_bottleneck_and_frozen_forward(self):
        model = CLIPModel_lora.__new__(CLIPModel_lora)
        nn.Module.__init__(model)
        model.model = nn.Module()
        model.model.visual_projection = nn.Identity()
        model.model.fc = nn.Linear(4, 1)
        model.residual_vib = ResidualVariationalBottleneck(4, latent_dim=2)
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
