import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from networks.trainer import CLIPModel_lora, PatchResidualHead


class PatchResidualHeadTests(unittest.TestCase):
    def test_zero_initialization_preserves_baseline_logits(self):
        head = PatchResidualHead(hidden_size=4, bottleneck_size=3)
        hidden_states = torch.randn(2, 17, 4)

        logits = head(hidden_states)

        torch.testing.assert_close(logits, torch.zeros(2, 1))

    def test_head_receives_gradient_from_binary_logit(self):
        head = PatchResidualHead(hidden_size=4, bottleneck_size=3)
        hidden_states = torch.randn(2, 17, 4)

        loss = (head(hidden_states) - 1.0).square().mean()
        loss.backward()

        gradient = head.mlp[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_rejects_non_square_patch_grid(self):
        head = PatchResidualHead(hidden_size=4, bottleneck_size=3)

        with self.assertRaisesRegex(ValueError, 'square patch grid'):
            head(torch.randn(2, 16, 4))

    def test_image_only_forward_fuses_patch_logit(self):
        model = CLIPModel_lora.__new__(CLIPModel_lora)
        nn.Module.__init__(model)
        model.model = nn.Module()
        model.model.visual_projection = nn.Identity()
        model.model.fc = nn.Linear(4, 1)
        nn.init.zeros_(model.model.fc.weight)
        nn.init.zeros_(model.model.fc.bias)
        model.patch_residual_head = PatchResidualHead(
            hidden_size=4,
            bottleneck_size=3,
        )
        nn.init.zeros_(model.patch_residual_head.mlp[-1].weight)
        nn.init.ones_(model.patch_residual_head.mlp[-1].bias)

        def encode_outputs(images, disable_lora=False):
            del disable_lora
            return SimpleNamespace(
                pooler_output=images,
                last_hidden_state=torch.randn(images.shape[0], 17, 4),
            )

        model._encode_image_outputs = encode_outputs

        logits = model(torch.randn(2, 4), cla=True)

        torch.testing.assert_close(logits, torch.ones(2, 1))


if __name__ == '__main__':
    unittest.main()
