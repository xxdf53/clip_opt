import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from networks.trainer import CLIPModel_lora, SymmetricPrototypeHead


class SymmetricPrototypeHeadTests(unittest.TestCase):
    def test_logits_are_bounded(self):
        head = SymmetricPrototypeHead(feature_size=4)

        logits = head(torch.randn(32, 4))

        self.assertTrue(torch.all(logits <= 2.0))
        self.assertTrue(torch.all(logits >= -2.0))

    def test_swapping_prototypes_flips_logit_sign(self):
        head = SymmetricPrototypeHead(feature_size=4)
        embeddings = torch.randn(3, 4)
        original = head(embeddings)
        with torch.no_grad():
            real = head.real_prototype.clone()
            head.real_prototype.copy_(head.fake_prototype)
            head.fake_prototype.copy_(real)

        swapped = head(embeddings)

        torch.testing.assert_close(swapped, -original)

    def test_both_prototypes_receive_gradients(self):
        head = SymmetricPrototypeHead(feature_size=4)

        head(torch.randn(4, 4)).square().mean().backward()

        self.assertGreater(head.real_prototype.grad.abs().sum().item(), 0.0)
        self.assertGreater(head.fake_prototype.grad.abs().sum().item(), 0.0)

    def test_image_only_forward_uses_sph(self):
        model = CLIPModel_lora.__new__(CLIPModel_lora)
        nn.Module.__init__(model)
        model.model = nn.Module()
        model.model.visual_projection = nn.Identity()
        model.model.fc = SymmetricPrototypeHead(feature_size=4)
        model.patch_residual_head = None

        def encode_outputs(images, disable_lora=False):
            del disable_lora
            return SimpleNamespace(pooler_output=images)

        model._encode_image_outputs = encode_outputs

        logits = model(torch.randn(2, 4), cla=True)

        self.assertEqual(logits.shape, (2, 1))

    def test_prh_and_sph_cannot_be_combined(self):
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            CLIPModel_lora(
                patch_residual_head=True,
                symmetric_prototype_head=True,
            )


if __name__ == '__main__':
    unittest.main()
