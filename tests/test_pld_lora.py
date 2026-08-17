import unittest

import torch
import torch.nn as nn

from utils.pld_lora import (
    DEFAULT_A_ROW_NORM,
    initialize_patchwise_discriminant_lora,
)


class FakeLoraProjection(nn.Module):
    def __init__(self, features=4, rank=2):
        super().__init__()
        self.lora_A = nn.ModuleDict({
            'default': nn.Linear(features, rank, bias=False),
        })
        self.lora_B = nn.ModuleDict({
            'default': nn.Linear(rank, features, bias=False),
        })

    def forward(self, inputs):
        return inputs + self.lora_B['default'](self.lora_A['default'](inputs))


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = FakeLoraProjection()
        self.k_proj = FakeLoraProjection()
        self.v_proj = FakeLoraProjection()

    def forward(self, inputs):
        return (
            self.q_proj(inputs)
            + self.k_proj(inputs)
            + self.v_proj(inputs)
        )


class FakeVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = FakeAttention()

    def forward(self, images):
        return self.attention(images)


def calibration_batch():
    real = torch.zeros(2, 4, 4)
    fake_sample = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0],
        [2.0, 3.0, 0.0, 0.0],
    ])
    fake = fake_sample.unsqueeze(0).repeat(2, 1, 1)
    return torch.cat((real, fake)), torch.tensor([0, 0, 1, 1])


class PLDLoRATests(unittest.TestCase):
    def test_initializes_discriminative_subspace_and_keeps_b_zero(self):
        torch.manual_seed(7)
        model = FakeVisionModel()
        images, labels = calibration_batch()

        summary = initialize_patchwise_discriminant_lora(
            model,
            images,
            labels,
            forward_images=model,
            microbatch_size=2,
        )

        self.assertEqual(summary.layers, 1)
        self.assertEqual(summary.modules, 3)
        self.assertEqual(summary.real_samples, 2)
        self.assertEqual(summary.fake_samples, 2)
        self.assertEqual(summary.rank, 2)
        self.assertAlmostEqual(summary.explained_energy, 1.0, places=6)

        weights = []
        for projection in (
            model.attention.q_proj,
            model.attention.k_proj,
            model.attention.v_proj,
        ):
            lora_a = projection.lora_A['default'].weight
            lora_b = projection.lora_B['default'].weight
            weights.append(lora_a)
            self.assertTrue(torch.allclose(
                lora_a.norm(dim=1),
                torch.full((2,), DEFAULT_A_ROW_NORM),
            ))
            self.assertTrue(torch.allclose(lora_a[:, 2:], torch.zeros(2, 2)))
            self.assertTrue(torch.count_nonzero(lora_b) == 0)

        self.assertTrue(torch.allclose(weights[0], weights[1]))
        self.assertTrue(torch.allclose(weights[1], weights[2]))

    def test_rejects_single_class_calibration_batch(self):
        model = FakeVisionModel()
        images, _ = calibration_batch()

        with self.assertRaisesRegex(ValueError, 'must contain real and fake'):
            initialize_patchwise_discriminant_lora(
                model,
                images[:2],
                torch.zeros(2),
                forward_images=model,
            )


if __name__ == '__main__':
    unittest.main()
