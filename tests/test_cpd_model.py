import unittest
from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import torch
import torch.nn as nn

from networks.trainer import CLIPModel_lora, Trainer


class FakeVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_enabled = True

    @contextmanager
    def disable_adapter(self):
        previous = self.adapter_enabled
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = previous

    def forward(self, pixel_values, **_):
        offset = 0.25 if self.adapter_enabled else 0.0
        return SimpleNamespace(pooler_output=pixel_values + offset)


def build_minimal_model():
    model = CLIPModel_lora.__new__(CLIPModel_lora)
    nn.Module.__init__(model)
    model.vision_tower_lora = FakeVisionTower()
    model.model = nn.Module()
    model.model.config = SimpleNamespace(output_attentions=False)
    model.model.visual_projection = nn.Identity()
    model.model.logit_scale = nn.Parameter(
        torch.tensor(0.0), requires_grad=False)
    model.model.fc = nn.Linear(2, 1)

    def encode_text(self, input_ids, attention_mask):
        del attention_mask
        return input_ids.to(dtype=torch.float32)

    model.encode_text = MethodType(encode_text, model)
    return model


class CounterfactualPromptModelTests(unittest.TestCase):
    def test_two_micro_batches_produce_one_optimizer_update(self):
        trainer = Trainer.__new__(Trainer)
        trainer.gradient_accumulation_steps = 2
        trainer.micro_steps = 0
        trainer.total_steps = 0
        trainer.optimizer = Mock()

        trainer._begin_gradient_accumulation()
        first_updated = trainer._finish_gradient_accumulation()
        trainer._begin_gradient_accumulation()
        second_updated = trainer._finish_gradient_accumulation()

        self.assertFalse(first_updated)
        self.assertTrue(second_updated)
        self.assertEqual(trainer.micro_steps, 2)
        self.assertEqual(trainer.total_steps, 1)
        trainer.optimizer.zero_grad.assert_called_once_with()
        trainer.optimizer.step.assert_called_once_with()

    def test_trainer_activates_cpd_only_after_scheduled_start(self):
        trainer = Trainer.__new__(Trainer)
        trainer.total_steps = 400
        trainer.cpd_start_step = 400
        trainer.cpd_warmup_steps = 400
        trainer.cpd_direction_weight = 0.5
        trainer.cpd_content_weight = 0.1
        trainer.cpd_enabled = True

        trainer.update_cpd_schedule()

        self.assertFalse(trainer.cpd_active)
        self.assertEqual(trainer.effective_cpd_direction_weight, 0.0)

        trainer.total_steps = 600
        trainer.update_cpd_schedule()

        self.assertTrue(trainer.cpd_active)
        self.assertEqual(trainer.cpd_schedule_scale, 0.5)
        self.assertEqual(trainer.effective_cpd_direction_weight, 0.25)
        self.assertEqual(trainer.effective_cpd_content_weight, 0.05)

    def test_cpd_forward_returns_lora_residual_and_prompt_components(self):
        model = build_minimal_model()
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        counterfactual_ids = torch.tensor([
            [[1, 1], [-1, 1]],
            [[1, 1], [-1, 1]],
        ])
        counterfactual_mask = torch.ones_like(counterfactual_ids)
        labels = torch.tensor([0.0, 1.0])

        contrastive, logits, components = model(
            images,
            cpd_input_ids=counterfactual_ids,
            cpd_attention_mask=counterfactual_mask,
            labels=labels,
            return_cpd=True,
        )

        self.assertEqual(contrastive.shape, (2, 2))
        self.assertEqual(logits.shape, (2,))
        self.assertEqual(components['image_residual'].shape, (2, 2))
        self.assertTrue(torch.any(components['image_residual'] != 0))
        self.assertEqual(
            components['authenticity_direction'].shape, (2, 2))
        self.assertEqual(components['content_center'].shape, (2, 2))

    def test_image_only_inference_does_not_require_cpd_inputs(self):
        model = build_minimal_model()

        logits = model(
            torch.tensor([[1.0, 0.0]]),
            cla=True,
        )

        self.assertEqual(logits.shape, (1, 1))

    def test_cpd_requires_real_fake_prompt_pair(self):
        model = build_minimal_model()

        with self.assertRaisesRegex(ValueError, r'\[batch, 2, sequence\]'):
            model(
                torch.tensor([[1.0, 0.0]]),
                cpd_input_ids=torch.ones(1, 3, 2, dtype=torch.long),
                cpd_attention_mask=torch.ones(
                    1, 3, 2, dtype=torch.long),
                labels=torch.tensor([0.0]),
                return_cpd=True,
            )


if __name__ == '__main__':
    unittest.main()
