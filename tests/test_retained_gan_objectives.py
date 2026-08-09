import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image

from data.datasets import ImageFolder2
from networks.trainer import CLIPModel_lora, Trainer
from utils.captions import build_label_caption
from utils.cpd import (
    build_counterfactual_captions,
    cpd_is_enabled,
    cpd_schedule_scale,
)
from utils.training_objectives import (
    cpd_direction_loss,
    symmetric_logit_anchor_loss,
)


class FakeTokenizer:
    model_max_length = 5

    def __call__(self, texts, **_):
        rows = []
        for text in texts:
            first_token = 10 if text.startswith('Camera.') else 20
            rows.append([first_token, 1, 2, 0, 0])
        input_ids = torch.tensor(rows)
        return {
            'input_ids': input_ids,
            'attention_mask': (input_ids != 0).long(),
        }


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
    model.residual_vib = None

    def encode_text(self, input_ids, attention_mask):
        del attention_mask
        return input_ids.to(dtype=torch.float32)

    model.encode_text = MethodType(encode_text, model)
    return model


class RetainedGanObjectiveTests(unittest.TestCase):
    def test_cpd_schedule_delays_and_warms_up(self):
        self.assertEqual(cpd_schedule_scale(400, 400, 400), 0.0)
        self.assertEqual(cpd_schedule_scale(600, 400, 400), 0.5)
        self.assertEqual(cpd_schedule_scale(800, 400, 400), 1.0)

    def test_counterfactual_text_order_matches_binary_labels(self):
        pair = build_counterfactual_captions(
            'A cat.', ['Deepfake', 'Camera'])
        self.assertTrue(pair[0].startswith('Camera.'))
        self.assertTrue(pair[1].startswith('Deepfake.'))
        self.assertTrue(build_label_caption(
            'A cat.', ['Deepfake', 'Camera'], 0).startswith('Camera.'))

    def test_cpd_dataset_returns_prompt_pair_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'images'
            captions = Path(directory) / 'captions'
            for label in ('0_real', '1_fake'):
                (root / label).mkdir(parents=True)
                (captions / label).mkdir(parents=True)
                Image.new('RGB', (4, 4), 'white').save(
                    root / label / 'sample.png')
                (captions / label / 'sample.txt').write_text(
                    'A white square.', encoding='utf-8')
            options = SimpleNamespace(
                imgroot=str(root),
                textroot=str(captions),
                isTrain=True,
                data_aug=False,
                clip='fake-clip',
                cates=['Deepfake', 'Camera'],
                cpd_direction_weight=0.5,
                cpd_content_weight=0.0,
            )
            dataset = ImageFolder2(str(root), options, transform=None)
            with patch(
                'data.datasets._get_tokenizer',
                return_value=FakeTokenizer(),
            ):
                sample = dataset[0]

        self.assertEqual(sample[3].shape, (2, 5))
        self.assertEqual(sample[3][:, 0].tolist(), [10, 20])

    def test_cpd_forward_returns_shared_lora_residual(self):
        model = build_minimal_model()
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        paired_ids = torch.tensor([
            [[1, 1], [-1, 1]],
            [[1, 1], [-1, 1]],
        ])
        labels = torch.tensor([0.0, 1.0])

        contrastive, logits, auxiliary = model(
            images,
            cpd_input_ids=paired_ids,
            cpd_attention_mask=torch.ones_like(paired_ids),
            labels=labels,
            return_cpd=True,
        )

        self.assertEqual(contrastive.shape, (2, 2))
        self.assertEqual(logits.shape, (2,))
        self.assertEqual(auxiliary['image_residual'].shape, (2, 2))
        self.assertTrue(torch.any(auxiliary['image_residual'] != 0))

    def test_cpd_and_slar_losses_favor_correct_separation(self):
        residual = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        direction = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        labels = torch.tensor([0.0, 1.0])
        correct = cpd_direction_loss(residual, direction, labels)
        reversed_loss = cpd_direction_loss(-residual, direction, labels)
        self.assertLess(correct, reversed_loss)

        anchored = symmetric_logit_anchor_loss(
            torch.tensor([-3.0, 3.0]), labels, anchor=3.0)
        self.assertEqual(anchored.item(), 0.0)

    def test_cpd_configuration_and_image_only_inference(self):
        options = SimpleNamespace(
            cpd_direction_weight=0.5,
            cpd_content_weight=0.0,
        )
        self.assertTrue(cpd_is_enabled(options))

        logits = build_minimal_model()(torch.tensor([[1.0, 0.0]]), cla=True)
        self.assertEqual(logits.shape, (1, 1))

    def test_trainer_schedule_uses_configured_weights(self):
        trainer = Trainer.__new__(Trainer)
        trainer.total_steps = 600
        trainer.cpd_start_step = 400
        trainer.cpd_warmup_steps = 400
        trainer.cpd_direction_weight = 0.5
        trainer.cpd_content_weight = 0.1
        trainer.cpd_enabled = True

        trainer.update_cpd_schedule()

        self.assertTrue(trainer.cpd_active)
        self.assertEqual(trainer.cpd_schedule_scale, 0.5)
        self.assertEqual(trainer.effective_cpd_direction_weight, 0.25)
        self.assertEqual(trainer.effective_cpd_content_weight, 0.05)


if __name__ == '__main__':
    unittest.main()
