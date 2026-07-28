import unittest

import torch

from utils.training_objectives import (
    counterfactual_prompt_components,
    cpd_content_rejection_loss,
    cpd_direction_loss,
    symmetric_logit_anchor_diagnostics,
    symmetric_logit_anchor_loss,
)


class TrainingObjectiveTests(unittest.TestCase):
    def test_counterfactual_prompts_separate_class_and_content(self):
        real_text = torch.tensor([[1.0, 1.0]])
        fake_text = torch.tensor([[-1.0, 1.0]])

        direction, content = counterfactual_prompt_components(
            real_text, fake_text)

        self.assertTrue(torch.allclose(
            direction, torch.tensor([[-1.0, 0.0]]), atol=1e-6))
        self.assertTrue(torch.allclose(
            content, torch.tensor([[0.0, 1.0]]), atol=1e-6))

    def test_cpd_direction_loss_rewards_label_aligned_residuals(self):
        labels = torch.tensor([0.0, 1.0])
        direction = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        aligned = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        reversed_residual = -aligned

        aligned_loss = cpd_direction_loss(
            aligned, direction, labels, margin=0.1)
        reversed_loss = cpd_direction_loss(
            reversed_residual, direction, labels, margin=0.1)

        self.assertLess(aligned_loss.item(), reversed_loss.item())

    def test_cpd_direction_updates_image_residual_not_frozen_text(self):
        residual = torch.zeros(2, 2, requires_grad=True)
        direction = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0]],
            requires_grad=True,
        )

        loss = cpd_direction_loss(
            residual,
            direction,
            torch.tensor([0.0, 1.0]),
            margin=0.1,
        )
        loss.backward()

        self.assertGreater(residual.grad.abs().sum().item(), 0.0)
        self.assertIsNone(direction.grad)

    def test_cpd_content_rejection_penalizes_content_alignment(self):
        content = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        orthogonal = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        aligned = torch.tensor([[0.0, 1.0], [0.0, -1.0]])

        orthogonal_loss = cpd_content_rejection_loss(
            orthogonal, content)
        aligned_loss = cpd_content_rejection_loss(aligned, content)

        self.assertEqual(orthogonal_loss.item(), 0.0)
        self.assertGreater(aligned_loss.item(), orthogonal_loss.item())

    def test_cpd_objectives_reject_shape_mismatch(self):
        with self.assertRaises(ValueError):
            cpd_direction_loss(
                torch.zeros(2, 3),
                torch.zeros(2, 2),
                torch.zeros(2),
            )

    def test_symmetric_logit_anchor_is_zero_at_both_targets(self):
        loss = symmetric_logit_anchor_loss(
            torch.tensor([-3.0, 3.0]),
            torch.tensor([0.0, 1.0]),
            anchor=3.0,
        )

        self.assertEqual(loss.item(), 0.0)

    def test_symmetric_logit_anchor_penalizes_target_deviation(self):
        labels = torch.tensor([0.0, 1.0])

        near = symmetric_logit_anchor_loss(
            torch.tensor([-2.5, 2.5]), labels, anchor=3.0)
        far = symmetric_logit_anchor_loss(
            torch.tensor([1.0, -1.0]), labels, anchor=3.0)

        self.assertLess(near.item(), far.item())

    def test_symmetric_logit_anchor_pulls_back_overconfident_logits(self):
        logits = torch.tensor([-5.0, 5.0], requires_grad=True)
        loss = symmetric_logit_anchor_loss(
            logits, torch.tensor([0.0, 1.0]), anchor=3.0)

        loss.backward()

        self.assertLess(logits.grad[0].item(), 0.0)
        self.assertGreater(logits.grad[1].item(), 0.0)

    def test_symmetric_logit_anchor_rejects_invalid_anchor(self):
        with self.assertRaises(ValueError):
            symmetric_logit_anchor_loss(
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                anchor=0.0,
            )

    def test_anchor_diagnostics_report_class_means_and_deviations(self):
        diagnostics = symmetric_logit_anchor_diagnostics(
            torch.tensor([-2.0, -4.0, 2.0, 4.0]),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            anchor=3.0,
        )

        self.assertEqual(diagnostics['real_logit_mean'].item(), -3.0)
        self.assertEqual(diagnostics['fake_logit_mean'].item(), 3.0)
        self.assertEqual(diagnostics['real_anchor_deviation'].item(), 1.0)
        self.assertEqual(diagnostics['fake_anchor_deviation'].item(), 1.0)

if __name__ == '__main__':
    unittest.main()
