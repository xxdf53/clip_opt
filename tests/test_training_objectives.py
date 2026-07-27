import unittest

import torch

from utils.training_objectives import (
    symmetric_logit_anchor_diagnostics,
    symmetric_logit_anchor_loss,
)


class TrainingObjectiveTests(unittest.TestCase):
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
