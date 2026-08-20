import unittest

import torch
import torch.nn.functional as F

from utils.training_objectives import (
    hard_fake_reweighting_loss,
    hard_real_reweighting_loss,
)


class HardFakeReweightingTests(unittest.TestCase):
    def test_weight_one_doubles_only_selected_fake_bce_weight(self):
        logits = torch.tensor([-0.2, -1.0, 0.5, 0.8])
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
        per_sample = F.binary_cross_entropy_with_logits(
            logits, labels, reduction='none')

        extra, _ = hard_fake_reweighting_loss(
            logits, labels, fraction=0.5)

        expected = (
            per_sample[0]
            + 2 * per_sample[1]
            + per_sample[2]
            + per_sample[3]
        ) / logits.numel()
        self.assertTrue(torch.allclose(per_sample.mean() + extra, expected))

    def test_selects_lowest_logit_fake_samples_from_global_batch(self):
        logits = torch.tensor([-2.0, 0.4, -0.1, 1.0, -1.0, 0.2])
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 1.0])

        loss, diagnostics = hard_fake_reweighting_loss(
            logits, labels, fraction=0.5)
        expected = F.binary_cross_entropy_with_logits(
            logits[[2, 4]], labels[[2, 4]], reduction='sum') / logits.numel()

        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(diagnostics['hard_fake_selected'].item(), 2)
        self.assertEqual(diagnostics['hard_fake_total'].item(), 4)

    def test_gradient_changes_only_selected_fake_sample(self):
        logits = torch.tensor(
            [-0.5, -1.0, 0.5, 1.0], requires_grad=True)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])

        loss, _ = hard_fake_reweighting_loss(logits, labels, fraction=0.5)
        loss.backward()

        self.assertEqual(torch.count_nonzero(logits.grad).item(), 1)
        self.assertNotEqual(logits.grad[1].item(), 0.0)

    def test_no_fake_or_too_few_fake_samples_returns_safe_zero(self):
        for labels in (torch.zeros(4), torch.tensor([0.0, 1.0, 0.0])):
            logits = torch.zeros(labels.numel(), requires_grad=True)
            loss, diagnostics = hard_fake_reweighting_loss(
                logits, labels, fraction=0.25)
            loss.backward()

            self.assertEqual(loss.item(), 0.0)
            self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)
            self.assertEqual(diagnostics['hard_fake_selected'].item(), 0)

    def test_concatenated_shards_use_one_global_ranking(self):
        first_logits = torch.tensor([-0.2, 0.3])
        second_logits = torch.tensor([-1.5, 0.7])

        _, diagnostics = hard_fake_reweighting_loss(
            torch.cat([first_logits, second_logits]),
            torch.ones(4),
            fraction=0.25,
        )

        self.assertAlmostEqual(
            diagnostics['hard_fake_logit_mean'].item(), -1.5)

    def test_hard_real_selects_highest_logit_real_samples(self):
        logits = torch.tensor([-2.0, 0.4, -0.1, 1.0, -1.0, 0.2])
        labels = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])

        loss, diagnostics = hard_real_reweighting_loss(
            logits, labels, fraction=0.5)
        expected = F.binary_cross_entropy_with_logits(
            logits[[3, 5]], labels[[3, 5]], reduction='sum') / logits.numel()

        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(diagnostics['hard_real_selected'].item(), 2)
        self.assertEqual(diagnostics['hard_real_total'].item(), 4)
        self.assertAlmostEqual(
            diagnostics['hard_real_logit_mean'].item(), 0.6)

    def test_budget_matched_symmetric_reweighting_splits_weight(self):
        logits = torch.tensor([-0.5, -1.0, 0.5, 1.0])
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
        per_sample = F.binary_cross_entropy_with_logits(
            logits, labels, reduction='none')

        hard_fake, _ = hard_fake_reweighting_loss(
            logits, labels, fraction=0.5)
        hard_real, _ = hard_real_reweighting_loss(
            logits, labels, fraction=0.5)
        combined = per_sample.mean() + 0.5 * hard_fake + 0.5 * hard_real
        expected = (
            per_sample[0]
            + 1.5 * per_sample[1]
            + per_sample[2]
            + 1.5 * per_sample[3]
        ) / logits.numel()

        self.assertTrue(torch.allclose(combined, expected))


if __name__ == '__main__':
    unittest.main()
