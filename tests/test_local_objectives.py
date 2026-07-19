import unittest

import torch

from utils.local_objectives import (
    confidence_preservation_loss,
    gate_sparsity_loss,
    pairwise_ranking_loss,
)


class LocalObjectiveTests(unittest.TestCase):
    def test_ranking_loss_rewards_fake_logits_above_real_logits(self):
        labels = torch.tensor([0.0, 1.0])

        good = pairwise_ranking_loss(torch.tensor([-2.0, 2.0]), labels)
        bad = pairwise_ranking_loss(torch.tensor([2.0, -2.0]), labels)

        self.assertLess(good.item(), bad.item())

    def test_ranking_loss_is_safe_for_single_class_batch(self):
        logits = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = pairwise_ranking_loss(logits, torch.ones(2))

        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(logits.grad)

    def test_preservation_weights_confident_global_predictions_more(self):
        uncertain = confidence_preservation_loss(
            torch.tensor([1.0]), torch.tensor([0.0]))
        confident = confidence_preservation_loss(
            torch.tensor([6.0]), torch.tensor([5.0]))

        self.assertGreater(confident.item(), uncertain.item())

    def test_gate_sparsity_penalizes_large_gates(self):
        small = gate_sparsity_loss(torch.tensor([0.1, 0.1]))
        large = gate_sparsity_loss(torch.tensor([0.9, 0.9]))

        self.assertLess(small.item(), large.item())


if __name__ == '__main__':
    unittest.main()
