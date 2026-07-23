import unittest

import torch

from utils.local_objectives import (
    confidence_preservation_loss,
    gate_sparsity_loss,
    pairwise_ranking_loss,
    relative_gate_supervision_loss,
    relative_gate_target,
    residual_candidate_loss,
    symmetric_logit_anchor_loss,
    zero_threshold_margin_loss,
)


class LocalObjectiveTests(unittest.TestCase):
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

    def test_zero_threshold_margin_is_zero_past_both_margins(self):
        loss = zero_threshold_margin_loss(
            torch.tensor([-1.5, 1.5]),
            torch.tensor([0.0, 1.0]),
            margin=1.0,
        )

        self.assertEqual(loss.item(), 0.0)

    def test_zero_threshold_margin_penalizes_wrong_side_logits(self):
        labels = torch.tensor([0.0, 1.0])

        correct = zero_threshold_margin_loss(
            torch.tensor([-0.5, 0.5]), labels, margin=1.0)
        wrong = zero_threshold_margin_loss(
            torch.tensor([0.5, -0.5]), labels, margin=1.0)

        self.assertLess(correct.item(), wrong.item())

    def test_zero_threshold_margin_provides_logit_gradients(self):
        logits = torch.tensor([0.5, -0.5], requires_grad=True)
        loss = zero_threshold_margin_loss(
            logits, torch.tensor([0.0, 1.0]), margin=1.0)

        loss.backward()

        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_zero_threshold_margin_rejects_invalid_margin(self):
        with self.assertRaises(ValueError):
            zero_threshold_margin_loss(
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                margin=0.0,
            )

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

    def test_residual_candidate_trains_only_local_correction(self):
        global_logits = torch.tensor([-2.0, 2.0], requires_grad=True)
        local_logits = torch.tensor([0.0, 0.0], requires_grad=True)
        labels = torch.tensor([1.0, 0.0])

        loss = residual_candidate_loss(
            global_logits, local_logits, labels)
        loss.backward()

        self.assertIsNone(global_logits.grad)
        self.assertGreater(local_logits.grad.abs().sum().item(), 0.0)

    def test_relative_gate_target_opens_only_for_helpful_residuals(self):
        targets = relative_gate_target(
            global_logits=torch.tensor([-2.0, 2.0]),
            local_logits=torch.tensor([4.0, -4.0]),
            labels=torch.tensor([1.0, 1.0]),
            margin=0.1,
        )

        self.assertEqual(targets[0].item(), 1.0)
        self.assertEqual(targets[1].item(), 0.0)

    def test_relative_gate_target_suppresses_tiny_confident_improvement(self):
        target = relative_gate_target(
            global_logits=torch.tensor([5.0]),
            local_logits=torch.tensor([1.0]),
            labels=torch.tensor([1.0]),
            margin=0.1,
        )

        self.assertGreater(target.item(), 0.0)
        self.assertLess(target.item(), 0.1)

    def test_gate_supervision_rewards_matching_reliability_target(self):
        inputs = dict(
            global_logits=torch.tensor([-2.0, 2.0]),
            local_logits=torch.tensor([4.0, -4.0]),
            labels=torch.tensor([1.0, 1.0]),
            margin=0.1,
        )

        matching, targets = relative_gate_supervision_loss(
            torch.tensor([0.99, 0.01]), **inputs)
        inverted, _ = relative_gate_supervision_loss(
            torch.tensor([0.01, 0.99]), **inputs)

        self.assertTrue(torch.equal(targets, torch.tensor([1.0, 0.0])))
        self.assertLess(matching.item(), inverted.item())

    def test_gate_supervision_cannot_change_its_detached_target(self):
        gates = torch.tensor([0.5], requires_grad=True)
        global_logits = torch.tensor([-2.0], requires_grad=True)
        local_logits = torch.tensor([4.0], requires_grad=True)

        loss, _ = relative_gate_supervision_loss(
            gates,
            global_logits,
            local_logits,
            torch.tensor([1.0]),
            margin=0.1,
        )
        loss.backward()

        self.assertIsNotNone(gates.grad)
        self.assertIsNone(global_logits.grad)
        self.assertIsNone(local_logits.grad)

    def test_relative_gate_target_rejects_non_positive_margin(self):
        with self.assertRaises(ValueError):
            relative_gate_target(
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                margin=0.0,
            )


if __name__ == '__main__':
    unittest.main()
