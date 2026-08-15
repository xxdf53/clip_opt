import unittest

import torch
import torch.nn.functional as F

from utils.training_objectives import (
    gradient_conflict_diagnostics,
    hard_fake_reweighting_loss,
)


class HardFakeReweightingTests(unittest.TestCase):
    def test_forward_value_matches_one_extra_selected_fake_bce(self):
        logits = torch.tensor([-0.2, -1.0, 0.5, 0.8])
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
        per_sample = F.binary_cross_entropy_with_logits(
            logits, labels, reduction='none')
        base = per_sample.mean()

        extra, _ = hard_fake_reweighting_loss(
            logits, labels, fraction=0.5)

        expected = (
            per_sample[0]
            + 2 * per_sample[1]
            + per_sample[2]
            + per_sample[3]
        ) / logits.numel()
        self.assertTrue(torch.allclose(base + extra, expected))

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

    def test_auxiliary_gradient_has_zero_common_mode(self):
        logits = torch.tensor(
            [-0.5, -1.0, 0.5, 1.0], requires_grad=True)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])

        loss, _ = hard_fake_reweighting_loss(
            logits, labels, fraction=0.5)
        loss.backward()

        self.assertAlmostEqual(logits.grad.sum().item(), 0.0, places=7)
        self.assertLess(logits.grad[1].item(), 0.0)
        for index in (0, 2, 3):
            self.assertGreater(logits.grad[index].item(), 0.0)

    def test_no_fake_or_too_few_fake_samples_returns_safe_zero(self):
        for labels in (
            torch.zeros(4),
            torch.tensor([0.0, 1.0, 0.0]),
        ):
            logits = torch.zeros(labels.numel(), requires_grad=True)
            loss, diagnostics = hard_fake_reweighting_loss(
                logits, labels, fraction=0.25)
            loss.backward()

            self.assertEqual(loss.item(), 0.0)
            self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)
            self.assertEqual(diagnostics['hard_fake_selected'].item(), 0)

    def test_shard_concatenation_uses_one_global_ranking(self):
        first_logits = torch.tensor([-2.0, 0.3, -0.2])
        second_logits = torch.tensor([-1.5, 0.7])
        labels = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])

        _, diagnostics = hard_fake_reweighting_loss(
            torch.cat([first_logits, second_logits]),
            labels,
            fraction=0.25,
        )

        self.assertAlmostEqual(
            diagnostics['hard_fake_logit_mean'].item(), -1.5)

    def test_gradient_diagnostics_preserve_parameter_gradients(self):
        parameter = torch.tensor([1.0, 1.0], requires_grad=True)
        contrastive = parameter[0] + parameter[1]
        classification = -parameter[0] + 0.5 * parameter[1]

        diagnostics = gradient_conflict_diagnostics(
            contrastive,
            classification,
            [parameter],
        )

        self.assertIsNone(parameter.grad)
        self.assertAlmostEqual(
            diagnostics['gradient_cosine'].item(),
            -0.31622776,
            places=6,
        )
        self.assertEqual(diagnostics['gradient_conflict'].item(), 1.0)
        self.assertAlmostEqual(
            diagnostics['gradient_contrastive_norm'].item(),
            2 ** 0.5,
            places=6,
        )
        self.assertEqual(diagnostics['gradient_shared_numel'].item(), 2.0)

        (contrastive + classification).backward()
        self.assertTrue(torch.allclose(
            parameter.grad,
            torch.tensor([0.0, 1.5]),
        ))


if __name__ == '__main__':
    unittest.main()
