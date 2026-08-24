import unittest

import torch
import torch.nn.functional as F

from utils.training_objectives import (
    AdaptiveHardLossController,
    fake_reweighting_loss,
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

    def test_random_mode_is_count_matched_and_seed_reproducible(self):
        logits = torch.tensor([-2.0, 0.4, -0.1, 1.0, -1.0, 0.2])
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        first_generator = torch.Generator().manual_seed(271828)
        second_generator = torch.Generator().manual_seed(271828)

        first_loss, first_diagnostics = fake_reweighting_loss(
            logits,
            labels,
            fraction=0.5,
            mode='random',
            generator=first_generator,
        )
        second_loss, second_diagnostics = fake_reweighting_loss(
            logits,
            labels,
            fraction=0.5,
            mode='random',
            generator=second_generator,
        )

        self.assertTrue(torch.equal(first_loss, second_loss))
        self.assertEqual(first_diagnostics['hard_fake_selected'].item(), 2)
        self.assertEqual(first_diagnostics['hard_fake_effective'].item(), 2)
        self.assertTrue(torch.equal(
            first_diagnostics['hard_fake_logit_mean'],
            second_diagnostics['hard_fake_logit_mean'],
        ))

    def test_uniform_mode_matches_selected_sample_weight_budget(self):
        logits = torch.tensor([-2.0, 0.4, -0.1, 1.0, -1.0, 0.2, 0.8])
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        per_sample = F.binary_cross_entropy_with_logits(
            logits, labels, reduction='none')

        loss, diagnostics = fake_reweighting_loss(
            logits, labels, fraction=0.25, mode='uniform')
        expected = 0.2 * per_sample[[1, 2, 4, 5, 6]].sum() / logits.numel()

        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(diagnostics['hard_fake_selected'].item(), 5)
        self.assertEqual(diagnostics['hard_fake_effective'].item(), 1)

    def test_fake_reweighting_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, 'mode must be one of'):
            fake_reweighting_loss(
                torch.zeros(4),
                torch.ones(4),
                mode='unknown',
            )

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

    def test_adaptive_equal_statistics_produce_equal_detached_shares(self):
        controller = AdaptiveHardLossController(temperature=0.5)
        statistic = torch.tensor(2.0, requires_grad=True)

        fake_share, real_share, _ = controller.route(
            statistic,
            statistic,
            fake_selected=torch.tensor(2.0),
            real_selected=torch.tensor(2.0),
            step=1,
        )

        self.assertAlmostEqual(fake_share.item(), 0.5)
        self.assertAlmostEqual(real_share.item(), 0.5)
        self.assertFalse(fake_share.requires_grad)
        self.assertFalse(real_share.requires_grad)

    def test_adaptive_larger_statistic_receives_larger_share(self):
        controller = AdaptiveHardLossController(temperature=1.0)

        fake_share, real_share, diagnostics = controller.route(
            torch.tensor(2.0, requires_grad=True),
            torch.tensor(1.0, requires_grad=True),
            fake_selected=torch.tensor(1.0),
            real_selected=torch.tensor(1.0),
            step=1,
        )

        self.assertGreater(fake_share.item(), real_share.item())
        self.assertFalse(diagnostics['adaptive_hard_fake_stat'].requires_grad)
        self.assertAlmostEqual(
            (fake_share + real_share).item(), 1.0, places=6)
        self.assertTrue(torch.isfinite(fake_share))
        self.assertTrue(torch.isfinite(real_share))
        self.assertGreaterEqual(fake_share.item(), 0.0)
        self.assertGreaterEqual(real_share.item(), 0.0)

    def test_adaptive_tiny_positive_temperature_stays_finite(self):
        controller = AdaptiveHardLossController(temperature=1.0e-45)

        fake_share, real_share, _ = controller.route(
            torch.tensor(2.0),
            torch.tensor(1.0),
            fake_selected=torch.tensor(1.0),
            real_selected=torch.tensor(1.0),
            step=1,
        )

        self.assertTrue(torch.isfinite(fake_share))
        self.assertTrue(torch.isfinite(real_share))
        self.assertAlmostEqual(
            (fake_share + real_share).item(), 1.0, places=6)
        self.assertGreater(fake_share.item(), real_share.item())

    def test_adaptive_statistics_use_ema_before_routing(self):
        controller = AdaptiveHardLossController(
            temperature=1.0,
            ema_decay=0.5,
        )
        selected = torch.tensor(1.0)
        controller.route(
            torch.tensor(2.0),
            torch.tensor(2.0),
            fake_selected=selected,
            real_selected=selected,
            step=1,
        )

        fake_share, real_share, diagnostics = controller.route(
            torch.tensor(4.0),
            torch.tensor(2.0),
            fake_selected=selected,
            real_selected=selected,
            step=2,
        )

        self.assertAlmostEqual(
            diagnostics['adaptive_hard_fake_stat'].item(), 3.0)
        self.assertAlmostEqual(
            diagnostics['adaptive_hard_real_stat'].item(), 2.0)
        self.assertGreater(fake_share.item(), real_share.item())

    def test_adaptive_warmup_and_missing_classes_are_deterministic(self):
        warmup = AdaptiveHardLossController(warmup_steps=2)
        fake_only = warmup.route(
            torch.tensor(4.0),
            torch.tensor(1.0),
            fake_selected=torch.tensor(3.0),
            real_selected=torch.tensor(0.0),
            step=1,
        )
        real_only = warmup.route(
            torch.tensor(1.0),
            torch.tensor(4.0),
            fake_selected=torch.tensor(0.0),
            real_selected=torch.tensor(3.0),
            step=1,
        )
        both = warmup.route(
            torch.tensor(4.0),
            torch.tensor(1.0),
            fake_selected=torch.tensor(3.0),
            real_selected=torch.tensor(3.0),
            step=1,
        )
        self.assertEqual(
            tuple(value.item() for value in fake_only[:2]), (1.0, 0.0))
        self.assertEqual(
            tuple(value.item() for value in real_only[:2]), (0.0, 1.0))
        self.assertEqual(
            tuple(value.item() for value in both[:2]), (0.5, 0.5))
        self.assertEqual(
            fake_only[2]['adaptive_hard_in_warmup'].item(), 0.0)
        self.assertEqual(
            real_only[2]['adaptive_hard_in_warmup'].item(), 0.0)
        self.assertEqual(
            both[2]['adaptive_hard_in_warmup'].item(), 1.0)

        controller = AdaptiveHardLossController()
        fake_only = controller.route(
            torch.tensor(2.0),
            torch.tensor(float('nan')),
            fake_selected=torch.tensor(1.0),
            real_selected=torch.tensor(0.0),
            step=1,
        )[:2]
        neither = controller.route(
            torch.tensor(float('nan')),
            torch.tensor(float('inf')),
            fake_selected=torch.tensor(0.0),
            real_selected=torch.tensor(0.0),
            step=2,
        )[:2]
        self.assertEqual(tuple(value.item() for value in fake_only), (1.0, 0.0))
        self.assertEqual(tuple(value.item() for value in neither), (0.5, 0.5))
        self.assertTrue(all(torch.isfinite(value) for value in (*fake_only, *neither)))

    def test_selected_bce_means_are_detached_from_budget_losses(self):
        logits = torch.tensor(
            [-0.5, -1.0, 0.5, 1.0], requires_grad=True)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])

        fake_loss, fake_diagnostics = hard_fake_reweighting_loss(
            logits, labels, fraction=0.5)
        real_loss, real_diagnostics = hard_real_reweighting_loss(
            logits, labels, fraction=0.5)

        self.assertTrue(fake_loss.requires_grad)
        self.assertTrue(real_loss.requires_grad)
        self.assertFalse(fake_diagnostics['hard_fake_bce_mean'].requires_grad)
        self.assertFalse(real_diagnostics['hard_real_bce_mean'].requires_grad)
        self.assertGreater(fake_diagnostics['hard_fake_bce_mean'].item(), 0.0)
        self.assertGreater(real_diagnostics['hard_real_bce_mean'].item(), 0.0)


if __name__ == '__main__':
    unittest.main()
