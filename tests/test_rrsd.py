import unittest

import torch

from networks.rrsd import (
    RealReferenceSpectralResidual,
    radial_log_power_features,
)


class RealReferenceSpectralResidualTests(unittest.TestCase):
    def test_radial_features_are_finite_and_fixed_width(self):
        images = torch.randn(3, 3, 15, 17)

        features = radial_log_power_features(images, bands=16)

        self.assertEqual(features.shape, (3, 16))
        self.assertTrue(torch.isfinite(features).all())

    def test_initial_residual_is_exactly_zero_and_bounded(self):
        module = RealReferenceSpectralResidual(max_delta=0.5)
        features = torch.randn(4, 16)
        module.update_real_prototype(features[:2].sum(dim=0), torch.tensor(2.0))

        initial, _ = module(features)
        self.assertTrue(torch.equal(initial, torch.zeros_like(initial)))

        with torch.no_grad():
            module.gate.fill_(10.0)
            module.head[-1].weight.fill_(10.0)
            module.head[-1].bias.fill_(10.0)
        correction, _ = module(features)
        self.assertTrue((correction.abs() <= 0.5).all())

    def test_global_real_statistics_match_the_full_real_batch(self):
        module = RealReferenceSpectralResidual(max_delta=0.5)
        features = torch.arange(96, dtype=torch.float32).reshape(6, 16)
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        sums = []
        counts = []
        for shard_features, shard_labels in zip(
            features.tensor_split(2),
            labels.tensor_split(2),
        ):
            real = shard_features[shard_labels == 0]
            sums.append(real.sum(dim=0))
            counts.append(torch.tensor(float(real.shape[0])))

        module.update_real_prototype(
            torch.stack(sums).sum(dim=0),
            torch.stack(counts).sum(),
        )

        expected = features[labels == 0].mean(dim=0)
        self.assertTrue(torch.equal(module.real_prototype, expected))
        self.assertEqual(module.real_count.item(), 3.0)

    def test_empty_real_batch_does_not_change_reference(self):
        module = RealReferenceSpectralResidual(max_delta=0.5)

        module.update_real_prototype(torch.ones(16), torch.tensor(0.0))

        self.assertEqual(module.real_count.item(), 0.0)
        self.assertTrue(torch.equal(
            module.real_prototype,
            torch.zeros_like(module.real_prototype),
        ))


if __name__ == '__main__':
    unittest.main()
