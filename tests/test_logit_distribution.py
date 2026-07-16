import math
import unittest

from utils.logit_distribution import build_shared_bin_edges, compute_logit_stats


class LogitStatisticsTests(unittest.TestCase):
    def test_computes_means_standard_deviations_and_separation(self):
        stats = compute_logit_stats([0.0, 2.0], [4.0, 6.0])

        self.assertEqual(stats['real_mean'], 1.0)
        self.assertEqual(stats['fake_mean'], 5.0)
        self.assertEqual(stats['real_std'], 1.0)
        self.assertEqual(stats['fake_std'], 1.0)
        self.assertEqual(stats['separation'], 4.0)

    def test_zero_variance_with_distinct_means_has_infinite_separation(self):
        stats = compute_logit_stats([-1.0, -1.0], [1.0, 1.0])
        self.assertTrue(math.isinf(stats['separation']))

    def test_zero_variance_with_equal_means_has_zero_separation(self):
        stats = compute_logit_stats([1.0], [1.0])
        self.assertEqual(stats['separation'], 0.0)

    def test_rejects_an_empty_class(self):
        with self.assertRaisesRegex(ValueError, 'both real and fake logits'):
            compute_logit_stats([], [1.0])


class SharedBinTests(unittest.TestCase):
    def test_uses_the_full_range_of_every_distribution(self):
        edges = build_shared_bin_edges(
            [[-3.0, -1.0], [1.0, 2.0], [-2.0], [5.0]], bins=4)

        self.assertEqual(len(edges), 5)
        self.assertEqual(edges[0], -3.0)
        self.assertEqual(edges[-1], 5.0)

    def test_expands_a_zero_width_range(self):
        edges = build_shared_bin_edges([[2.0], [2.0]], bins=2)
        self.assertLess(edges[0], 2.0)
        self.assertGreater(edges[-1], 2.0)

    def test_rejects_non_positive_bin_count(self):
        with self.assertRaisesRegex(ValueError, 'bins must be positive'):
            build_shared_bin_edges([[1.0]], bins=0)


if __name__ == '__main__':
    unittest.main()
