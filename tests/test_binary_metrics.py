import unittest

from utils.binary_metrics import compute_binary_metrics


class BinaryMetricsTests(unittest.TestCase):
    def test_computes_balanced_binary_metrics(self):
        metrics = compute_binary_metrics(
            labels=[0, 0, 1, 1],
            scores=[0.1, 0.4, 0.6, 0.9],
        )

        self.assertEqual(metrics['n'], 4)
        self.assertEqual(metrics['acc'], 100.0)
        self.assertEqual(metrics['real_acc'], 100.0)
        self.assertEqual(metrics['fake_acc'], 100.0)
        self.assertEqual(metrics['ap'], 100.0)

    def test_reports_class_specific_accuracy(self):
        metrics = compute_binary_metrics(
            labels=[0, 0, 1, 1],
            scores=[0.6, 0.4, 0.7, 0.8],
        )

        self.assertEqual(metrics['acc'], 75.0)
        self.assertEqual(metrics['real_acc'], 50.0)
        self.assertEqual(metrics['fake_acc'], 100.0)

    def test_rejects_missing_real_or_fake_samples(self):
        with self.assertRaisesRegex(ValueError, 'both real and fake samples'):
            compute_binary_metrics(labels=[1, 1], scores=[0.8, 0.9])


if __name__ == '__main__':
    unittest.main()
