import unittest

import numpy as np
import torch

from utils.bias_calibration import (
    balanced_error_rate,
    find_balanced_error_threshold,
    fold_threshold_into_linear_bias,
)


class BiasCalibrationTests(unittest.TestCase):
    def test_finds_exact_balanced_error_partition(self):
        logits = np.array([-2.0, -1.0, -0.5, 1.0])
        labels = np.array([0, 0, 1, 1])

        threshold = find_balanced_error_threshold(logits, labels)

        self.assertAlmostEqual(threshold, -0.75)
        self.assertEqual(
            balanced_error_rate(logits, labels, threshold), 0.0)
        self.assertEqual(
            balanced_error_rate(logits, labels, threshold=0.0), 0.25)

    def test_balanced_objective_is_not_dominated_by_class_count(self):
        logits = np.array([-3.0, -2.0, -1.0, -0.5, 0.5])
        labels = np.array([0, 0, 0, 1, 1])

        threshold = find_balanced_error_threshold(logits, labels)

        self.assertAlmostEqual(threshold, -0.75)
        self.assertEqual(
            balanced_error_rate(logits, labels, threshold), 0.0)

    def test_folds_threshold_into_existing_linear_bias(self):
        linear = torch.nn.Linear(2, 1)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor([[1.0, -1.0]]))
            linear.bias.fill_(0.25)
        inputs = torch.tensor([[2.0, 0.5]])
        original = linear(inputs).detach()

        fold_threshold_into_linear_bias(linear, threshold=-0.75)

        self.assertTrue(torch.allclose(linear(inputs), original + 0.75))

    def test_rejects_missing_classes_and_nonfinite_logits(self):
        with self.assertRaisesRegex(ValueError, 'both real and fake'):
            find_balanced_error_threshold([0.0, 1.0], [1, 1])
        with self.assertRaisesRegex(ValueError, 'finite'):
            find_balanced_error_threshold([0.0, np.nan], [0, 1])


if __name__ == '__main__':
    unittest.main()
