import unittest

import torch

from utils.degradation_consistency import (
    degradation_consistency_loss,
    resize_degraded_view,
)


class DegradationConsistencyTests(unittest.TestCase):
    def test_resize_degradation_preserves_batch_shape(self):
        images = torch.arange(2 * 3 * 8 * 10, dtype=torch.float32).reshape(
            2, 3, 8, 10)

        degraded = resize_degraded_view(images, scale=0.75)

        self.assertEqual(degraded.shape, images.shape)
        self.assertFalse(torch.equal(degraded, images))

    def test_consistency_loss_detaches_teacher(self):
        student = torch.tensor([-0.5, 0.5], requires_grad=True)
        teacher = torch.tensor([-0.25, 0.25], requires_grad=True)

        loss = degradation_consistency_loss(student, teacher)
        loss.backward()

        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_invalid_scale_is_rejected(self):
        images = torch.zeros(1, 3, 8, 8)
        for scale in (0.0, 1.1):
            with self.assertRaisesRegex(ValueError, 'scale'):
                resize_degraded_view(images, scale=scale)


if __name__ == '__main__':
    unittest.main()
