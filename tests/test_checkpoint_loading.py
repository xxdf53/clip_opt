import unittest

import torch

from utils.checkpoint_loading import (
    extract_training_state_dict,
    residual_vib_dim,
)


class TrainingCheckpointTests(unittest.TestCase):
    def test_extracts_model_and_total_steps(self):
        state_dict, total_steps = extract_training_state_dict({
            'model': {'module.layer.weight': 'weight'},
            'total_steps': 2251,
        })

        self.assertEqual(state_dict, {'layer.weight': 'weight'})
        self.assertEqual(total_steps, 2251)

    def test_removes_only_a_leading_dataparallel_prefix(self):
        state_dict, _ = extract_training_state_dict({
            'model': {
                'module.fc.weight': 1,
                'module_name.weight': 2,
                'encoder.module.weight': 3,
            },
        })

        self.assertEqual(state_dict, {
            'fc.weight': 1,
            'module_name.weight': 2,
            'encoder.module.weight': 3,
        })

    def test_rejects_payload_without_model_state(self):
        with self.assertRaisesRegex(ValueError, "missing a 'model' state_dict"):
            extract_training_state_dict({'total_steps': 10})

    def test_rejects_non_mapping_model_state(self):
        with self.assertRaisesRegex(
            ValueError,
            "'model' must be a state_dict mapping",
        ):
            extract_training_state_dict({'model': None})

    def test_detects_residual_vib_dimension(self):
        self.assertEqual(residual_vib_dim({
            'residual_vib.mu.weight': torch.empty(32, 768),
        }), 32)
        self.assertIsNone(residual_vib_dim({'model.fc.weight': 1}))


if __name__ == '__main__':
    unittest.main()
