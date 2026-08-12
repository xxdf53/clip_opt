import unittest

import torch

from utils.checkpoint_loading import (
    extract_training_state_dict,
    infer_rrsd_max_correction,
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

    def test_infers_baseline_and_rrsd_architectures(self):
        self.assertEqual(infer_rrsd_max_correction({'model.fc.weight': 1}), 0.0)
        state_dict = {
            'rrsd.head.0.weight': torch.zeros(8, 16),
            'rrsd.head.0.bias': torch.zeros(8),
            'rrsd.head.2.weight': torch.zeros(1, 8),
            'rrsd.head.2.bias': torch.zeros(1),
            'rrsd.gate': torch.tensor(0.0),
            'rrsd.real_prototype': torch.zeros(16),
            'rrsd.real_count': torch.tensor(10.0),
            'rrsd.max_delta': torch.tensor(0.5),
        }

        self.assertEqual(infer_rrsd_max_correction(state_dict), 0.5)

    def test_rejects_partial_rrsd_state(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            infer_rrsd_max_correction({
                'rrsd.max_delta': torch.tensor(0.5),
            })


if __name__ == '__main__':
    unittest.main()
