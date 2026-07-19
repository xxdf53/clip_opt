import unittest

from utils.checkpoint_loading import (
    extract_training_state_dict,
    resolve_local_fusion,
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
        with self.assertRaisesRegex(ValueError, "'model' must be a state_dict mapping"):
            extract_training_state_dict({'model': None})

    def test_detects_legacy_concat_local_head(self):
        fusion = resolve_local_fusion(
            {'model.fc.0.weight': 1, 'local_projector.0.weight': 2},
            use_local_features=True,
        )

        self.assertEqual(fusion, 'concat')

    def test_detects_residual_gate_local_head(self):
        fusion = resolve_local_fusion(
            {'model.fc.weight': 1, 'local_gate_logit': 2,
             'local_classifier.weight': 3},
            use_local_features=True,
        )

        self.assertEqual(fusion, 'residual_gate')

    def test_rejects_explicit_fusion_mismatch(self):
        with self.assertRaisesRegex(ValueError, "checkpoint uses local_fusion='concat'"):
            resolve_local_fusion(
                {'model.fc.0.weight': 1},
                requested='residual_gate',
                use_local_features=True,
            )

    def test_rejects_checkpoint_with_both_local_heads(self):
        with self.assertRaisesRegex(ValueError, 'both concat and residual-gate'):
            resolve_local_fusion(
                {'model.fc.0.weight': 1, 'local_gate_logit': 2},
                use_local_features=True,
            )


if __name__ == '__main__':
    unittest.main()
