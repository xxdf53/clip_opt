import unittest

import torch

from utils.checkpoint_loading import (
    extract_training_state_dict,
    parse_gate_override,
    resolve_local_fusion,
    select_baseline_initialization_state,
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

    def test_detects_adaptive_residual_head(self):
        fusion = resolve_local_fusion(
            {'model.fc.weight': 1, 'local_classifier.weight': 2,
             'local_gate_network.weight': 3},
            use_local_features=True,
        )

        self.assertEqual(fusion, 'adaptive_residual')

    def test_rejects_explicit_fusion_mismatch(self):
        with self.assertRaisesRegex(ValueError, "checkpoint uses local_fusion='concat'"):
            resolve_local_fusion(
                {'model.fc.0.weight': 1},
                requested='residual_gate',
                use_local_features=True,
            )

    def test_rejects_local_fusion_without_local_features(self):
        with self.assertRaisesRegex(ValueError, '--use_local_features'):
            resolve_local_fusion(
                {'model.fc.weight': 1},
                requested='adaptive_residual',
                use_local_features=False,
            )

    def test_rejects_checkpoint_with_both_local_heads(self):
        with self.assertRaisesRegex(ValueError, 'multiple incompatible'):
            resolve_local_fusion(
                {'model.fc.0.weight': 1, 'local_gate_logit': 2},
                use_local_features=True,
            )

    def test_parses_learned_and_fixed_gate_overrides(self):
        self.assertIsNone(parse_gate_override('learned'))
        self.assertEqual(parse_gate_override('0'), 0.0)
        self.assertEqual(parse_gate_override('0.25'), 0.25)
        with self.assertRaisesRegex(ValueError, r'\[0, 1\]'):
            parse_gate_override('1.1')

    def test_selects_only_compatible_baseline_weights(self):
        source = {
            'model.fc.weight': torch.ones(1, 2),
            'model.fc.bias': torch.ones(1),
            'vision.lora_A.weight': torch.ones(2, 2),
            'ignored.weight': torch.ones(3),
        }
        target = {
            'model.fc.weight': torch.zeros(1, 2),
            'model.fc.bias': torch.zeros(1),
            'vision.lora_A.weight': torch.zeros(2, 2),
        }

        selected = select_baseline_initialization_state(source, target)

        self.assertEqual(set(selected), set(target))

    def test_rejects_local_checkpoint_as_baseline_initialization(self):
        with self.assertRaisesRegex(ValueError, 'non-local baseline'):
            select_baseline_initialization_state(
                {'local_projector.0.weight': torch.ones(1)}, {})


if __name__ == '__main__':
    unittest.main()
