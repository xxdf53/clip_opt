import unittest

import torch
import torch.nn as nn

from networks.trainer import CLIPModel_lora


class LocalResidualGateTests(unittest.TestCase):
    def build_minimal_model(self, gate):
        model = CLIPModel_lora.__new__(CLIPModel_lora)
        nn.Module.__init__(model)
        model.use_local_features = True
        model.local_fusion = 'residual_gate'
        model.model = nn.Module()
        model.model.fc = nn.Linear(2, 1, bias=False)
        model.local_classifier = nn.Linear(2, 1, bias=False)
        model.local_gate_logit = nn.Parameter(torch.logit(torch.tensor(gate)))
        with torch.no_grad():
            model.model.fc.weight.copy_(torch.tensor([[1.0, 0.0]]))
            model.local_classifier.weight.copy_(torch.tensor([[0.0, 1.0]]))
        return model

    def test_residual_gate_preserves_global_logit_plus_bounded_correction(self):
        model = self.build_minimal_model(gate=0.25)

        _, logits = model.classify(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
        )

        self.assertTrue(torch.allclose(logits, torch.tensor([[1.25]])))
        self.assertAlmostEqual(model.local_gate_value().item(), 0.25, places=6)

    def test_gate_and_local_branch_receive_gradients(self):
        model = self.build_minimal_model(gate=0.01)

        _, logits = model.classify(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
        )
        logits.sum().backward()

        self.assertIsNotNone(model.local_gate_logit.grad)
        self.assertNotEqual(model.local_gate_logit.grad.item(), 0.0)
        self.assertIsNotNone(model.local_classifier.weight.grad)
        self.assertNotEqual(model.local_classifier.weight.grad.abs().sum().item(), 0.0)


class AdaptiveResidualGateTests(unittest.TestCase):
    def build_minimal_model(self, gate=0.01):
        model = CLIPModel_lora.__new__(CLIPModel_lora)
        nn.Module.__init__(model)
        model.use_local_features = True
        model.local_fusion = 'adaptive_residual'
        model.model = nn.Module()
        model.model.fc = nn.Linear(2, 1, bias=False)
        model.local_classifier = nn.Linear(2, 1, bias=False)
        model.local_gate_network = nn.Linear(6, 1)
        with torch.no_grad():
            model.model.fc.weight.copy_(torch.tensor([[1.0, 0.0]]))
            model.local_classifier.weight.copy_(torch.tensor([[0.0, 1.0]]))
            model.local_gate_network.weight.zero_()
            model.local_gate_network.bias.fill_(
                torch.logit(torch.tensor(gate)))
        return model

    def test_adaptive_gate_starts_near_configured_probability(self):
        model = self.build_minimal_model(gate=0.01)

        _, outputs = model.classification_outputs(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        )

        self.assertTrue(torch.allclose(
            outputs['gate'], torch.full((2, 1), 0.01), atol=1e-6))
        self.assertTrue(torch.allclose(
            outputs['final_logits'],
            outputs['global_logits'] + outputs['gate'] * outputs['local_logits']))

    def test_gate_override_changes_only_applied_gate(self):
        model = self.build_minimal_model(gate=0.25)
        image = torch.tensor([[1.0, 0.0]])
        local = torch.tensor([[0.0, 1.0]])

        _, outputs = model.classification_outputs(
            image, local, gate_override=0.0)

        self.assertAlmostEqual(outputs['learned_gate'].item(), 0.25, places=6)
        self.assertEqual(outputs['gate'].item(), 0.0)
        self.assertTrue(torch.equal(
            outputs['final_logits'], outputs['global_logits']))

    def test_gate_can_become_sample_adaptive(self):
        model = self.build_minimal_model(gate=0.5)
        with torch.no_grad():
            model.local_gate_network.weight[0, 0] = 2.0

        _, outputs = model.classification_outputs(
            torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
            torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        )

        self.assertGreater(
            outputs['gate'][0].item(), outputs['gate'][1].item())

    def test_freeze_global_parameters_keeps_local_modules_trainable(self):
        model = self.build_minimal_model()
        model.vision_tower_lora = nn.Linear(2, 2)

        model.freeze_global_parameters()

        self.assertFalse(any(
            parameter.requires_grad
            for parameter in model.vision_tower_lora.parameters()))
        self.assertFalse(any(
            parameter.requires_grad for parameter in model.model.fc.parameters()))
        self.assertTrue(any(
            parameter.requires_grad
            for parameter in model.local_classifier.parameters()))
        self.assertTrue(any(
            parameter.requires_grad
            for parameter in model.local_gate_network.parameters()))


if __name__ == '__main__':
    unittest.main()
