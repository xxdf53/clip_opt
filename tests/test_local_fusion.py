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


if __name__ == '__main__':
    unittest.main()
