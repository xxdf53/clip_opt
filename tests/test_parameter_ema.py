import unittest

import torch
import torch.nn as nn

from utils.parameter_ema import TrainableParameterEMA


class TrainableParameterEMATests(unittest.TestCase):
    def test_updates_and_temporarily_applies_average(self):
        model = nn.Linear(1, 1)
        ema = TrainableParameterEMA(decay=0.5)

        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(1.0)
        ema.update(model)

        with torch.no_grad():
            model.weight.fill_(3.0)
            model.bias.fill_(3.0)
        ema.update(model)

        with ema.average_parameters(model):
            self.assertEqual(model.weight.item(), 2.0)
            self.assertEqual(model.bias.item(), 2.0)

        self.assertEqual(model.weight.item(), 3.0)
        self.assertEqual(model.bias.item(), 3.0)

    def test_tracks_only_trainable_parameters(self):
        model = nn.Linear(1, 1)
        model.bias.requires_grad_(False)
        ema = TrainableParameterEMA(decay=0.99)

        ema.update(model)

        self.assertEqual(set(ema.shadow), {'weight'})

    def test_rejects_invalid_decay(self):
        for decay in (0.0, 1.0, -0.1):
            with self.assertRaises(ValueError):
                TrainableParameterEMA(decay)


if __name__ == '__main__':
    unittest.main()
