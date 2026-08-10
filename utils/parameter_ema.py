"""Exponential moving average for trainable model parameters."""

from contextlib import contextmanager

import torch


class TrainableParameterEMA:
    """Track and temporarily apply an EMA without changing checkpoints."""

    def __init__(self, decay):
        if not 0.0 < decay < 1.0:
            raise ValueError('EMA decay must be between 0 and 1')
        self.decay = float(decay)
        self.shadow = {}

    @staticmethod
    def _parameters(model):
        return {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        parameters = self._parameters(model)
        if not self.shadow:
            self.shadow = {
                name: parameter.detach().clone()
                for name, parameter in parameters.items()
            }
            return
        if parameters.keys() != self.shadow.keys():
            raise RuntimeError('trainable parameters changed after EMA started')
        for name, parameter in parameters.items():
            self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    @contextmanager
    def average_parameters(self, model):
        """Temporarily replace trainable parameters with their EMA values."""
        if not self.shadow:
            raise RuntimeError('EMA has not received an optimizer update')
        parameters = self._parameters(model)
        backup = {
            name: parameter.detach().clone()
            for name, parameter in parameters.items()
        }
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(backup[name])
