"""
NoAdapt: passthrough learner that forwards data through the model in eval
mode without any adaptation. The `.model` attribute is exposed so that
post-TTA diagnostic code works unchanged.
"""

import torch
import torch.nn as nn
from .learner import register

__all__ = ['no_adapt']


@register('no_adapt')
def no_adapt(model, optimizer=None, **kwargs):
    return NoAdapt(model=model)


class NoAdapt(nn.Module):
    """Eval-mode forward pass without parameter updates."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

    def predict(self, feats):
        return self.model(feats)

    def reset(self):
        pass  # no state to reset
