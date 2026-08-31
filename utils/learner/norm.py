"""
Adapted from https://github.com/DequanWang/tent/blob/master/norm.py

BP-free test-time adaptation via Batch Normalization recalibration.
This method adapts by using batch-wise statistics during test time,
without any backpropagation or gradient computation.
"""

from copy import deepcopy

import torch.nn as nn
from .learner import register

__all__ = ['norm']


@register('norm')
def norm(model, optimizer=None, eps=1e-5, momentum=0.1, reset_stats=False, no_stats=False, **kwargs):
    """
    Factory function for NORM adaptation (BP-free).
    
    Args:
        model: Model to adapt (must contain BatchNorm layers)
        optimizer: Ignored (NORM is BP-free, no optimizer needed)
        eps: BatchNorm epsilon for numerical stability
        momentum: BatchNorm momentum for running stats update
        reset_stats: If True, reset running statistics before adaptation
        no_stats: If True, disable running stats entirely (batch-only)
    
    Returns:
        Norm wrapper around the model
    """
    return Norm(model, eps=eps, momentum=momentum, reset_stats=reset_stats, no_stats=no_stats)


class Norm(nn.Module):
    """Norm adapts a model by estimating feature statistics during testing.

    Once equipped with Norm, the model normalizes its features during testing
    with batch-wise statistics, just like batch norm does during training.
    """

    def __init__(self, model, eps=1e-5, momentum=0.1,
                 reset_stats=False, no_stats=False):
        super().__init__()
        self.model = model
        self.model = configure_model(model, eps, momentum, reset_stats,
                                     no_stats)
        self.model_state = deepcopy(self.model.state_dict())

    def forward(self, x):
        return self.model(x)

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)


def collect_stats(model):
    """Collect the normalization stats from batch norms.

    Walk the model's modules and collect all batch normalization stats.
    Return the stats and their names.
    """
    stats = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            state = m.state_dict()
            if m.affine:
                del state['weight'], state['bias']
            for ns, s in state.items():
                stats.append(s)
                names.append(f"{nm}.{ns}")
    return stats, names


def configure_model(model, eps, momentum, reset_stats, no_stats):
    """Configure model for adaptation by test-time normalization."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # use batch-wise statistics in forward
            m.train()
            # configure epsilon for stability, and momentum for updates
            m.eps = eps
            m.momentum = momentum
            if reset_stats:
                # reset state to estimate test stats without train stats
                m.reset_running_stats()
            if no_stats:
                # disable state entirely and use only batch stats
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model
