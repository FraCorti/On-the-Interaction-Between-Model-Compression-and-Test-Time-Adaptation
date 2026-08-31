"""
Supervised Oracle for TENT: identical parameter set (BN/GN/LN affines) and
optimizer as TENT; the entropy objective is replaced by cross-entropy on the
target labels. Requires labels at test time, so it is a diagnostic upper
bound rather than a TTA method.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .learner import register
from . import tent as tent_mod

__all__ = ['oracle_tent']


@register('oracle_tent')
def oracle_tent(model, optimizer, steps=1, episodic=False):
    return OracleTent(model=model, optimizer=optimizer, steps=steps, episodic=episodic)


class OracleTent(nn.Module):
    """Supervised cross-entropy adaptation on TENT's parameter subspace.

    `forward(x, y)` performs `steps` update cycles; `forward(x)` without
    labels is plain inference.
    """

    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "OracleTent requires >= 1 step"
        self.episodic = episodic
        self.model_state, self.optimizer_state = \
            _copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, y=None):
        if self.episodic:
            self.reset()

        if y is None:
            with torch.no_grad():
                return self.model(x)

        for _ in range(self.steps):
            outputs = forward_and_adapt_oracle_tent(x, y, self.model, self.optimizer)
        return outputs

    def predict(self, feats):
        return self.model(feats)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("OracleTent: cannot reset without saved state")
        _load_model_and_optimizer(self.model, self.optimizer,
                                  self.model_state, self.optimizer_state)


@torch.enable_grad()
def forward_and_adapt_oracle_tent(x, y, model, optimizer):
    """One supervised cross-entropy step on TENT's parameter subspace."""
    optimizer.zero_grad()
    outputs = model(x)
    loss = F.cross_entropy(outputs, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def configure_model(model, args=None):
    """Delegates to tent.configure_model."""
    return tent_mod.configure_model(model=model, args=args)


def collect_params(model):
    """Same as tent.collect_params."""
    return tent_mod.collect_params(model)


def _copy_model_and_optimizer(model, optimizer):
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def _load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
