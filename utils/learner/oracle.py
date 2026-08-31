"""
Supervised Oracle for SAR: identical parameter set, optimizer (SAM) and
schedule as SAR; the entropy objective is replaced by cross-entropy on the
target labels.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from .learner import register

__all__ = ['oracle']


@register('oracle')
def oracle(model, optimizer, steps=1, episodic=False):
    return Oracle(model=model, optimizer=optimizer, steps=steps, episodic=episodic)


class Oracle(nn.Module):
    """Adapts BN/LN affine parameters with SAM on the supervised cross-entropy.

    Serves as a diagnostic upper bound, not a practical TTA method.
    """

    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "Oracle requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # Save state for episodic resetting
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, y=None):
        """Forward with optional supervised adaptation.

        Args:
            x: input tensor [B, C, H, W]
            y: ground-truth labels [B]. If None, just forward (no adaptation).
        """
        if self.episodic:
            self.reset()

        if y is not None:
            for _ in range(self.steps):
                outputs = forward_and_adapt_oracle(x, y, self.model, self.optimizer)
        else:
            with torch.no_grad():
                outputs = self.model(x)

        return outputs

    def predict(self, feats):
        return self.model(feats)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)


@torch.enable_grad()
def forward_and_adapt_oracle(x, y, model, optimizer):
    """One SAM step on the supervised cross-entropy; both the ascent and the
    descent step use the labelled loss."""
    optimizer.zero_grad()
    outputs = model(x)
    loss = F.cross_entropy(outputs, y)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    outputs_second = model(x)
    loss_second = F.cross_entropy(outputs_second, y)
    loss_second.backward()
    optimizer.second_step(zero_grad=True)

    return outputs


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model, args=None):
    """Configure model for Oracle adaptation.

    Identical to SAR's configure_model: train mode, freeze everything
    except BN/LN affine params, force batch statistics.
    """
    model.train()
    model.requires_grad_(False)
    # BN layers: enable grad + force batch stats
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ViT-LN and ResNet-GN
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model
