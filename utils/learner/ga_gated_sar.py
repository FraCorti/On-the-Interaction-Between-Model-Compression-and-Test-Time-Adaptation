"""
Gradient-alignment-gated SAR.

Before SAR's sharpness-aware update, the cosine alignment between the gradient
of SAR's entropy loss and the gradient of a cross-entropy against the model's own
argmax pseudo-labels (confidence-filtered subset) is computed, i.e. the per-batch
form of the paper's gradient-alignment metric. If the alignment is below tau_ga
the update is skipped for that batch. No target labels are used.
"""

from copy import deepcopy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .learner import register
from .sar import SAR, softmax_entropy, collect_params

__all__ = ['ga_gated_sar']


@register('ga_gated_sar')
def ga_gated_sar(model, optimizer, steps=1, episodic=False,
                 margin_e0=0.4 * math.log(1000),
                 reset_constant_em=0.2,
                 tau_ga=0.0):
    return GAGatedSAR(model=model, optimizer=optimizer, steps=steps,
                      episodic=episodic, margin_e0=margin_e0,
                      reset_constant_em=reset_constant_em,
                      tau_ga=tau_ga)


class GAGatedSAR(SAR):
    """SAR + per-batch gradient-alignment gate (Sec. 4.2 of main paper)."""

    def __init__(self, *args, tau_ga=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau_ga = tau_ga
        # Cache the SAR-adapted parameter list (BN/LN affine) once.
        adaptation_params, _ = collect_params(self.model)
        self._adaptation_params = adaptation_params
        # Counters for diagnostic reporting.
        self.batches_total = 0
        self.batches_gated_out = 0

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, ema, reset_flag, step_taken = \
                forward_and_adapt_ga_gated_sar(
                    x, self.model, self.optimizer,
                    self.margin_e0, self.reset_constant_em, self.ema,
                    self.tau_ga, self._adaptation_params,
                )
            if reset_flag:
                self.reset()
            self.ema = ema
            self.batches_total += 1
            if not step_taken:
                self.batches_gated_out += 1
        return outputs

    def gate_stats(self):
        """Fraction of batches in which the GA gate suppressed the update."""
        if self.batches_total == 0:
            return 0.0
        return self.batches_gated_out / self.batches_total


@torch.enable_grad()
def forward_and_adapt_ga_gated_sar(x, model, optimizer, margin, reset_constant,
                                    ema, tau_ga, adaptation_params):
    """SAR forward pass + GA gate + SAR sharpness-aware update."""
    # Forward and confidence filter
    optimizer.zero_grad()
    outputs = model(x)
    entropys_all = softmax_entropy(outputs)
    filter_ids = torch.where(entropys_all < margin)
    if filter_ids[0].numel() == 0:
        # Nothing reliable enough -- skip and report.
        return outputs, ema, False, False

    logits_sel = outputs[filter_ids]
    entropys_sel = entropys_all[filter_ids]

    # Entropy gradient (snapshot)
    optimizer.zero_grad()
    loss_ent = entropys_sel.mean(0)
    loss_ent.backward(retain_graph=True)
    saved_ent_grads = [
        p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
        for p in adaptation_params
    ]
    grad_ent_flat = torch.cat([g.flatten() for g in saved_ent_grads])

    # Pseudo-label cross-entropy gradient
    optimizer.zero_grad()
    with torch.no_grad():
        pseudo_labels = logits_sel.argmax(dim=1)
    loss_ce = F.cross_entropy(logits_sel, pseudo_labels)
    loss_ce.backward()
    grad_ce_flat = torch.cat([
        (p.grad.detach().clone() if p.grad is not None
         else torch.zeros_like(p)).flatten()
        for p in adaptation_params
    ])

    # Gradient-alignment gate
    cos = torch.dot(grad_ent_flat, grad_ce_flat) / (
        grad_ent_flat.norm() * grad_ce_flat.norm() + 1e-12
    )
    if cos.item() < tau_ga:
        # Active divergence: drop this batch.
        optimizer.zero_grad()
        return outputs, ema, False, False

    # Restore entropy grads and run SAR's two-step update
    for p, g in zip(adaptation_params, saved_ent_grads):
        p.grad = g.clone()
    optimizer.first_step(zero_grad=True)
    entropys2 = softmax_entropy(model(x))[filter_ids]
    filter_ids_2 = torch.where(entropys2 < margin)
    if filter_ids_2[0].numel() == 0:
        return outputs, ema, False, True
    loss_second = entropys2[filter_ids_2].mean(0)
    if not np.isnan(loss_second.item()):
        ema_val = loss_second.item()
        ema = ema_val if ema is None else 0.9 * ema + 0.1 * ema_val
    loss_second.backward()
    optimizer.second_step(zero_grad=True)

    reset_flag = ema is not None and ema < reset_constant
    return outputs, ema, reset_flag, True
