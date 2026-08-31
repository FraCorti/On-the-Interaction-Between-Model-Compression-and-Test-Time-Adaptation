"""
Correctness-filtered SAR: SAR whose reliability filter is intersected with a
correctness filter, (entropy < E0) AND (argmax == y). The SAM update, second
entropy re-filter, EMA-based recovery and margins are inherited from sar.py.
Labels are used only to build the update mask, so this is a diagnostic and
not a deployable TTA method; the returned logits are the first-forward
outputs as in SAR. Batches where no sample passes the joint filter are
skipped to avoid a NaN loss.
"""

import math

import numpy as np
import torch

from .learner import register
from .sar import SAR, softmax_entropy

__all__ = ['sar_correct']


@register('sar_correct')
def sar_correct(model, optimizer, steps=1, episodic=False,
                margin_e0=0.4 * math.log(1000),
                reset_constant_em=0.2):
    return SARCorrect(model=model, optimizer=optimizer, steps=steps,
                      episodic=episodic, margin_e0=margin_e0,
                      reset_constant_em=reset_constant_em)


class SARCorrect(SAR):
    """SAR whose reliability filter additionally drops confident-wrong
    samples (requires ground-truth labels; diagnostic use only)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Diagnostic counters (fractions reported by mask_stats()).
        self.n_samples_total = 0
        self.n_reliable = 0
        self.n_reliable_correct = 0
        self.batches_skipped = 0
        self.batches_total = 0

    def forward(self, x, y):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, ema, reset_flag, stats = forward_and_adapt_sar_correct(
                x, y, self.model, self.optimizer, self.margin_e0,
                self.reset_constant_em, self.ema)
            if reset_flag:
                self.reset()
            self.ema = ema
            self.batches_total += 1
            self.n_samples_total += stats['n']
            self.n_reliable += stats['n_reliable']
            self.n_reliable_correct += stats['n_reliable_correct']
            if stats['skipped']:
                self.batches_skipped += 1
        return outputs

    def mask_stats(self):
        n = max(self.n_samples_total, 1)
        return {
            'frac_reliable': self.n_reliable / n,
            'frac_reliable_correct': self.n_reliable_correct / n,
            'frac_confident_wrong': (self.n_reliable
                                     - self.n_reliable_correct) / n,
            'frac_batches_skipped': (self.batches_skipped
                                     / max(self.batches_total, 1)),
        }


@torch.enable_grad()
def forward_and_adapt_sar_correct(x, y, model, optimizer, margin,
                                  reset_constant, ema):
    """SAR's forward_and_adapt with the joint reliable-and-correct filter."""
    optimizer.zero_grad()
    outputs = model(x)
    entropys_all = softmax_entropy(outputs)
    with torch.no_grad():
        preds = outputs.argmax(dim=1)
    reliable = entropys_all < margin
    joint = reliable & (preds == y)
    filter_ids_1 = torch.where(joint)
    stats = {'n': int(x.size(0)),
             'n_reliable': int(reliable.sum().item()),
             'n_reliable_correct': int(joint.sum().item()),
             'skipped': False}

    if filter_ids_1[0].numel() == 0:
        # No sample passes the joint filter: skip the update.
        optimizer.zero_grad()
        stats['skipped'] = True
        return outputs, ema, False, stats

    entropys = entropys_all[filter_ids_1]
    loss = entropys.mean(0)
    loss.backward()

    optimizer.first_step(zero_grad=True)  # \Theta + \hat{\epsilon}(\Theta), SAR Eqn. (4)
    entropys2 = softmax_entropy(model(x))
    entropys2 = entropys2[filter_ids_1]   # second forward, same joint subset
    filter_ids_2 = torch.where(entropys2 < margin)  # SAR's second entropy re-filter
    if filter_ids_2[0].numel() == 0:
        optimizer.zero_grad()
        stats['skipped'] = True
        return outputs, ema, False, stats
    loss_second = entropys2[filter_ids_2].mean(0)
    if not np.isnan(loss_second.item()):
        ema = _update_ema(ema, loss_second.item())

    loss_second.backward()
    optimizer.second_step(zero_grad=True)

    reset_flag = ema is not None and ema < reset_constant
    return outputs, ema, reset_flag, stats


def _update_ema(ema, new_data):
    if ema is None:
        return new_data
    return 0.9 * ema + (1 - 0.9) * new_data
