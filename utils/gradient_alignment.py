"""
Gradient alignment: cosine similarity between the unsupervised TTA gradient and
the supervised Oracle gradient, restricted to the normalization parameters.
Gradients are accumulated over the full evaluation set, g = (1/N) sum_i grad(loss_i),
and a single cosine similarity is computed on the accumulated vectors. NaN is
returned when the metric is undefined (zero-norm gradient).
"""

import torch
import torch.nn as nn
from typing import Callable, Optional
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math


def compute_gradient_alignment(
    device: torch.device,
    model: nn.Module,
    dataset: Dataset,
    tta_loss_fn: Callable,
    oracle_loss_fn: Callable = F.cross_entropy,
    physical_batch_size: int = 64,
    max_samples: int = None,
    args=None,
    preprocess: Callable = None,
    match_tta_bn: bool = False,
) -> dict:
    """
    Compute the cosine similarity between accumulated TTA and Oracle gradients:

        cos(g_TTA, g_Oracle)

    where g_TTA = (1/N) sum_i grad_theta L_TTA(x_i)
      and g_Oracle = (1/N) sum_i grad_theta L_Oracle(x_i, y_i)

    accumulated over N samples from the corruption dataset. Accumulating before
    the cosine avoids per-sample artifacts from zero gradients (e.g. SAR's
    reliable-sample filter).

    Args:
        device: Target GPU/CPU device
        model: Neural network model
        dataset: Corrupted dataset (must yield image and target label)
        tta_loss_fn: The unsupervised TTA loss function. If it sets
                     `requires_model_x_output = True` (e.g. SPAConsistencyLoss),
                     it is called as `tta_loss_fn(model, X, clean_output)`;
                     otherwise as `tta_loss_fn(output, X)`.
        oracle_loss_fn: The supervised Oracle loss of the method paired with the
                     TTA method (cross-entropy for Oracle-SAR, OracleSPALoss for
                     Oracle-SPA). If it sets `requires_model_x_y = True`, it is
                     called as `oracle_loss_fn(model, X, y)`; otherwise as
                     `oracle_loss_fn(output_oracle, y)`.
        physical_batch_size: Batch size for forward passes. Default 64 matches the
                            TTA adaptation batch size so BN batch statistics match.
        max_samples: Number of samples to use. None (default) = full dataset.
        args: Extra hyperparams
        preprocess: Data preprocessing logic
        match_tta_bn: When True, BN layers use batch statistics (as under
                      SAR/TENT's track_running_stats=False) instead of running
                      statistics. LayerNorm is unaffected.

    Returns:
        dict with keys:
            'cosine_sim': float in [-1, 1], or NaN if undefined (zero-norm gradient).
            'norm_tta': float, L2 norm of the accumulated TTA gradient.
            'norm_oracle': float, L2 norm of the accumulated Oracle gradient.
    """
    # Trainable parameters (BN/LN affine)
    trainable_params = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if module.weight is not None and module.weight.requires_grad:
                trainable_params.append(module.weight)
            if module.bias is not None and module.bias.requires_grad:
                trainable_params.append(module.bias)
                
    _nan_result = {'cosine_sim': float('nan'), 'norm_tta': float('nan'), 'norm_oracle': float('nan')}

    if not trainable_params:
        print("Warning: No trainable params found for gradient alignment.")
        return _nan_result

    # persistent_workers stays False: workers kept alive across the two
    # accumulation passes exhaust file descriptors.
    pin_memory = True if (device.type == 'cuda') else False

    # Full dataset by default for population-level gradient.
    n_total = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=physical_batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=pin_memory,
        drop_last=match_tta_bn,  # drop incomplete last batch when using batch stats
    )

    if match_tta_bn:
        # Match the SAR/TENT BN configuration: batch statistics instead of running statistics.
        model.eval()
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.train()
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
    else:
        model.eval()
    
    # Accumulate the TTA gradient g = (1/N) sum_i grad_theta L(x_i; theta)
    # via loss.backward() with per-batch scaling.
    
    model.zero_grad()
    n_seen = 0
    
    for X, y in loader:
        if n_seen >= n_total:
            break
            
        X = X.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        batch_size = X.shape[0]
        
        if preprocess is not None:
            X = preprocess(X)
        
        output = model(X)
        
        if getattr(tta_loss_fn, 'requires_model_x_output', False):
            tta_loss = tta_loss_fn(model, X, output)
        else:
            tta_loss = tta_loss_fn(output, X)
        
        # Scale by batch fraction for correct mean-gradient accumulation.
        scaled_loss = tta_loss * (batch_size / n_total)
        scaled_loss.backward()
            
        n_seen += batch_size
    
    grad_tta_flat = _extract_grad(trainable_params, device)
    if grad_tta_flat is None:
        return _nan_result
    
    # Accumulate the Oracle gradient
    
    model.zero_grad()
    n_seen = 0
    
    for X, y in loader:
        if n_seen >= n_total:
            break
            
        X = X.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        batch_size = X.shape[0]
        
        if preprocess is not None:
            X = preprocess(X)
        
        if getattr(oracle_loss_fn, 'requires_model_x_y', False):
            oracle_loss = oracle_loss_fn(model, X, y)
        else:
            output_oracle = model(X)
            oracle_loss = oracle_loss_fn(output_oracle, y)

        scaled_loss = oracle_loss * (batch_size / n_total)
        scaled_loss.backward()
            
        n_seen += batch_size
    
    grad_oracle_flat = _extract_grad(trainable_params, device)
    if grad_oracle_flat is None:
        return _nan_result

    # Cosine similarity of the accumulated gradients
    
    norm_tta = grad_tta_flat.norm().item()
    norm_oracle = grad_oracle_flat.norm().item()

    if norm_tta == 0 or norm_oracle == 0:
        return {'cosine_sim': float('nan'), 'norm_tta': norm_tta, 'norm_oracle': norm_oracle}

    sim = F.cosine_similarity(
        grad_tta_flat.unsqueeze(0),
        grad_oracle_flat.unsqueeze(0)
    ).item()

    cos_val = sim if not math.isnan(sim) else float('nan')
    return {'cosine_sim': cos_val, 'norm_tta': norm_tta, 'norm_oracle': norm_oracle}


def _extract_grad(params, device):
    """Extract and concatenate .grad from a list of parameters."""
    parts = []
    for p in params:
        if p.grad is not None:
            parts.append(p.grad.detach().reshape(-1).clone())
        else:
            parts.append(torch.zeros(p.numel(), device=device))
    if len(parts) == 0:
        return None
    return torch.cat(parts)