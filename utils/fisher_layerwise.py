"""
Layer-wise Fisher Information Trace computation.

Computes Fisher trace per BN/LN layer for analyzing adaptation potential at each layer.
Only trainable layers (requires_grad=True) are included.
"""

import torch
import torch.nn as nn
from typing import Dict, Callable, Optional, List
from torch.utils.data import DataLoader, Dataset


def compute_layerwise_fisher(
    device: torch.device,
    model: nn.Module,
    dataset: Dataset,
    loss_fn: Callable,
    physical_batch_size: int = 1,
    max_samples: int = 500,
    args=None,
    preprocess: Callable = None
) -> Dict[str, float]:
    """
    Compute layer-wise Fisher Information Trace for each trainable BN/LN layer.
    
    Fisher Trace per layer = E[||grad_layer(L)||^2]
    
    Only layers with requires_grad=True are included (matching TTA training behavior).
    
    Args:
        device: GPU/CPU device
        model: Neural network model
        dataset: Dataset to compute Fisher over
        loss_fn: Loss function (e.g., cross-entropy or SPAConsistencyLoss)
        physical_batch_size: Batch size for gradient computation (1 for exact Fisher)
        max_samples: Maximum samples to use for estimation
        args: Additional arguments (for debug mode, etc.)
        preprocess: Optional preprocessing function for inputs
        
    Returns:
        Dict mapping layer name -> Fisher trace value
    """
    # Trainable BN/LN layers and their parameters
    layer_params: Dict[str, List[nn.Parameter]] = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            params = []
            if module.weight is not None and module.weight.requires_grad:
                params.append(module.weight)
            if module.bias is not None and module.bias.requires_grad:
                params.append(module.bias)
            if params:
                layer_params[name] = params
    
    if not layer_params:
        print("Warning: No trainable BN/LN layers found. Returning empty dict.")
        return {}
    
    # Initialize accumulators
    layer_fisher: Dict[str, float] = {name: 0.0 for name in layer_params}
    
    pin_memory = getattr(args, 'pin_memory', False) if args else False
    
    loader = DataLoader(
        dataset,
        batch_size=physical_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory
    )
    
    model.eval()
    count = 0
    
    # Accumulate gradients per layer
    for i, (X, y) in enumerate(loader):
        if i >= max_samples:
            break
        
        X = X.to(device).float()  # Ensure float32
        y = y.to(device)
        
        if preprocess is not None:
            X = preprocess(X)
        
        model.zero_grad()

        
        # Enforce Math SDPA for gradient computation
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            output = model(X)
            
            # Handle different loss function types
            if "SPAConsistencyLoss" in str(type(loss_fn)):
                loss = loss_fn(model, X, output)
            else:
                loss = loss_fn(output, y)
            
            # Compute gradients for all trainable params at once
            all_params = []
            param_to_layer = {}
            for layer_name, params in layer_params.items():
                for p in params:
                    all_params.append(p)
                    param_to_layer[id(p)] = layer_name
            
            grads = torch.autograd.grad(
                loss, all_params, create_graph=False, allow_unused=True
            )
            
            # Accumulate squared gradient norms per layer
            for p, g in zip(all_params, grads):
                layer_name = param_to_layer[id(p)]
                if g is not None:
                    layer_fisher[layer_name] += torch.sum(g ** 2).item()
        
        count += 1
        
        if args is not None and getattr(args, 'debug', False):
            break
    
    # Average over samples
    if count > 0:
        layer_fisher = {name: val / count for name, val in layer_fisher.items()}
    
    return layer_fisher


def get_sorted_layer_fisher_list(layer_fisher: Dict[str, float]) -> List[float]:
    """
    Convert layer Fisher dict to a sorted list (by layer name) for consistent plotting.
    
    This ensures layer order matches across different runs/models.
    """
    sorted_names = sorted(layer_fisher.keys())
    return [layer_fisher[name] for name in sorted_names]


def get_layer_fisher_names(layer_fisher: Dict[str, float]) -> List[str]:
    """
    Get sorted layer names for axis labels.
    """
    return sorted(layer_fisher.keys())
