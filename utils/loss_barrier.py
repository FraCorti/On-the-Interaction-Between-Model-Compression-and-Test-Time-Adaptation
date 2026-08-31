"""
Loss barrier along the linear path between two models (Frankle et al., ICML 2020):
    B(θ₁, θ₂) = sup_{α∈[0,1]} L(αθ₁ + (1-α)θ₂) - mean(L(θ₁), L(θ₂)).
Structurally pruned models are inflated back to the dense parameter space before
interpolation.
"""

import copy
from collections import OrderedDict
from typing import Dict, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def restore_bn_running_stats(
    model: nn.Module,
    calibration_loader: DataLoader,
    device: torch.device = None,
    num_batches: int = 50
) -> nn.Module:
    """
    Restore BatchNorm running statistics after TTA adaptation.
    
    SAR/TENT adaptation sets track_running_stats=False and running_mean/var=None,
    which causes issues when the model is later used in eval() mode (as needed for
    loss barrier computation). This function recollects running statistics by
    passing calibration data through the model.
    
    Args:
        model: Model after TTA adaptation (with potentially missing BN running stats)
        calibration_loader: DataLoader for calibration data (clean or corruption)
        device: Target device
        num_batches: Number of batches to use for recollecting statistics
        
    Returns:
        Model with restored BatchNorm running statistics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.to(device)
    
    # First, re-enable running stats tracking and initialize buffers
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True
            # Initialize buffers if they are None
            if m.running_mean is None:
                m.running_mean = torch.zeros(m.num_features, device=device)
            if m.running_var is None:
                m.running_var = torch.ones(m.num_features, device=device)
            if m.num_batches_tracked is None:
                m.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=device)
            # Reset to fresh state
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = 0.1  # Default PyTorch momentum
    
    # Put model in train mode to accumulate running stats
    model.train()
    
    # Disable gradient computation and pass calibration data
    for batch_idx, (inputs, _) in enumerate(calibration_loader):
        if batch_idx >= num_batches:
            break
        inputs = inputs.to(device)
        _ = model(inputs)
    
    # Return to eval mode
    model.eval()
    
    return model


def inflate_state_dict(
    pruned_sd: Dict[str, torch.Tensor],
    reference_sd: Dict[str, torch.Tensor],
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    Inflate a structurally pruned state dict to match the reference shape by zero-padding.
    
    For each parameter in pruned_sd that has a smaller shape than in reference_sd,
    zero-pad it to match the reference shape. Parameters with matching shapes are copied as-is.
    
    Args:
        pruned_sd: State dict from structurally pruned model
        reference_sd: State dict from dense reference model
        device: Target device for inflated tensors
        
    Returns:
        Inflated state dict with same shapes as reference_sd
    """
    if device is None:
        device = next(iter(reference_sd.values())).device
        
    inflated_sd = OrderedDict()
    
    for key in reference_sd.keys():
        ref_param = reference_sd[key]
        
        if key not in pruned_sd:
            # Parameter doesn't exist in pruned model, use zeros
            inflated_sd[key] = torch.zeros_like(ref_param, device=device)
            continue
            
        pruned_param = pruned_sd[key]
        
        # Handle shape mismatches for each dimension
        if pruned_param.shape == ref_param.shape:
            inflated_sd[key] = pruned_param.clone().to(device)
        else:
            # Create zero tensor with reference shape
            inflated = torch.zeros_like(ref_param, device=device)
            
            # Build slices for the pruned dimensions
            slices = []
            for i, (p_dim, r_dim) in enumerate(zip(pruned_param.shape, ref_param.shape)):
                slices.append(slice(0, min(p_dim, r_dim)))
            
            # Build source slices (in case pruned is larger than reference somehow)
            src_slices = []
            for i, (p_dim, r_dim) in enumerate(zip(pruned_param.shape, ref_param.shape)):
                src_slices.append(slice(0, min(p_dim, r_dim)))
            
            # Copy pruned values into inflated tensor
            inflated[tuple(slices)] = pruned_param[tuple(src_slices)].to(device)
            inflated_sd[key] = inflated
    
    return inflated_sd


def inflate_state_dict_indexed(
    pruned_sd: Dict[str, torch.Tensor],
    reference_sd: Dict[str, torch.Tensor],
    keep_indices: Dict[str, torch.Tensor],
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    Inflate a structurally pruned state dict using stored channel indices.

    Unlike plain inflate_state_dict (which compacts surviving channels to
    positions 0…K-1), this function places each surviving channel at its
    **original position** in the dense model, preserving alignment for
    valid loss-barrier interpolation.

    The keep_indices dict maps layer names (e.g. 'conv1', 'layer1.0.conv2',
    'blocks.3.mlp.fc1') to a LongTensor of the original channel positions
    that survived pruning. For ResNet conv layers the indices correspond to
    output channels; for ViT MLP layers they correspond to hidden units
    (fc1 output / fc2 input).

    Pruned positions are zero-filled.

    Args:
        pruned_sd:   State dict from the structurally pruned model.
        reference_sd: State dict from the dense reference model (target shape).
        keep_indices: {layer_name: LongTensor[K]} of original positions kept.
        device:      Target device for inflated tensors.

    Returns:
        Inflated state dict with same shapes as reference_sd.
    """
    if device is None:
        device = next(iter(reference_sd.values())).device

    inflated_sd = OrderedDict()

    # Each key is resolved to the layer whose keep_indices governs its output
    # (dim 0) or input (dim 1) channels. BN layers share the channel dimension
    # with the conv they follow (bn1 with conv1, layer1.0.bn1 with layer1.0.conv1).
    bn_to_conv = {}
    for ki_name in keep_indices:
        # Corresponding BN name
        bn_name = ki_name.replace('conv', 'bn')
        bn_to_conv[bn_name] = ki_name
        # downsample.0 (conv) pairs with downsample.1 (bn)
        if 'downsample.0' in ki_name:
            bn_name = ki_name.replace('downsample.0', 'downsample.1')
            bn_to_conv[bn_name] = ki_name

    for key in reference_sd.keys():
        ref_param = reference_sd[key]

        if key not in pruned_sd:
            inflated_sd[key] = torch.zeros_like(ref_param, device=device)
            continue

        pruned_param = pruned_sd[key]

        if pruned_param.shape == ref_param.shape:
            inflated_sd[key] = pruned_param.clone().to(device)
            continue

        # Determine which keep_indices govern this key
        inflated = torch.zeros_like(ref_param, device=device)

        # Strip suffix to get the layer name
        layer_name = None
        for suffix in ['.weight', '.bias', '.running_mean', '.running_var']:
            if key.endswith(suffix):
                layer_name = key[:-len(suffix)]
                break

        if layer_name is None:
            # Unknown suffix: fall back to plain zero-pad
            slices = [slice(0, min(p, r)) for p, r in zip(pruned_param.shape, ref_param.shape)]
            inflated[tuple(slices)] = pruned_param[tuple(slices)].to(device)
            inflated_sd[key] = inflated
            continue

        # Find matching keep_indices: direct match or via the bn-to-conv mapping
        idx_key = layer_name if layer_name in keep_indices else bn_to_conv.get(layer_name)

        if idx_key is not None:
            idx = keep_indices[idx_key].to(device)

            if len(pruned_param.shape) == 1:
                # 1D: bias, running_mean, running_var, BN weight/bias
                inflated[idx] = pruned_param.to(device)

            elif len(pruned_param.shape) == 2:
                # 2D linear weight
                if pruned_param.shape[0] < ref_param.shape[0] and pruned_param.shape[1] == ref_param.shape[1]:
                    # Output dim compressed (fc1-style)
                    inflated[idx, :] = pruned_param.to(device)
                elif pruned_param.shape[1] < ref_param.shape[1] and pruned_param.shape[0] == ref_param.shape[0]:
                    # Input dim compressed (fc2-style)
                    inflated[:, idx] = pruned_param.to(device)
                elif pruned_param.shape[0] < ref_param.shape[0] and pruned_param.shape[1] < ref_param.shape[1]:
                    # Both dims compressed (FC after a pruned conv): scatter dim 0, compact-copy dim 1
                    inflated[idx, :pruned_param.shape[1]] = pruned_param.to(device)
                else:
                    slices = [slice(0, min(p, r)) for p, r in zip(pruned_param.shape, ref_param.shape)]
                    inflated[tuple(slices)] = pruned_param[tuple(slices)].to(device)

            elif len(pruned_param.shape) == 4:
                # 4D conv weight: [C_out, C_in, kH, kW]
                if pruned_param.shape[0] < ref_param.shape[0] and pruned_param.shape[1] == ref_param.shape[1]:
                    # Output channels pruned only
                    inflated[idx, :, :, :] = pruned_param.to(device)
                elif pruned_param.shape[0] == ref_param.shape[0] and pruned_param.shape[1] < ref_param.shape[1]:
                    # Input channels pruned only
                    inflated[:, idx, :, :] = pruned_param.to(device)
                elif pruned_param.shape[0] < ref_param.shape[0] and pruned_param.shape[1] < ref_param.shape[1]:
                    # Both dims pruned: scatter output dim, compact-copy input dim
                    tmp = torch.zeros(ref_param.shape[0], pruned_param.shape[1],
                                      *pruned_param.shape[2:], device=device)
                    tmp[idx, :, :, :] = pruned_param.to(device)
                    inflated[:, :pruned_param.shape[1], :, :] = tmp[:, :pruned_param.shape[1], :, :]
                else:
                    slices = [slice(0, min(p, r)) for p, r in zip(pruned_param.shape, ref_param.shape)]
                    inflated[tuple(slices)] = pruned_param[tuple(slices)].to(device)
            else:
                slices = [slice(0, min(p, r)) for p, r in zip(pruned_param.shape, ref_param.shape)]
                inflated[tuple(slices)] = pruned_param[tuple(slices)].to(device)
        else:
            # No keep_indices for this key: check whether a preceding layer's
            # keep_indices prune its input dimension.
            handled = False
            for ki_name, ki_idx in keep_indices.items():
                ki_idx_d = ki_idx.to(device)
                if len(pruned_param.shape) == 4 and pruned_param.shape[1] < ref_param.shape[1] and \
                   pruned_param.shape[0] == ref_param.shape[0] and ki_idx.shape[0] == pruned_param.shape[1]:
                    inflated[:, ki_idx_d, :, :] = pruned_param.to(device)
                    handled = True
                    break
                elif len(pruned_param.shape) == 2 and pruned_param.shape[1] < ref_param.shape[1] and \
                     pruned_param.shape[0] == ref_param.shape[0] and ki_idx.shape[0] == pruned_param.shape[1]:
                    inflated[:, ki_idx_d] = pruned_param.to(device)
                    handled = True
                    break

            if not handled:
                # Final fallback: plain zero-pad (compact to top-left)
                slices = [slice(0, min(p, r)) for p, r in zip(pruned_param.shape, ref_param.shape)]
                inflated[tuple(slices)] = pruned_param[tuple(slices)].to(device)

        inflated_sd[key] = inflated

    return inflated_sd


def inflate_state_dict_centroid(
    pruned_sd: Dict[str, torch.Tensor],
    reference_sd: Dict[str, torch.Tensor],
    cluster_labels: Dict[str, torch.Tensor],
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    Inflate a structurally pruned (folded) state dict by replicating centroids.
    
    For fold compression, each cluster of neurons is replaced by a single centroid.
    This function reverses that by replicating each centroid to fill its original
    cluster positions. This avoids the artificial barrier created by zero-padding,
    where interpolation blends real values with zeros.
    
    Args:
        pruned_sd: State dict from folded model (smaller dimensions)
        reference_sd: State dict from dense reference model (target dimensions)
        cluster_labels: Dict mapping parameter name (without .weight/.bias suffix)
                       to a LongTensor of shape [n_original] containing cluster
                       assignments for each original neuron/channel.
                       E.g. {'blocks.0.mlp.fc1': tensor([0, 0, 1, 1, 2, ...])}
        device: Target device for inflated tensors
        
    Returns:
        Inflated state dict with same shapes as reference_sd, centroids replicated.
    """
    if device is None:
        device = next(iter(reference_sd.values())).device
        
    inflated_sd = OrderedDict()
    
    for key in reference_sd.keys():
        ref_param = reference_sd[key]
        
        if key not in pruned_sd:
            inflated_sd[key] = torch.zeros_like(ref_param, device=device)
            continue
            
        pruned_param = pruned_sd[key]
        
        if pruned_param.shape == ref_param.shape:
            # Same shape: nothing to reconstruct
            inflated_sd[key] = pruned_param.clone().to(device)
            continue
        
        # Determine the base name (strip .weight / .bias / .running_mean etc.)
        # Try to find matching cluster labels
        base_key = None
        for suffix in ['.weight', '.bias', '.running_mean', '.running_var']:
            if key.endswith(suffix):
                candidate = key[:-len(suffix)]
                if cluster_labels is not None and candidate in cluster_labels:
                    base_key = candidate
                    break
        
        if base_key is not None:
            labels = cluster_labels[base_key]  # [n_original] with values in [0, n_clusters)
            n_original = labels.shape[0]
            n_clusters = pruned_param.shape[0]  # first dim is the compressed one

            # merge_channel_clustering stores cluster sums (count_j * centroid_j) for
            # Conv/Linear weights on axis 0, while _fold_bn_params stores means;
            # only ND weights are divided by the cluster counts.
            counts = torch.bincount(labels, minlength=n_clusters).float().clamp(min=1)

            # Replicate centroids: each position gets its cluster's centroid
            if len(pruned_param.shape) == 1:
                # 1D BN params already store means; no count correction
                inflated = pruned_param[labels].to(device)
            elif len(pruned_param.shape) == 2:
                # 2D weight: could be output dim (fc1.weight) or input dim (fc2.weight)
                if pruned_param.shape[0] < ref_param.shape[0] and pruned_param.shape[1] == ref_param.shape[1]:
                    # Output dimension compressed: divide sums by counts
                    centroids = pruned_param.float() / counts.to(pruned_param.device).view(-1, 1)
                    inflated = centroids[labels].to(device)
                elif pruned_param.shape[1] < ref_param.shape[1] and pruned_param.shape[0] == ref_param.shape[0]:
                    # Input dimension compressed (fc2-style); axis=1 merge already stores centroids
                    inflated = pruned_param[:, labels].to(device)
                else:
                    # Both dims differ or unexpected: fall back to zero-padding
                    inflated = torch.zeros_like(ref_param, device=device)
                    slices = [slice(0, min(p, r)) for p, r in zip(pruned_param.shape, ref_param.shape)]
                    inflated[tuple(slices)] = pruned_param[tuple(slices)].to(device)
            elif len(pruned_param.shape) == 4:
                # Conv2d weight: [C_out, C_in, H, W]
                # Both output (dim 0) and input (dim 1) may be compressed
                result = pruned_param.float().to(device)
                if result.shape[0] < ref_param.shape[0]:
                    # Output channels compressed: divide sums by counts
                    centroids = result / counts.to(device).view(-1, 1, 1, 1)
                    result = centroids[labels]
                if result.shape[1] < ref_param.shape[1]:
                    # Input channels compressed by a preceding layer (centroids, no count
                    # correction); replicate uniformly since its cluster labels are unknown.
                    n_in_pruned = result.shape[1]
                    n_in_ref = ref_param.shape[1]
                    repeats = torch.full((n_in_pruned,), n_in_ref // n_in_pruned, dtype=torch.long)
                    remainder = n_in_ref - repeats.sum().item()
                    if remainder > 0:
                        repeats[:remainder] += 1
                    result = result.repeat_interleave(repeats.to(result.device), dim=1)
                inflated = result
            else:
                # Fallback: zero-pad
                inflated = torch.zeros_like(ref_param, device=device)
                slices = [slice(0, min(p, r)) for p, r in zip(pruned_param.shape, ref_param.shape)]
                inflated[tuple(slices)] = pruned_param[tuple(slices)].to(device)
            
            inflated_sd[key] = inflated
        else:
            # No cluster labels: repeat each centroid ceil(n_original/n_clusters) times
            # and truncate to the reference dimension.
            inflated = pruned_param.clone().to(device)
            for dim_idx in range(len(ref_param.shape)):
                n_pruned = inflated.shape[dim_idx]
                n_ref = ref_param.shape[dim_idx]
                if n_pruned < n_ref:
                    # Compute how many times each element needs to repeat
                    repeats = torch.full((n_pruned,), n_ref // n_pruned, dtype=torch.long)
                    # Distribute remainder across first elements
                    remainder = n_ref - repeats.sum().item()
                    if remainder > 0:
                        repeats[:remainder] += 1
                    inflated = inflated.repeat_interleave(repeats.to(inflated.device), dim=dim_idx)
            inflated_sd[key] = inflated
    
    return inflated_sd


def interpolate_state_dicts(
    sd1: Dict[str, torch.Tensor],
    sd2: Dict[str, torch.Tensor],
    alpha: float
) -> Dict[str, torch.Tensor]:
    """
    Linearly interpolate between two state dicts.
    
    Result = α * sd1 + (1 - α) * sd2
    
    Args:
        sd1: First state dict (endpoint at α=1)
        sd2: Second state dict (endpoint at α=0)
        alpha: Interpolation coefficient in [0, 1]
        
    Returns:
        Interpolated state dict
    """
    interpolated = OrderedDict()
    
    for key in sd1.keys():
        if key in sd2:
            interpolated[key] = alpha * sd1[key] + (1 - alpha) * sd2[key]
        else:
            interpolated[key] = sd1[key]
            
    return interpolated


@torch.no_grad()
def _restore_bn_if_needed(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 20
) -> None:
    """
    Check if any BatchNorm layer has destroyed running stats (from SAR/TENT)
    and recollect them via a forward pass. Operates in-place.
    """
    needs_restoration = False
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.running_mean is None or m.running_var is None or not m.track_running_stats:
                needs_restoration = True
                break
    
    if not needs_restoration:
        return
    
    # Re-enable tracking and initialize fresh buffers
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.track_running_stats = True
            if m.running_mean is None:
                m.running_mean = torch.zeros(m.num_features, device=device)
            if m.running_var is None:
                m.running_var = torch.ones(m.num_features, device=device)
            if m.num_batches_tracked is None:
                m.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=device)
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = 0.1
    
    # Collect stats in train mode
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        _ = model(inputs)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    dataloader: DataLoader,
    loss_fn: Callable = None,
    device: torch.device = None,
    max_batches: int = None
) -> float:
    """
    Evaluate model loss with given state dict.
    
    Handles the case where BatchNorm running stats were destroyed by TTA
    (SAR/TENT set running_mean=None). In that case, recollects stats via
    a forward pass on the evaluation data before switching to eval mode.
    
    Args:
        model: Model architecture (will be loaded with state_dict)
        state_dict: Weights to evaluate
        dataloader: Data to evaluate on
        loss_fn: Loss function (default: cross entropy)
        device: Target device
        max_batches: Max batches to evaluate (for speed)
        
    Returns:
        Average loss value
    """
    if device is None:
        device = next(iter(state_dict.values())).device
    if loss_fn is None:
        loss_fn = F.cross_entropy
        
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    
    # Fix BN running stats if destroyed by SAR/TENT adaptation
    _restore_bn_if_needed(model, dataloader, device, num_batches=20)
    
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
            
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        if hasattr(outputs, 'logits'):
            outputs = outputs.logits
            
        loss = loss_fn(outputs, targets)
        
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    
    return total_loss / total_samples if total_samples > 0 else 0.0


def compute_loss_barrier(
    model: nn.Module,
    sd1: Dict[str, torch.Tensor],
    sd2: Dict[str, torch.Tensor],
    dataloader: DataLoader,
    loss_fn: Callable = None,
    n_alphas: int = 11,
    device: torch.device = None,
    max_batches: int = None
) -> Tuple[float, Dict]:
    """
    Compute the loss barrier between two model weight configurations.
    
    Loss Barrier B(θ₁, θ₂) = sup_{α∈[0,1]} L(αθ₁ + (1-α)θ₂) - mean(L(θ₁), L(θ₂))
    where mean(a, b) = (a + b) / 2, following Frankle et al. (ICML 2020).
    
    Args:
        model: Model architecture template
        sd1: State dict for θ₁ (anchor model)
        sd2: State dict for θ₂ (comparison model)
        dataloader: Data to compute loss on
        loss_fn: Loss function (default: cross entropy)
        n_alphas: Number of interpolation points (default: 11)
        device: Target device
        max_batches: Max batches per evaluation (for speed)
        
    Returns:
        Tuple of (barrier_value, details_dict)
        details_dict contains: losses_at_alphas, alphas, loss_sd1, loss_sd2, max_alpha
    """
    if device is None:
        device = next(iter(sd1.values())).device
    if loss_fn is None:
        loss_fn = F.cross_entropy
    
    # Create a working copy of the model
    model = copy.deepcopy(model)
    model.to(device)
    
    alphas = np.linspace(0.0, 1.0, n_alphas)
    
    # Compute losses at endpoints
    loss_sd1 = evaluate_loss(model, sd1, dataloader, loss_fn, device, max_batches)  # α=1
    loss_sd2 = evaluate_loss(model, sd2, dataloader, loss_fn, device, max_batches)  # α=0
    
    # Compute losses along the linear path
    losses_at_alphas = []
    barriers_at_alphas = []
    
    for alpha in alphas:
        # Interpolated loss
        interp_sd = interpolate_state_dicts(sd1, sd2, alpha)
        loss_interp = evaluate_loss(model, interp_sd, dataloader, loss_fn, device, max_batches)
        losses_at_alphas.append(loss_interp)
        
        # Frankle et al. (ICML 2020): barrier = sup loss - mean(endpoint losses)
        mean_endpoint_loss = (loss_sd1 + loss_sd2) / 2.0
        
        # Barrier at this alpha (how much above the mean endpoint loss)
        barrier_at_alpha = loss_interp - mean_endpoint_loss
        barriers_at_alphas.append(barrier_at_alpha)
    
    # Loss barrier is the maximum deviation
    barrier = max(barriers_at_alphas)
    max_alpha_idx = np.argmax(barriers_at_alphas)
    
    details = {
        'alphas': alphas.tolist(),
        'losses_at_alphas': losses_at_alphas,
        'barriers_at_alphas': barriers_at_alphas,
        'loss_sd1': loss_sd1,
        'loss_sd2': loss_sd2,
        'max_alpha': alphas[max_alpha_idx],
        'barrier': barrier
    }
    
    return barrier, details


def compute_loss_barrier_structurally_pruned(
    model_dense: nn.Module,
    model_pruned: nn.Module,
    sd_anchor: Dict[str, torch.Tensor],
    sd_pruned: Dict[str, torch.Tensor],
    dataloader: DataLoader,
    loss_fn: Callable = None,
    n_alphas: int = 11,
    device: torch.device = None,
    max_batches: int = None
) -> Tuple[float, Dict]:
    """
    Compute loss barrier for structurally pruned models.
    
    This function handles the case where sd_pruned has different shapes than sd_anchor
    by inflating the pruned state dict to match the anchor's shape.
    
    Args:
        model_dense: Dense model architecture (used for evaluation)
        model_pruned: Pruned model architecture (used to get pruned weights)
        sd_anchor: State dict for anchor model (θ₁)
        sd_pruned: State dict for pruned model (may have different shapes)
        dataloader: Data to compute loss on
        loss_fn: Loss function
        n_alphas: Number of interpolation points
        device: Target device
        max_batches: Max batches per evaluation
        
    Returns:
        Tuple of (barrier_value, details_dict)
    """
    if device is None:
        device = next(iter(sd_anchor.values())).device
    
    # Inflate pruned state dict to match anchor shape
    sd_pruned_inflated = inflate_state_dict(sd_pruned, sd_anchor, device)
    
    # Compute barrier using the dense model architecture
    return compute_loss_barrier(
        model=model_dense,
        sd1=sd_anchor,
        sd2=sd_pruned_inflated,
        dataloader=dataloader,
        loss_fn=loss_fn,
        n_alphas=n_alphas,
        device=device,
        max_batches=max_batches
    )


def get_id_barrier(
    model_dense: nn.Module,
    sd_dense_pretrained: Dict[str, torch.Tensor],
    sd_pruned: Dict[str, torch.Tensor],
    val_dataloader: DataLoader,
    loss_fn: Callable = None,
    n_alphas: int = 11,
    device: torch.device = None,
    max_batches: int = None
) -> Tuple[float, Dict]:
    """
    Compute ID barrier: B_ID(θ₀, θ̂_pruned) on clean validation data.
    
    Measures whether the pruned model remains in the pretrained basin.
    
    Args:
        model_dense: Dense model architecture
        sd_dense_pretrained: θ₀ - dense pretrained weights
        sd_pruned: Pruned model weights (pre-TTA or post-TTA)
        val_dataloader: Clean validation data (CIFAR10 test / ImageNet val)
        loss_fn: Loss function
        n_alphas: Number of interpolation points
        device: Target device
        max_batches: Max batches per evaluation
        
    Returns:
        Tuple of (barrier_value, details_dict)
    """
    # Inflate pruned weights to dense shape
    sd_pruned_inflated = inflate_state_dict(sd_pruned, sd_dense_pretrained, device)
    
    return compute_loss_barrier(
        model=model_dense,
        sd1=sd_dense_pretrained,  # Anchor: θ₀
        sd2=sd_pruned_inflated,   # Comparison: θ̂_pruned
        dataloader=val_dataloader,
        loss_fn=loss_fn,
        n_alphas=n_alphas,
        device=device,
        max_batches=max_batches
    )


def get_ood_barrier(
    model_dense: nn.Module,
    sd_dense_tta: Dict[str, torch.Tensor],
    sd_pruned: Dict[str, torch.Tensor],
    corruption_dataloader: DataLoader,
    loss_fn: Callable = None,
    n_alphas: int = 11,
    device: torch.device = None,
    max_batches: int = None
) -> Tuple[float, Dict]:
    """
    Compute OOD barrier: B_Qc(θ^post_dense(c), θ̂_pruned) on corruption data.
    
    Measures whether the pruned model reaches the same low-loss region as dense TTA.
    
    Args:
        model_dense: Dense model architecture
        sd_dense_tta: θ^post_dense(c) - dense model after TTA on corruption c
        sd_pruned: Pruned model weights (pre-TTA or post-TTA)
        corruption_dataloader: Corruption-specific data Q_c
        loss_fn: Loss function
        n_alphas: Number of interpolation points
        device: Target device
        max_batches: Max batches per evaluation
        
    Returns:
        Tuple of (barrier_value, details_dict)
    """
    # Inflate pruned weights to dense shape
    sd_pruned_inflated = inflate_state_dict(sd_pruned, sd_dense_tta, device)
    
    return compute_loss_barrier(
        model=model_dense,
        sd1=sd_dense_tta,         # Anchor: θ^post_dense(c)
        sd2=sd_pruned_inflated,   # Comparison: θ̂_pruned
        dataloader=corruption_dataloader,
        loss_fn=loss_fn,
        n_alphas=n_alphas,
        device=device,
        max_batches=max_batches
    )
