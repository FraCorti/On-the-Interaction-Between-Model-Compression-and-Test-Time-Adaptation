import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Dict, List
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, Dataset
from scipy.sparse.linalg import LinearOperator, eigsh

__all__ = [
    "pack_to_gpu_from_loader",
    "pack_first_batches_as_dataset_cpu",
    "lanczos",
    "compute_hvp",
    "get_filtered_parameters",
    "get_hessian",
]


# Dataset helpers

def pack_to_gpu_from_loader(loader, max_batches: int, device: torch.device):
    """
    Collect the first `max_batches` from a loader and return a CUDA TensorDataset.
    Use only if you have ample VRAM; otherwise prefer the CPU version below.
    """
    xs, ys = [], []
    with torch.no_grad():
        for b, (x, y) in enumerate(loader):
            xs.append(x.to(device, non_blocking=True))
            ys.append(y.to(device, non_blocking=True))
            if (b + 1) >= max_batches:
                break
    if not xs:
        raise RuntimeError("No data collected for Hessian dataset.")
    X = torch.cat(xs, 0)
    Y = torch.cat(ys, 0)
    return torch.utils.data.TensorDataset(X, Y)


def pack_first_batches_as_dataset_cpu(loader, max_batches: int):
    """
    Collect the first `max_batches` from a loader and return a CPU TensorDataset.
    Batches are streamed to the GPU during the computation.
    """
    xs, ys = [], []
    with torch.no_grad():
        for b, (x, y) in enumerate(loader):
            xs.append(x.cpu())
            ys.append(y.cpu())
            if (b + 1) >= max_batches:
                break
    if not xs:
        raise RuntimeError("No data collected for Hessian dataset.")
    return torch.utils.data.TensorDataset(torch.cat(xs, 0), torch.cat(ys, 0))


# Lanczos (eigsh) wrapper

def lanczos(device: torch.device,
            matrix_vector: Callable,
            dim: int,
            neigs: int):
    """
    Top-k eigenpairs of the symmetric operator defined implicitly by `matrix_vector`,
    via ARPACK Lanczos (scipy.sparse.linalg.eigsh); eigenvalues are returned in
    descending order. By default ARPACK draws the start vector from NumPy's RNG;
    setting HESSIAN_LANCZOS_DETERMINISTIC=1 uses a fixed start vector and fixed
    solver settings (which="LA", tol, ncv).
    """
    deterministic = os.getenv("HESSIAN_LANCZOS_DETERMINISTIC", "0") == "1"

    def mv(vec: np.ndarray):
        # SciPy feeds a NumPy array; we move to device, apply Mv, then return NumPy on CPU.
        gpu_vec = torch.tensor(vec, dtype=torch.float32, device=device)
        out_tensor = matrix_vector(gpu_vec)
        if torch.isnan(out_tensor).any():
            # Fallback: replace NaNs with zeros to prevent ARPACK crash
            out_tensor = torch.nan_to_num(out_tensor, nan=0.0)
        out = out_tensor.detach().cpu().numpy()
        return out

    operator = LinearOperator((dim, dim), matvec=mv, dtype=np.float32)

    if deterministic:
        # Fixed, unit-norm start; fixed knobs.
        v0 = np.full((dim,), 1.0 / np.sqrt(dim), dtype=np.float32)
        ncv = max(2 * neigs + 2, 20)
        evals, evecs = eigsh(operator, k=neigs, which="LA", v0=v0, tol=1e-6, ncv=ncv)
    else:
        # Default eigsh call.
        evals, evecs = eigsh(operator, k=neigs, which="LA")

    # Return in descending order
    return (torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(),
            torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float())


# HVP and parameter filtering

def compute_hvp(device: torch.device,
                model: nn.Module,
                dataset: Dataset,
                loss_fn: Callable,
                vector: torch.Tensor,
                args: any,
                physical_batch_size,
                exclude_ln: bool = True) -> torch.Tensor:
    """
    Compute H*v robustly (handles non-contiguous tensors and params unused by the graph).
    Assumes `loss_fn` returns a sum; it is divided by N for the empirical mean Hessian.
    """

    # Filter params and verify the vector length
    filtered_params, _ = get_filtered_parameters(model, exclude_ln)
    # keep only differentiable params
    filtered_params = [p for p in filtered_params if p.requires_grad]
    p = sum(p.numel() for p in filtered_params)
    n = len(dataset)

    hvp = torch.zeros(p, dtype=torch.float32, device=device)
    vector = vector.to(device)
    assert vector.numel() == p, f"probe vector length {vector.numel()} != #params {p}"

    torch.backends.cuda.enable_math_sdp(True)

    # Preprocess extraction from args if passed, or argument
    preprocess = getattr(args, 'preprocess', None) if args else None

    # Loader options from args when available
    pin_memory = getattr(args, 'pin_memory', False) if args else False
    persistent_workers = getattr(args, 'persistent_workers', False) if args else False

    dataloader = DataLoader(dataset, batch_size=physical_batch_size, shuffle=False,
                            pin_memory=pin_memory, persistent_workers=persistent_workers)

    for X, y in dataloader:
        # Folded models may need contiguous inputs
        X = X.to(device, non_blocking=True).contiguous()
        y = y.to(device, non_blocking=True).contiguous()

        if preprocess is not None:
            X = preprocess(X)

        torch.set_grad_enabled(True)  # ensure grads are enabled even in eval()

        # Enforce Math SDPA for double-backward support
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            outputs = model(X)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # reshape rather than .view: tensors may be non-contiguous
            if logits.dim() > 2:
                logits_flat = logits.reshape(-1, logits.shape[-1])
            else:
                logits_flat = logits.reshape(-1, logits.size(-1))
            y_flat = y.reshape(-1)

            # dataset-average curvature
            if "SPAConsistencyLoss" in str(type(loss_fn)):
                loss = loss_fn(model, X, logits) / n
            else:
                loss = loss_fn(logits_flat, y_flat) / n

            # First gradient; allow_unused since pruned/folded params may be disconnected
            grads = torch.autograd.grad(
                loss, inputs=filtered_params, create_graph=True, retain_graph=True, allow_unused=True
            )
            grads = [g.contiguous() if g is not None else torch.zeros_like(p_) for g, p_ in zip(grads, filtered_params)]

            # Hessian-vector product via grad of dot(grads, v)
            dot = torch.dot(parameters_to_vector(grads), vector)
            if dot.requires_grad:
                hv_list = torch.autograd.grad(
                    dot, filtered_params, retain_graph=False, allow_unused=True
                )
                hv_list = [h if h is not None else torch.zeros_like(p_) for h, p_ in zip(hv_list, filtered_params)]
            else:
                hv_list = [torch.zeros_like(p_) for p_ in filtered_params]

        hvp += parameters_to_vector([h.contiguous() for h in hv_list])

        if args.debug:
            break

    if torch.isnan(hvp).any():
        hvp = torch.nan_to_num(hvp, nan=0.0)

    return hvp


def get_filtered_parameters(model, exclude_ln=True):
    """
    Get model parameters that require gradients, optionally excluding
    LayerNorm/BatchNorm-like parameters.
    """
    filtered_params = []
    param_names = []

    for name, param in model.named_parameters():
        # Skip frozen parameters
        if not param.requires_grad:
            continue

        # Skip norm layers if requested
        if exclude_ln and 'norm' in name.lower():
            continue

        filtered_params.append(param)
        param_names.append(name)

    return filtered_params, param_names

# Public API: get_hessian

def get_hessian(device: torch.device,
                model: nn.Module,
                dataset: Dataset,
                loss_fn: nn.Module,
                neigs: int = 6,
                physical_batch_size: int = 1000,
                args=None,
                exclude_ln: bool = True,
                preprocess: Callable = None) -> tuple[Tensor, Tensor]:
    """
    Compute the leading Hessian eigenpairs (evals, evecs) of the empirical mean loss
    with Lanczos (descending eigenvalues). The loss uses reduction='sum' divided by N
    and data are streamed from `dataset` with a DataLoader. Set
    HESSIAN_LANCZOS_DETERMINISTIC=1 for a deterministic start vector.
    """

    def hvp_delta(delta):
        # compute_hvp reads `preprocess` from args
        if args is not None and preprocess is not None:
            setattr(args, 'preprocess', preprocess)
        return compute_hvp(device, model, dataset, loss_fn, delta, physical_batch_size=physical_batch_size,
                           exclude_ln=exclude_ln, args=args)

    filtered_params, _ = get_filtered_parameters(model, exclude_ln)
    # Filter by requires_grad to match compute_hvp behavior and prevent vector size mismatch
    filtered_params = [p for p in filtered_params if p.requires_grad]
    nparams = len(parameters_to_vector(filtered_params))

    evals, evecs = lanczos(device, hvp_delta, nparams, neigs=neigs)
    return evals, evecs


def compute_fisher_trace_subspace(device: torch.device,
                                  model: nn.Module,
                                  dataset: Dataset,
                                  loss_fn: Callable,
                                  physical_batch_size: int = 128,
                                  args=None,
                                  preprocess: Callable = None) -> float:
    """
    Compute the trace of the Fisher Information Matrix (FIM) restricted to the subspace U
    (BatchNorm or LayerNorm affine parameters).
    Trace(F) = E[||grad(log p(y|x))||^2]
    We approximate this by the average squared norm of the gradients of the loss function
    over the dataset.
    """
    # Parameters in U (BN/LN affine params)
    params_in_U = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if module.weight is not None and module.weight.requires_grad:
                params_in_U.append(module.weight)
            if module.bias is not None and module.bias.requires_grad:
                params_in_U.append(module.bias)

    n_params = sum(p.numel() for p in params_in_U)
    if not params_in_U:
        print("Warning: No parameters found in subspace U (BN/LN). Returning 0.")
        return {"raw": 0.0, "normalized": 0.0, "n_params": 0}

    # Accumulate squared gradient norms over the dataset
    pin_memory = getattr(args, 'pin_memory', False) if args else False
    persistent_workers = getattr(args, 'persistent_workers', False) if args else False

    model.eval()
    # SAR/Oracle configure_model disables BN running stats; re-enable them so
    # BN uses stored statistics in eval mode with batch_size=1.
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.track_running_stats = True
            if m.running_mean is None:
                m.running_mean = torch.zeros(m.num_features, device=device)
                m.running_var = torch.ones(m.num_features, device=device)

    total_norm_sq = 0.0
    count = 0
    max_samples = 2000  # Limit to 2000 samples for estimation to be fast enough

    # Use a separate loader with BS=1 for correct Fisher estimation
    loader_bs1 = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2,
                            pin_memory=pin_memory, persistent_workers=persistent_workers)

    for i, (X, y) in enumerate(loader_bs1):
        if i >= max_samples:
            break

        X = X.to(device).float()  # Ensure float32
        y = y.to(device)

        if preprocess is not None:
            X = preprocess(X)

        model.zero_grad()

        # Enforce Math SDPA for double-backward support
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            output = model(X)
            
            if "SPAConsistencyLoss" in str(type(loss_fn)):
                 loss = loss_fn(model, X, output)
            else:
                 loss = loss_fn(output, y)

            grads = torch.autograd.grad(loss, params_in_U, create_graph=False, allow_unused=True)
            # Handle unused gradients (None) by replacing them with zeros
            grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params_in_U)]


        # Square and sum
        norm_sq = sum(torch.sum(g ** 2).item() for g in grads)
        total_norm_sq += norm_sq
        count += 1

        if args.debug:
            break

    raw_trace = total_norm_sq / count if count > 0 else 0.0
    normalized_trace = raw_trace / n_params if n_params > 0 else 0.0
    return {"raw": raw_trace, "normalized": normalized_trace, "n_params": n_params}


def compute_activation_outlier_metrics(device: torch.device,
                                       model: nn.Module,
                                       dataset: Dataset,
                                       physical_batch_size: int = 128,
                                       max_samples: int = 1000,
                                       args=None) -> tuple[float, float]:
    """
    Compute outlier metrics for activations:
    1. Max-to-Mean Ratio (MMR): max(|x|) / mean(|x|)
    2. Kurtosis: E[(x-mu)^4] / sigma^4
    
    We compute these on the *inputs* to Conv2d and Linear layers.
    Returns averaged metrics over all monitored layers.
    """

    # Register hooks
    hooks = []
    layer_metrics = {}  # id(module) -> {'mmr': [], 'kurtosis': []}

    def get_activation_hook(module):
        def hook(model, input, output):
            # input is a tuple (x,)
            x = input[0].detach()
            # Flatten: (B, ...) -> (B, N)
            x_flat = x.reshape(x.size(0), -1)

            # MMR
            x_abs = x_flat.abs()
            # Avoid division by zero
            mean_abs = x_abs.mean(dim=1)
            max_abs = x_abs.max(dim=1).values
            mmr = max_abs / (mean_abs + 1e-8)

            # Pearson kurtosis (normal=3)
            mean = x_flat.mean(dim=1, keepdim=True)
            var = x_flat.var(dim=1, unbiased=False, keepdim=True)
            std = var.sqrt()
            diff = x_flat - mean
            fourth_moment = (diff ** 4).mean(dim=1)
            kurtosis = fourth_moment / (var ** 2 + 1e-8)

            if id(module) not in layer_metrics:
                layer_metrics[id(module)] = {'mmr': [], 'kurtosis': []}

            layer_metrics[id(module)]['mmr'].extend(mmr.cpu().tolist())
            layer_metrics[id(module)]['kurtosis'].extend(kurtosis.cpu().tolist())

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(get_activation_hook(module)))

    if not hooks:
        return 0.0, 0.0

    # Iterate over the dataset
    pin_memory = getattr(args, 'pin_memory', False) if args else False
    persistent_workers = getattr(args, 'persistent_workers', False) if args else False

    dataloader = DataLoader(dataset, batch_size=physical_batch_size, shuffle=True,
                            pin_memory=pin_memory, persistent_workers=persistent_workers)

    model.eval()
    total_samples = 0

    with torch.no_grad():
        for X, _ in dataloader:
            if total_samples >= max_samples:
                break
            X = X.to(device, non_blocking=True)
            model(X)
            total_samples += X.size(0)

    for h in hooks:
        h.remove()

    # Average over samples for each layer, then over layers
    avg_mmrs = []
    avg_kurtoses = []

    for mod_id, metrics in layer_metrics.items():
        if metrics['mmr']:
            avg_mmrs.append(np.mean(metrics['mmr']))
            avg_kurtoses.append(np.mean(metrics['kurtosis']))

    final_mmr = np.mean(avg_mmrs) if avg_mmrs else 0.0
    final_kurtosis = np.mean(avg_kurtoses) if avg_kurtoses else 0.0

    return float(final_mmr), float(final_kurtosis)


# Expected Calibration Error (ECE)

def compute_ece(
    device: torch.device,
    model: nn.Module,
    dataset: Dataset,
    physical_batch_size: int = 128,
    n_bins: int = 15,
    max_samples: int = 5000,
    args=None,
    preprocess: Callable = None
) -> float:
    """
    Compute Expected Calibration Error (ECE) for a classification model.
    
    ECE measures the discrepancy between predicted confidence and actual accuracy.
    A well-calibrated model should have ECE close to 0.
    
    Reference: Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
    
    Args:
        device: GPU/CPU device
        model: Classification model
        dataset: Dataset to evaluate on
        physical_batch_size: Batch size for forward pass
        n_bins: Number of bins for calibration (default 15)
        max_samples: Maximum samples to use
        args: Additional arguments
        preprocess: Optional preprocessing function
        
    Returns:
        ECE value (0 = perfect calibration, 1 = worst)
    """
    loader = DataLoader(dataset, batch_size=physical_batch_size, shuffle=False, num_workers=2)
    
    all_confidences = []
    all_predictions = []
    all_labels = []
    
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for X, y in loader:
            if sample_count >= max_samples:
                break
                
            X = X.to(device).float()  # Ensure float32
            y = y.to(device)
            
            if preprocess is not None:
                X = preprocess(X)
            
            logits = model(X)
            softmax = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(softmax, dim=1)
            
            all_confidences.append(confidences.cpu())
            all_predictions.append(predictions.cpu())
            all_labels.append(y.cpu())
            
            sample_count += X.size(0)
            
            if args is not None and getattr(args, 'debug', False):
                break
    
    confidences = torch.cat(all_confidences)
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    
    accuracies = predictions.eq(labels).float()
    
    # Compute ECE using binning
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().sum() / total_samples
        
        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * abs(avg_confidence - avg_accuracy)
    
    return ece.item()


# Gradient Signal-to-Noise Ratio (SNR)

def compute_gradient_snr(
    device: torch.device,
    model: nn.Module,
    dataset: Dataset,
    loss_fn: Callable,
    physical_batch_size: int = 1,
    max_samples: int = 128,
    args=None,
    preprocess: Callable = None
) -> float:
    """
    Compute Gradient Signal-to-Noise Ratio (SNR) for the adaptation subspace U.
    
    SNR = ||E[g]||^2 / E[||g - E[g]||^2]
    
    Higher SNR indicates more consistent gradient direction, suggesting stable adaptation.
    Lower SNR indicates noisy gradients, potentially unstable adaptation.
    
    Reference: Roberts & Rojas "Signal-to-Noise Ratio Analysis of Policy Gradient Algorithms" (NeurIPS 2008)
    
    Args:
        device: GPU/CPU device
        model: Neural network model
        dataset: Dataset to compute SNR over
        loss_fn: Loss function for gradient computation
        physical_batch_size: Batch size (1 gives cleanest estimate)
        max_samples: Maximum samples to use
        args: Additional arguments
        preprocess: Optional preprocessing function
        
    Returns:
        SNR value (higher = more stable gradients)
    """
    # Identify subspace U (BN/LN affine parameters)
    params_in_U = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if module.weight is not None and module.weight.requires_grad:
                params_in_U.append(module.weight)
            if hasattr(module, 'bias') and module.bias is not None and module.bias.requires_grad:
                params_in_U.append(module.bias)
    
    if not params_in_U:
        print("Warning: No subspace params found for SNR. Using all trainable.")
        params_in_U = [p for p in model.parameters() if p.requires_grad]
    
    if not params_in_U:
        return 0.0
    
    loader = DataLoader(dataset, batch_size=physical_batch_size, shuffle=True, num_workers=2)
    
    # Accumulate gradient statistics
    grad_sum = None
    grad_sq_sum = None
    count = 0
    
    model.eval()
    
    for i, (X, y) in enumerate(loader):
        if i >= max_samples:
            break
            
        X = X.to(device).float()  # Ensure float32
        y = y.to(device)
        
        if preprocess is not None:
            X = preprocess(X)
        
        model.zero_grad()
        
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            output = model(X)
            
            # Handle different loss function types
            if "SPAConsistencyLoss" in str(type(loss_fn)):
                loss = loss_fn(model, X, output)
            else:
                loss = loss_fn(output, y)
            
            grads = torch.autograd.grad(loss, params_in_U, create_graph=False, allow_unused=True)
            
            # Flatten all gradients into a single vector (skip if all None)
            grad_list = [g.view(-1) for g in grads if g is not None]
            if not grad_list:
                continue  # Skip batch if no gradients
            grad_vec = torch.cat(grad_list)
            
            if grad_sum is None:
                grad_sum = grad_vec.clone()
                grad_sq_sum = grad_vec ** 2
            else:
                grad_sum = grad_sum + grad_vec
                grad_sq_sum = grad_sq_sum + grad_vec ** 2
            
            count += 1
        
        if args is not None and getattr(args, 'debug', False):
            break
    
    if count == 0:
        return 0.0
    
    # Compute mean and variance
    mean_grad = grad_sum / count
    mean_sq_grad = grad_sq_sum / count
    variance = mean_sq_grad - mean_grad ** 2
    
    # SNR = ||E[g]||^2 / Var[g]
    signal = (mean_grad ** 2).sum()
    noise = variance.sum().clamp(min=1e-10)  # Avoid division by zero
    
    snr = (signal / noise).item()
    
    return snr


def compute_layerwise_gradient_snr(
    device: torch.device,
    model: nn.Module,
    dataset: Dataset,
    loss_fn: Callable,
    physical_batch_size: int = 1,
    max_samples: int = 128,
    args=None,
    preprocess: Callable = None
) -> Dict[str, float]:
    """
    Compute layer-wise Gradient Signal-to-Noise Ratio (SNR) for each BN/LN layer.
    
    SNR_layer = ||E[g_layer]||^2 / Var[g_layer]
    
    High SNR = stable gradients; Low SNR = noisy updates.
    Returns dict mapping layer name -> SNR value.
    
    Reference: Roberts & Rojas, NeurIPS 2008.
    """
    # BN/LN layers and their parameters
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
        print("Warning: No BN/LN layers found. Returning empty dict.")
        return {}
    
    # Initialize per-layer accumulators
    layer_grad_sum: Dict[str, torch.Tensor] = {}
    layer_grad_sq_sum: Dict[str, torch.Tensor] = {}
    
    for name, params in layer_params.items():
        total_size = sum(p.numel() for p in params)
        layer_grad_sum[name] = torch.zeros(total_size, device=device)
        layer_grad_sq_sum[name] = torch.zeros(total_size, device=device)
    
    pin_memory = getattr(args, 'pin_memory', False) if args else False
    loader = DataLoader(dataset, batch_size=physical_batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
    
    model.eval()
    count = 0
    
    # Accumulate gradients per layer
    for X, y in loader:
        if count >= max_samples:
            break
        
        X = X.to(device).float()  # Ensure float32
        y = y.to(device)
        
        if preprocess is not None:
            X = preprocess(X)
        
        model.zero_grad()
        
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            output = model(X)
            
            if "SPAConsistencyLoss" in str(type(loss_fn)):
                loss = loss_fn(model, X, output)
            else:
                loss = loss_fn(output, y)
            
            # Compute gradients for all params at once
            all_params = []
            param_to_layer = {}
            for layer_name, params in layer_params.items():
                for p in params:
                    all_params.append(p)
                    param_to_layer[id(p)] = layer_name
            
            grads = torch.autograd.grad(loss, all_params, create_graph=False, allow_unused=True)
            
            # Accumulate per layer
            layer_grads: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_params}
            for p, g in zip(all_params, grads):
                layer_name = param_to_layer[id(p)]
                if g is not None:
                    layer_grads[layer_name].append(g.view(-1))
            
            # Skip batch if no gradients at all
            has_any_grads = any(layer_grads[name] for name in layer_params)
            if not has_any_grads:
                continue
            
            for name in layer_params:
                if layer_grads[name]:
                    grad_vec = torch.cat(layer_grads[name])
                    layer_grad_sum[name] += grad_vec
                    layer_grad_sq_sum[name] += grad_vec ** 2
        
        count += 1
        
        if args is not None and getattr(args, 'debug', False):
            break
    
    if count == 0:
        return {name: 0.0 for name in layer_params}
    
    # SNR per layer
    layer_snr: Dict[str, float] = {}
    
    for name in layer_params:
        mean_grad = layer_grad_sum[name] / count
        mean_sq_grad = layer_grad_sq_sum[name] / count
        variance = mean_sq_grad - mean_grad ** 2
        
        signal = (mean_grad ** 2).sum()
        noise = variance.sum().clamp(min=1e-10)
        
        layer_snr[name] = (signal / noise).item()
    
    return layer_snr


def get_sorted_layer_snr_list(layer_snr: Dict[str, float]) -> List[float]:
    """Convert layer SNR dict to sorted list (by layer name) for consistent plotting."""
    sorted_names = sorted(layer_snr.keys())
    return [layer_snr[name] for name in sorted_names]


def compute_layerwise_hessian_trace(
    device: torch.device,
    model: nn.Module,
    dataset: Dataset,
    loss_fn: Callable,
    max_samples: int = 128,
    n_hutchinson_iters: int = 10,
    args=None,
    preprocess=None
) -> Dict[str, float]:
    """
    Compute layer-wise Hessian trace using Hutchinson's method.
    
    For each adaptation layer (BN/LN), estimate Tr(H_l) where H_l is the Hessian
    restricted to layer l's parameters.
    
    Uses Hutchinson estimator: Tr(H) ≈ E[v^T H v] where v is Rademacher random.
    
    Args:
        device: CUDA or CPU
        model: Neural network model
        dataset: Dataset for Hessian computation
        loss_fn: Loss function
        max_samples: Maximum samples to use
        n_hutchinson_iters: Number of Hutchinson iterations for trace estimation
        args: Optional args object
        preprocess: Optional preprocessing transform
        
    Returns:
        Dict mapping layer name to its Hessian trace estimate
    """
    model = model.to(device)
    model.eval()
    
    # Adaptation layers (BN/LN with affine params)
    layer_params: Dict[str, List[nn.Parameter]] = {}
    param_to_layer: Dict[int, str] = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                params = []
                if module.weight is not None and module.weight.requires_grad:
                    params.append(module.weight)
                    param_to_layer[id(module.weight)] = name
                if hasattr(module, 'bias') and module.bias is not None and module.bias.requires_grad:
                    params.append(module.bias)
                    param_to_layer[id(module.bias)] = name
                if params:
                    layer_params[name] = params
    
    if not layer_params:
        return {}
    
    batch_size = getattr(args, 'm_sharpness', 32) if args else 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Hutchinson trace estimation per layer
    layer_traces: Dict[str, List[float]] = {name: [] for name in layer_params}
    
    for hutchinson_iter in range(n_hutchinson_iters):
        # Generate Rademacher vectors for each layer
        layer_v: Dict[str, List[torch.Tensor]] = {}
        for name, params in layer_params.items():
            layer_v[name] = []
            for p in params:
                v = torch.randint(0, 2, p.shape, device=device, dtype=p.dtype) * 2 - 1
                layer_v[name].append(v)
        
        # Accumulate Hv^T v over dataset
        layer_vhv: Dict[str, float] = {name: 0.0 for name in layer_params}
        n_samples = 0
        
        for batch_idx, (X, y) in enumerate(dataloader):
            if n_samples >= max_samples:
                break
                
            X = X.to(device, non_blocking=True).contiguous()
            y = y.to(device, non_blocking=True).contiguous()
            
            if preprocess is not None:
                X = preprocess(X)
            
            batch_size_actual = X.size(0)
            n_samples += batch_size_actual
            
            model.zero_grad()
            
            # Forward pass
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                outputs = model(X)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                if logits.dim() > 2:
                    logits = logits.reshape(-1, logits.shape[-1])
                y_flat = y.reshape(-1)
                
                if "SPAConsistencyLoss" in str(type(loss_fn)):
                    loss = loss_fn(model, X, logits)
                else:
                    loss = loss_fn(logits, y_flat)
            
            # Compute gradient for each layer and then HVP
            for name, params in layer_params.items():
                grads = torch.autograd.grad(
                    loss, params, create_graph=True, retain_graph=True, allow_unused=True
                )
                grads = [g.contiguous() if g is not None else torch.zeros_like(p) 
                        for g, p in zip(grads, params)]
                
                # Compute v^T grad
                v_list = layer_v[name]
                dot = sum(torch.sum(v * g) for v, g in zip(v_list, grads))
                
                # Compute Hv via grad of dot
                if dot.requires_grad:
                    Hv = torch.autograd.grad(
                        dot, params, retain_graph=True, allow_unused=True
                    )
                    Hv = [h if h is not None else torch.zeros_like(p) 
                          for h, p in zip(Hv, params)]
                else:
                    Hv = [torch.zeros_like(p) for p in params]
                
                # v^T H v
                vhv = sum(torch.sum(v * h) for v, h in zip(v_list, Hv))
                layer_vhv[name] += vhv.item() * batch_size_actual
            
            if args is not None and getattr(args, 'debug', False) and batch_idx >= 2:
                break
        
        # Average over samples and store
        for name in layer_params:
            if n_samples > 0:
                layer_traces[name].append(layer_vhv[name] / n_samples)
    
    # Average over Hutchinson iterations
    result: Dict[str, float] = {}
    for name in layer_params:
        if layer_traces[name]:
            result[name] = float(np.mean(layer_traces[name]))
        else:
            result[name] = 0.0
    
    return result


def get_sorted_layer_hessian_trace_list(layer_traces: Dict[str, float]) -> List[float]:
    """Convert layer Hessian trace dict to sorted list (by layer name) for consistent plotting."""
    sorted_names = sorted(layer_traces.keys())
    return [layer_traces[name] for name in sorted_names]
