"""Layer-wise Activation Map Entropy (AME) and effective rank.

For each layer l, AME is the Shannon entropy of the normalized eigenvalues of
the activation Gram matrix G_l in R^{C_l x C_l} (Sec. 3.2 of the paper).
Following Skean et al. (2025), each (sample, spatial position) or (sample, token)
pair is one C_l-dim feature vector: activations are reshaped to (B*H*W, C) so the
Gram matrix stays C_l x C_l for any batch size, whereas a (B, C*H*W) reshape would
bound the rank by min(B, C*H*W). Multi-batch accumulation sums per-batch Grams.
"""

import torch
import torch.nn as nn
from collections import OrderedDict


def _reshape_to_samples_x_channels(x: torch.Tensor) -> torch.Tensor:
    """Reshape an activation tensor so rows index (batch, spatial)/(batch, token)
    samples and columns index the C_l feature channels.

    Conv outputs (B, C, H, W) -> (B*H*W, C) via permute + reshape.
    Sequence outputs (B, T, D) -> (B*T, D).
    Already-2D outputs (B, C) are returned unchanged.
    """
    if x.dim() == 4:                       # (B, C, H, W): Conv2d / BasicBlock
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    if x.dim() == 3:                       # (B, T, D): ViT Block / LayerNorm
        return x.reshape(-1, x.shape[-1])
    if x.dim() == 2:                       # already (N, D)
        return x
    raise ValueError(f"Unsupported activation rank: {x.dim()}")


def effective_rank_from_gram(gram: torch.Tensor, eps: float = 1e-10) -> tuple:
    """Effective rank = exp(Shannon entropy of normalized eigenvalues of `gram`).

    `gram` must be symmetric PSD of shape (C, C).  Returns (entropy, eff_rank)
    where `entropy` is in nats and `eff_rank = exp(entropy)` per Roy & Vetterli
    (2007).  Negative eigenvalues from numerical jitter are clamped to 0.
    """
    eigvals = torch.linalg.eigvalsh(gram).clamp_min(0)
    total = eigvals.sum()
    if total <= 0:
        return torch.tensor(0.0, device=gram.device), torch.tensor(1.0, device=gram.device)
    p = eigvals / total
    entropy = -(p * (p + eps).log()).sum()
    return entropy, entropy.exp()


def compute_layerwise_ame_paper_recipe(model, dataloader, device='cuda',
                                       num_batches: int = 20,
                                       target_layer_types=None,
                                       eps: float = 1e-6):
    """AME with per-batch entropy averaged over batches.

    For each batch b and hooked layer l, build the uncentered C_l x C_l Gram
    G_b = sum_{n,s} A[n,c,s] A[n,k,s], add a Tikhonov ridge eps * tr(G_b)/C * I,
    and take the Shannon entropy H_{b,l} (nats) of its normalized eigenvalues.
    AME_l = mean_b H_{b,l} and eff_rank_l = exp(AME_l). Unlike
    `compute_layerwise_effective_ranks`, the Gram is uncentered and the entropy
    is averaged over batches rather than taken on the pooled covariance.

    Returns OrderedDict {layer_name: eff_rank}; the entropy is `log(eff_rank)`.
    """
    import math
    model.eval()
    model.to(device)

    if target_layer_types is None:
        target_layer_types = (nn.Conv2d, nn.Linear, nn.LayerNorm, nn.BatchNorm2d)
    else:
        target_layer_types = tuple(target_layer_types)

    sum_H = OrderedDict()
    cnt_H = OrderedDict()
    # Per-batch container; the closure rebinds this name.
    batch_grams = {}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            x = output[0] if isinstance(output, (list, tuple)) else output
            if not torch.is_tensor(x):
                return
            with torch.no_grad():
                x = x.detach().to(torch.float32)
                if x.dim() == 4:
                    N, C, H_dim, W_dim = x.shape
                    A = x.view(N, C, H_dim * W_dim)
                    Gb = torch.einsum('ncs,nks->ck', A, A)   # C x C
                elif x.dim() == 3:
                    # ViT block output: (N, T, D)
                    N, T, D = x.shape
                    A = x  # (N, T, D)
                    Gb = torch.einsum('ntc,ntk->ck', A, A)   # D x D
                elif x.dim() == 2:
                    Gb = x.t().mm(x)
                else:
                    return
                batch_grams[name] = batch_grams.get(name, 0) + Gb.to('cpu')
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, target_layer_types):
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                # Reset the per-batch container before the forward pass.
                batch_grams.clear()
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device, non_blocking=True)
                model(x)
                # Compute per-layer per-batch entropy.
                for name, G in batch_grams.items():
                    C = G.shape[0]
                    if eps > 0.0:
                        tr = float(torch.trace(G))
                        if tr > 0.0:
                            ridge = eps * tr / max(C, 1)
                            G = G + ridge * torch.eye(C, dtype=G.dtype, device=G.device)
                    G = 0.5 * (G + G.t())
                    entropy, _ = effective_rank_from_gram(G)
                    h_val = float(entropy.item())
                    if not (math.isnan(h_val) or math.isinf(h_val)):
                        sum_H[name] = sum_H.get(name, 0.0) + h_val
                        cnt_H[name] = cnt_H.get(name, 0) + 1
    finally:
        for h in hooks:
            h.remove()

    results = OrderedDict()
    for name in sum_H:
        n = cnt_H[name]
        if n <= 0:
            results[name] = 1.0
            continue
        mean_H = sum_H[name] / n
        results[name] = math.exp(mean_H)
    return results


def compute_layerwise_effective_ranks(model, dataloader, device='cuda',
                                      num_batches: int = 20,
                                      target_layer_types=None,
                                      center: bool = True,
                                      return_entropy: bool = False):
    """Layer-wise AME / effective rank over `num_batches` mini-batches.

    Args:
        model: PyTorch model (already on `device`).
        dataloader: yields (inputs, targets); only `inputs` are used.
        device: 'cuda' or 'cpu'.
        num_batches: how many mini-batches to aggregate into the Gram matrix.
        target_layer_types: tuple of nn.Module subclasses to hook.  Defaults
            to (Conv2d, Linear, LayerNorm, BatchNorm2d).  For post-residual
            analysis pass (BasicBlock_Custom, BasicBlock_TV) for ResNet and
            (Block,) for ViT.
        center: subtract the per-channel mean across all aggregated samples
            before forming the Gram matrix.  The Gram of centered features
            is the covariance up to a scale; eigvals are then variances.
        return_entropy: if True, return (entropy_nats, eff_rank) tuples
            rather than just `eff_rank`.

    Returns:
        OrderedDict {layer_name: eff_rank} or {layer_name: (entropy, eff_rank)}.
    """
    model.eval()
    model.to(device)

    if target_layer_types is None:
        target_layer_types = (nn.Conv2d, nn.Linear, nn.LayerNorm, nn.BatchNorm2d)
    else:
        target_layer_types = tuple(target_layer_types)

    # Per-layer running Gram accumulators and sample counts.
    gram_acc = OrderedDict()
    mean_acc = OrderedDict()
    count_acc = OrderedDict()

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            x = _reshape_to_samples_x_channels(output.detach().float())
            n, c = x.shape
            # Welford-style running mean keeps centering accurate across batches.
            if name not in gram_acc:
                gram_acc[name] = torch.zeros(c, c, device=device, dtype=torch.float32)
                mean_acc[name] = torch.zeros(c, device=device, dtype=torch.float32)
                count_acc[name] = 0
            count_acc[name] += n
            mean_acc[name] += x.sum(dim=0)
            gram_acc[name] += x.t() @ x         # X^T X over (B*H*W) samples
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, target_layer_types):
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device, non_blocking=True)
                model(x)
    finally:
        for h in hooks:
            h.remove()

    results = OrderedDict()
    for name in gram_acc:
        n_total = count_acc[name]
        if n_total < 2:
            results[name] = (torch.tensor(0.0), torch.tensor(1.0)) if return_entropy else 1.0
            continue
        gram = gram_acc[name]
        if center:
            # Cov = (X^T X - n * mu mu^T) / n.  mu = sum / n.
            mu = mean_acc[name] / n_total
            gram = gram - n_total * torch.outer(mu, mu)
        gram = gram / max(n_total - 1, 1)        # unbiased covariance
        # Symmetrize against accumulated float jitter.
        gram = 0.5 * (gram + gram.t())
        entropy, eff_rank = effective_rank_from_gram(gram)
        if return_entropy:
            results[name] = (entropy.item(), eff_rank.item())
        else:
            results[name] = eff_rank.item()

    return results
