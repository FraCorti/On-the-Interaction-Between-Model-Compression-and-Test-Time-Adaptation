"""
Metric Similarity Analysis (MSA), Cayco-Gajic & Pellegrino (2026), arXiv:2603.28764.

Two networks are compared through the pullback metrics G(p) = J(p)^T J(p) of the
output Euclidean metric onto the input manifold; the spectral-ratio distance
d_SR(G^1, G^2) = 1 - sqrt(lambda_min / lambda_max) of the generalised eigenvalues
is averaged over samples. For image inputs the Jacobian is evaluated only along
the top-K PCA directions of a calibration batch, so G is K x K.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Core building blocks


def _flatten_output(y: torch.Tensor) -> torch.Tensor:
    """Flatten everything past the batch dim. Works for logits, feature maps,
    or arbitrary intermediate tensors."""
    return y.reshape(y.shape[0], -1)


def fit_pca_basis(samples: torch.Tensor, n_components: int) -> torch.Tensor:
    """Fit a PCA basis on a batch of inputs.

    Parameters
    ----------
    samples : (N, *input_shape) tensor
        Calibration batch. Flattened internally.
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    basis : (K, D) tensor
        Top-K right-singular-vectors of the (centred) data matrix, where
        D = prod(input_shape). Rows are the directions along which we will
        compute Jacobian-vector products.
    """
    flat = samples.reshape(samples.shape[0], -1).float()
    mean = flat.mean(dim=0, keepdim=True)
    centred = flat - mean
    # economy SVD: U(N x r) S(r) V^T(r x D)
    _, _, vt = torch.linalg.svd(centred, full_matrices=False)
    k = min(n_components, vt.shape[0])
    return vt[:k].contiguous()


def pullback_metric(
    net: nn.Module,
    x: torch.Tensor,
    basis: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pullback metric G(p) = J(p)^T J(p) restricted to a basis subspace.

    Parameters
    ----------
    net : nn.Module
        Network to differentiate. Assumed to be in eval mode.
    x : tensor with shape matching net's input (no batch dim, or a single
        sample with batch dim 1).
    basis : (K, D) tensor or None
        Basis directions in flattened input space. If None, uses the
        identity, i.e. full-input Jacobian (only sane for tiny D).
    eps : float
        Tikhonov regularisation added to the diagonal of G to keep it SPD
        in finite precision.

    Returns
    -------
    G : (K, K) tensor
        Pullback metric expressed in the chosen basis. Symmetric SPD by
        construction.
    """
    # Detect whether x already has a leading batch dim of size 1, otherwise
    # treat it as an unbatched single sample and add one.
    if x.dim() >= 1 and x.shape[0] == 1 and x.dim() >= 2:
        x_b = x
    else:
        x_b = x.unsqueeze(0)

    input_shape = tuple(x_b.shape[1:])
    d = 1
    for s in input_shape:
        d *= int(s)

    if basis is None:
        basis = torch.eye(d, device=x_b.device, dtype=x_b.dtype)
    else:
        basis = basis.to(device=x_b.device, dtype=x_b.dtype)

    k = basis.shape[0]

    def f(flat_input: torch.Tensor) -> torch.Tensor:
        # flat_input: (D,) -> reshape to input shape with batch dim
        y = net(flat_input.reshape((1,) + tuple(input_shape)))
        return _flatten_output(y).squeeze(0)

    flat_x = x_b.reshape(d)

    # Build J_K of shape (output_dim, K) by jvp along each basis direction.
    # torch.autograd.functional.jvp returns (output, jvp_value).
    jvp_cols = []
    for i in range(k):
        v = basis[i]
        _, jv = torch.autograd.functional.jvp(
            f, (flat_x,), (v,), create_graph=False, strict=False
        )
        jvp_cols.append(jv)
    J_K = torch.stack(jvp_cols, dim=1)  # (output_dim, K)

    G = J_K.t() @ J_K  # (K, K)
    # Symmetrise (kill numerical drift) and add small Tikhonov term.
    G = 0.5 * (G + G.t())
    G = G + eps * torch.eye(k, device=G.device, dtype=G.dtype)
    return G


def spectral_ratio(G1: torch.Tensor, G2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Spectral ratio distance d_SR between two SPD matrices.

    Solves the generalised eigenvalue problem G1 v = lambda G2 v via
    Cholesky pre-conditioning (G2 = L L^T, H = L^{-1} G1 L^{-T}, eigvalsh(H)).
    Falls back to a regularised inverse if the Cholesky fails.

    Returns d_SR = 1 - sqrt(lambda_min / lambda_max) in [0, 1].
    """
    assert G1.shape == G2.shape and G1.dim() == 2 and G1.shape[0] == G1.shape[1], (
        "G1, G2 must be square matrices of the same size"
    )
    k = G1.shape[0]
    I = torch.eye(k, device=G1.device, dtype=G1.dtype)
    G1 = 0.5 * (G1 + G1.t()) + eps * I
    G2 = 0.5 * (G2 + G2.t()) + eps * I

    try:
        L = torch.linalg.cholesky(G2)
    except Exception:
        # Increase regularisation and retry once.
        G2 = G2 + 1e-3 * I
        L = torch.linalg.cholesky(G2)

    # H = L^{-1} G1 L^{-T}.
    L_inv_G1 = torch.linalg.solve_triangular(L, G1, upper=False)
    H = torch.linalg.solve_triangular(L, L_inv_G1.t(), upper=False).t()
    H = 0.5 * (H + H.t())
    eigvals = torch.linalg.eigvalsh(H)
    # Numerical safety: clamp.
    eigvals = torch.clamp(eigvals, min=eps)
    lam_min = eigvals.min()
    lam_max = eigvals.max()
    ratio = torch.sqrt(lam_min / lam_max)
    d = 1.0 - ratio
    # Numerical guard: keep inside [0, 1].
    return torch.clamp(d, min=0.0, max=1.0)


# Aggregate MSA over a dataloader


def _iter_samples(dataloader: DataLoader, n_samples: int):
    """Yield up to n_samples individual inputs from a DataLoader."""
    count = 0
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        for i in range(x.shape[0]):
            if count >= n_samples:
                return
            yield x[i]
            count += 1


def msa_distance(
    net1: nn.Module,
    net2: nn.Module,
    dataloader: DataLoader,
    n_samples: int = 32,
    n_components: int = 8,
    device: Optional[torch.device] = None,
    basis: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """Mean +/- std of the spectral-ratio distance between two networks.

    Parameters
    ----------
    net1, net2 : nn.Module
        Networks to compare. Must accept inputs of the same shape and
        produce outputs of the same shape. Set to eval() inside.
    dataloader : DataLoader
        Source of input samples. The first element of each batch is treated
        as the input. The first batch is also used to fit the PCA basis
        (unless `basis` is supplied).
    n_samples : int
        Number of points p_i over which to average d_SR.
    n_components : int
        Number of PCA components K to retain for the input-manifold
        approximation.
    device : torch.device or None
        Device on which to perform all computations.
    basis : (K, D) tensor or None
        Pre-computed PCA basis. If None it is fit from the first batch of
        `dataloader`.
    eps : float
        Tikhonov regularisation.

    Returns
    -------
    mean, std : float, float
        Sample mean and sample standard deviation of d_SR over n_samples.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net1 = net1.to(device).eval()
    net2 = net2.to(device).eval()

    # Fit PCA basis from the first batch if not supplied.
    if basis is None:
        first = next(iter(dataloader))
        x0 = first[0] if isinstance(first, (tuple, list)) else first
        x0 = x0.to(device)
        basis = fit_pca_basis(x0, n_components=n_components).to(device)

    distances = []
    for x in _iter_samples(dataloader, n_samples):
        x = x.to(device)
        G1 = pullback_metric(net1, x, basis=basis, eps=eps)
        G2 = pullback_metric(net2, x, basis=basis, eps=eps)
        d = spectral_ratio(G1, G2, eps=eps)
        distances.append(d.detach().item())

    if len(distances) == 0:
        return 0.0, 0.0
    t = torch.tensor(distances)
    if t.numel() == 1:
        return float(t.item()), 0.0
    return float(t.mean().item()), float(t.std(unbiased=False).item())


def msa_similarity(
    net1: nn.Module,
    net2: nn.Module,
    dataloader: DataLoader,
    n_samples: int = 32,
    n_components: int = 8,
    device: Optional[torch.device] = None,
    basis: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """MSA similarity = 1 - MSA distance, in [0, 1]. Returns (mean, std)."""
    mean, std = msa_distance(
        net1,
        net2,
        dataloader,
        n_samples=n_samples,
        n_components=n_components,
        device=device,
        basis=basis,
        eps=eps,
    )
    return 1.0 - mean, std
