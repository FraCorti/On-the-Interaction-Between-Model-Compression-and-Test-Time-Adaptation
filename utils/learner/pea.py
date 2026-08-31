"""
PEA, Progressive Embedding Alignment (Xiao et al., ICLR 2026).
Adapted from the official code: https://github.com/TheMaXiao/PEA_TTA

Backpropagation-free TTA: per-block CORAL alignment of intermediate activations
between offline source statistics and an online EMA of target statistics.
Factories `pea_resnet18` (CIFAR / torchvision ResNet) and `pea_vit` (timm
ViT-Base) both require a one-off `precompute_source_stats(loader, device)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from .learner import register

__all__ = [
    'pea_resnet18', 'pea_vit',
    'PEA', 'PEAResNetWrapper', 'PEAViTWrapper',
    'compute_source_stats_resnet', 'compute_source_stats_vit',
    'PEAStatsResNet', 'PEAStatsViT',
]


# Linear algebra helpers (ported from PEA_TTA/pea_{resnet,vit}.py)

@torch.no_grad()
def matrix_sqrt_fp32(M: torch.Tensor) -> torch.Tensor:
    """Symmetric matrix square root in fp32 via eigendecomposition."""
    M = M.float()
    M = 0.5 * (M + M.T)
    L, V = torch.linalg.eigh(M)
    L = torch.clamp(L, min=0.0)
    return V @ torch.diag(torch.sqrt(L)) @ V.T


@torch.no_grad()
def invsqrt_spd(A: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Return A^{-1/2} for symmetric positive (semi)definite A in fp32."""
    A = A.float()
    A = 0.5 * (A + A.T)
    L, V = torch.linalg.eigh(A)
    L = torch.clamp(L, min=eps)
    Linv2 = L.rsqrt()
    return (V * Linv2.unsqueeze(0)) @ V.T


def _group_slices(D: int, group_size: int) -> List[Tuple[int, int]]:
    return [(g, min(g + group_size, D)) for g in range(0, D, group_size)]


# Backbone shim: unifies stem, pooling and head across timm, torchvision and
# CIFAR ResNets so the per-block alignment loop is shared.

class _ResNetShim:
    """Detect which ResNet variant is wrapped, and provide a unified stem,
    pooling, and head forward."""

    def __init__(self, base_model: nn.Module):
        self.base = base_model
        self.has_maxpool = hasattr(base_model, 'maxpool') and not isinstance(
            getattr(base_model, 'maxpool', None), nn.Identity
        ) and getattr(base_model, 'maxpool', None) is not None
        # Activation: timm uses .act1, torchvision uses .relu, CIFAR uses .relu_conv
        if hasattr(base_model, 'act1'):
            self.stem_act = base_model.act1
        elif hasattr(base_model, 'relu'):
            self.stem_act = base_model.relu
        elif hasattr(base_model, 'relu_conv'):
            self.stem_act = base_model.relu_conv
        else:
            self.stem_act = F.relu
        # Pooling: timm has .global_pool, torchvision has .avgpool;
        # CIFAR ResNet_Custom uses F.avg_pool2d(out, 4) inline.
        if hasattr(base_model, 'global_pool') and not isinstance(
            base_model.global_pool, nn.Identity
        ):
            self._pool_kind = 'timm'
            self.pool = base_model.global_pool
        elif hasattr(base_model, 'avgpool'):
            self._pool_kind = 'torchvision'
            self.pool = base_model.avgpool
        else:
            self._pool_kind = 'cifar'
            self.pool = None
        # Head: timm + torchvision use .fc, the older CIFAR ResNet uses .linear
        if hasattr(base_model, 'fc'):
            self.head = base_model.fc
            self._head_name = 'fc'
        elif hasattr(base_model, 'linear'):
            self.head = base_model.linear
            self._head_name = 'linear'
        else:
            raise ValueError(
                "PEA-ResNet: model exposes no recognised classifier head "
                "(looked for .fc, .linear)."
            )

    def stem(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        if callable(self.stem_act):
            x = self.stem_act(x)
        if self.has_maxpool:
            x = self.base.maxpool(x)
        return x

    def pool_and_flatten(self, x: torch.Tensor) -> torch.Tensor:
        if self._pool_kind in ('timm', 'torchvision'):
            x = self.pool(x)
        else:  # CIFAR: hard-coded 4x4 avgpool in source forward
            x = F.avg_pool2d(x, 4)
        return torch.flatten(x, 1)


# Offline source statistics, ResNet

@torch.no_grad()
def compute_source_stats_resnet(
    model: nn.Module,
    clean_loader,
    device,
    *,
    layers: Tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4"),
    max_samples: int = 2000,
    reg: float = 1e-5,
) -> Tuple[
    Dict[Tuple[str, int], torch.Tensor],
    Dict[Tuple[str, int], torch.Tensor],
    Dict[Tuple[str, int], torch.Tensor],
]:
    """Streamed source-domain mean/var/cov^{1/2} per residual block."""
    model = model.to(device).eval()
    source_means: Dict[Tuple[str, int], torch.Tensor] = {}
    source_vars: Dict[Tuple[str, int], torch.Tensor] = {}
    source_cov_sqrts: Dict[Tuple[str, int], torch.Tensor] = {}

    for layer in layers:
        seq = getattr(model, layer)
        for idx in range(len(seq)):
            run_sum = None
            run_sqsum = None
            pix_count = 0
            cov_accum = None
            n_accum = 0
            C_cached = None

            def hook_fn(_m, _inp, out):
                nonlocal run_sum, run_sqsum, pix_count, cov_accum, n_accum, C_cached
                feat = out.detach()
                B, C, H, W = feat.shape
                C_cached = C
                if run_sum is None:
                    run_sum = torch.zeros((1, C, 1, 1), device=feat.device, dtype=feat.dtype)
                    run_sqsum = torch.zeros((1, C, 1, 1), device=feat.device, dtype=feat.dtype)
                run_sum += feat.sum(dim=(0, 2, 3), keepdim=True)
                run_sqsum += (feat * feat).sum(dim=(0, 2, 3), keepdim=True)
                pix_count += B * H * W

                X = feat.permute(0, 2, 3, 1).reshape(-1, C).float()
                mu_b = X.mean(dim=0, keepdim=True)
                Xc = X - mu_b
                cov_b = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
                cov_accum = cov_b if cov_accum is None else (cov_accum + cov_b)
                n_accum += 1

            handle = seq[idx].register_forward_hook(hook_fn)

            seen = 0
            for batch in clean_loader:
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                _ = model(imgs.to(device, non_blocking=True))
                seen += imgs.size(0)
                if seen >= max_samples:
                    break

            handle.remove()

            if C_cached is None:
                # Should not happen unless the loader is empty
                raise RuntimeError(
                    f"PEA source-stats: no batches seen while hooking {layer}.{idx}"
                )
            mu = run_sum / max(1, pix_count)
            sq_mean = run_sqsum / max(1, pix_count)
            var = torch.clamp(sq_mean - mu * mu, min=0.0)

            cov = (cov_accum / max(1, n_accum)).to(device)
            cov = cov + reg * torch.eye(C_cached, device=device, dtype=cov.dtype)
            cov_sqrt = matrix_sqrt_fp32(cov)

            key = (layer, idx)
            source_means[key] = mu
            source_vars[key] = var
            source_cov_sqrts[key] = cov_sqrt

    return source_means, source_vars, source_cov_sqrts


# Offline source statistics, ViT

@torch.no_grad()
def compute_source_stats_vit(
    model: nn.Module,
    clean_loader,
    device,
    *,
    use_cls_only: bool = False,
    max_batches: Optional[int] = None,
    reg: float = 1e-5,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Streamed source-domain mean/var/cov^{1/2} per transformer block."""
    model = model.to(device).eval()
    source_means: Dict[str, torch.Tensor] = {}
    source_vars: Dict[str, torch.Tensor] = {}
    source_cov_sqrts: Dict[str, torch.Tensor] = {}

    if not hasattr(model, 'blocks'):
        raise ValueError(
            "PEA-ViT: model has no .blocks attribute (expected timm "
            "VisionTransformer)."
        )

    num_blocks = len(model.blocks)
    for idx in range(num_blocks):
        running_mean = None
        running_m2 = None
        running_c2 = None
        n_samples = 0
        feature_dim = 0
        captured = {'X': None}

        def hook_fn(_m, _inp, out):
            feat = out.detach().float()
            X = feat[:, 0, :] if use_cls_only else feat.reshape(-1, feat.shape[-1])
            captured['X'] = X

        h = model.blocks[idx].register_forward_hook(hook_fn)

        for bi, batch in enumerate(clean_loader):
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            _ = model(imgs.to(device, non_blocking=True).float())
            X = captured['X']
            if X is None:
                raise RuntimeError("PEA-ViT hook did not capture features.")
            N, D = X.shape
            if feature_dim == 0:
                feature_dim = D
                running_mean = torch.zeros(D, device=device)
                running_m2 = torch.zeros(D, device=device)
                running_c2 = torch.zeros(D, D, device=device)

            n_total = n_samples + N
            delta = X.mean(0) - running_mean
            running_mean += delta * (N / max(n_total, 1))
            Xc_prev = X - (running_mean - delta)
            Xc_new = X - running_mean
            running_m2 += (Xc_prev * Xc_new).sum(0)

            Xc_batch = X - X.mean(0, keepdim=True)
            running_c2 += Xc_batch.T @ Xc_batch

            n_samples = n_total
            if max_batches is not None and (bi + 1) >= max_batches:
                break

        h.remove()

        key = f"block{idx}"
        if n_samples > 1:
            mu = running_mean.unsqueeze(0)
            var = (running_m2 / (n_samples - 1)).unsqueeze(0)
            cov = running_c2 / (n_samples - 1)
        else:
            mu = torch.zeros(1, feature_dim, device=device)
            var = torch.zeros_like(mu)
            cov = torch.zeros(feature_dim, feature_dim, device=device)

        cov = cov + reg * torch.eye(feature_dim, device=device)
        source_means[key] = mu
        source_vars[key] = var
        source_cov_sqrts[key] = matrix_sqrt_fp32(cov)

    return source_means, source_vars, source_cov_sqrts


# EMA target-domain stats containers

@dataclass(frozen=True)
class _PEAStatsCfg:
    momentum: float = 0.05
    alpha: float = 0.1
    gamma: float = 2.5
    entropy_threshold: float = 0.5
    reg: float = 1e-5


class PEAStatsResNet:
    """EMA mean + covariance (per residual block) with drift tracking."""
    def __init__(self, *, momentum=0.05, alpha=0.1, gamma=2.5,
                 entropy_threshold=0.5, reg=1e-5):
        self.cfg = _PEAStatsCfg(momentum, alpha, gamma, entropy_threshold, reg)
        self.mu: Dict[Tuple[str, int], torch.Tensor] = {}
        self.cov: Dict[Tuple[str, int], torch.Tensor] = {}
        self.drift_smooth: Dict[Tuple[str, int], float] = {}
        self.entropy_smooth: Optional[float] = None

    @torch.no_grad()
    def reset(self) -> 'PEAStatsResNet':
        return PEAStatsResNet(**self.cfg.__dict__)

    @torch.no_grad()
    def update_from_feat4d(self, key, feat: torch.Tensor, *, tile_h: int = 8) -> bool:
        feat = feat.float()
        B, C, H, W = feat.shape
        device = feat.device
        N = B * H * W
        sum_x = torch.zeros(1, C, device=device, dtype=torch.float32)
        sum_xx = torch.zeros(C, C, device=device, dtype=torch.float32)
        for h0 in range(0, H, tile_h):
            sl = feat[:, :, h0:h0 + tile_h, :]
            X = sl.permute(0, 2, 3, 1).reshape(-1, C)
            sum_x += X.sum(0, keepdim=True)
            sum_xx += X.T @ X

        mu_b = sum_x / max(1, N)
        cov_b = (sum_xx - N * (mu_b.T @ mu_b)) / max(1, N - 1)
        cov_b = cov_b + self.cfg.reg * torch.eye(C, device=device, dtype=torch.float32)

        if key not in self.mu:
            self.mu[key] = mu_b.detach()
            self.cov[key] = cov_b.detach()
            self.drift_smooth[key] = 0.0
            return False

        mu_ema = self.mu[key]
        drift = (mu_b - mu_ema).abs().mean().item()
        prev = self.drift_smooth.get(key, drift)
        smoothed = self.cfg.alpha * drift + (1.0 - self.cfg.alpha) * prev
        self.drift_smooth[key] = smoothed

        spike = (smoothed > 0.0) and (drift > self.cfg.gamma * smoothed)
        if spike:
            self.mu[key] = mu_b.detach()
            self.cov[key] = cov_b.detach()
        else:
            m = self.cfg.momentum
            self.mu[key] = (m * mu_b + (1.0 - m) * mu_ema).detach()
            self.cov[key] = (m * cov_b + (1.0 - m) * self.cov[key]).detach()
        return spike

    @torch.no_grad()
    def update_entropy(self, e: float) -> None:
        a = self.cfg.alpha
        self.entropy_smooth = e if self.entropy_smooth is None else (
            a * e + (1.0 - a) * self.entropy_smooth
        )

    @torch.no_grad()
    def entropy_spike(self, e: float) -> bool:
        return (self.entropy_smooth is not None) and (
            e > self.entropy_smooth + self.cfg.entropy_threshold
        )


class PEAStatsViT:
    """EMA mean + covariance (per transformer block) with drift tracking."""
    def __init__(self, *, momentum=0.02, alpha=0.2, gamma=4.0,
                 entropy_threshold=1.0, reg=1e-5):
        self.cfg = _PEAStatsCfg(momentum, alpha, gamma, entropy_threshold, reg)
        self.mu: Dict[str, torch.Tensor] = {}
        self.cov: Dict[str, torch.Tensor] = {}
        self.drift_smooth: Dict[str, float] = {}
        self.entropy_smooth: Optional[float] = None

    @torch.no_grad()
    def reset(self) -> 'PEAStatsViT':
        return PEAStatsViT(**self.cfg.__dict__)

    @torch.no_grad()
    def _cov_from_tokens(self, X: torch.Tensor):
        mu = X.mean(0, keepdim=True)
        Xc = X - mu
        cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
        cov.view(-1)[:: cov.shape[0] + 1] += self.cfg.reg
        return mu, cov

    @torch.no_grad()
    def update(self, key: str, X: torch.Tensor) -> bool:
        mu_b, cov_b = self._cov_from_tokens(X)
        if key not in self.mu:
            self.mu[key] = mu_b.detach()
            self.cov[key] = cov_b.detach()
            self.drift_smooth[key] = 0.0
            return False
        mu_ema = self.mu[key]
        drift = (mu_b - mu_ema).abs().mean().item()
        prev = self.drift_smooth.get(key, drift)
        smoothed = self.cfg.alpha * drift + (1.0 - self.cfg.alpha) * prev
        self.drift_smooth[key] = smoothed
        spike = (smoothed > 1e-6) and (drift > self.cfg.gamma * smoothed)
        if spike:
            self.mu[key] = mu_b.detach()
            self.cov[key] = cov_b.detach()
        else:
            m = self.cfg.momentum
            self.mu[key] = (m * mu_b + (1.0 - m) * mu_ema).detach()
            self.cov[key] = (m * cov_b + (1.0 - m) * self.cov[key]).detach()
        return spike

    @torch.no_grad()
    def update_entropy(self, e: float) -> None:
        a = self.cfg.alpha
        self.entropy_smooth = e if self.entropy_smooth is None else (
            a * e + (1.0 - a) * self.entropy_smooth
        )

    @torch.no_grad()
    def entropy_spike(self, e: float) -> bool:
        return (self.entropy_smooth is not None) and (
            e > self.entropy_smooth + self.cfg.entropy_threshold
        )


# Geometry-only TTA helpers

def _tta_candidates(center_crop_ratio: float, do_hflip: bool) -> List[str]:
    have_crop = (center_crop_ratio is not None) and (center_crop_ratio < 1.0)
    names: List[str] = []
    if have_crop:
        names.append("centercrop")
    if do_hflip:
        names.append("hflip")
    if have_crop and do_hflip:
        names.append("centercrop_hflip")
    return names


def _select_tta_names(*, use_augmentation, n_aug_max, center_crop_ratio,
                     do_hflip, include_original) -> List[str]:
    names: List[str] = ["orig"] if include_original else []
    if use_augmentation and n_aug_max > 0:
        names += _tta_candidates(center_crop_ratio, do_hflip)[:n_aug_max]
    return names


@torch.no_grad()
def _apply_variant(imgs: torch.Tensor, name: str, center_crop_ratio: float) -> torch.Tensor:
    if name == "orig":
        return imgs
    x = imgs
    _, _, H, W = x.shape
    if name in ("centercrop", "centercrop_hflip"):
        ch = max(1, int(round(H * center_crop_ratio)))
        cw = max(1, int(round(W * center_crop_ratio)))
        x = TF.center_crop(x, [ch, cw])
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    if name in ("hflip", "centercrop_hflip"):
        x = torch.flip(x, dims=[3])
    return x


# Pass-1 block-weight reducers (per-block mu/var across all views)

class _BlockMVReducer:
    def __init__(self, C: int, device):
        self.sum = torch.zeros(1, C, 1, 1, device=device, dtype=torch.float32)
        self.sumsq = torch.zeros(1, C, 1, 1, device=device, dtype=torch.float32)
        self.count = 0

    @torch.no_grad()
    def update(self, feat: torch.Tensor) -> None:
        self.sum += feat.sum(dim=(0, 2, 3), keepdim=True)
        self.sumsq += (feat * feat).sum(dim=(0, 2, 3), keepdim=True)
        self.count += feat.shape[0] * feat.shape[2] * feat.shape[3]

    @torch.no_grad()
    def finalize(self):
        N = max(1, self.count)
        mu = self.sum / N
        var = (self.sumsq / N - mu * mu).clamp_min(0.0)
        var = var * (N / max(1, N - 1))
        return mu, var


class _TokenMVReducer:
    def __init__(self, D: int, device):
        self.sum = torch.zeros(1, D, device=device, dtype=torch.float32)
        self.sumsq = torch.zeros(1, D, device=device, dtype=torch.float32)
        self.count = 0

    @torch.no_grad()
    def update(self, X: torch.Tensor) -> None:
        self.sum += X.sum(0, keepdim=True)
        self.sumsq += (X * X).sum(0, keepdim=True)
        self.count += X.shape[0]

    @torch.no_grad()
    def finalize(self):
        T = max(1, self.count)
        mu = self.sum / T
        var = (self.sumsq / T - mu * mu).clamp_min(0.0)
        var = var * (T / max(1, T - 1))
        return mu, var


# PEA wrappers, one per architecture family

class PEA(nn.Module):
    """Base wrapper exposing `.model`, `.forward` and `.reset`; subclassed per architecture.

    `_pea_mode` controls the alignment hooks:
      "bare":  hooks are no-ops (pass-1, discrepancy measured on unaligned features).
      "adapt": hooks align and update the EMA target stats (pass-2).
      "eval":  hooks align without updating the EMA stats (external callers).
    `precompute_source_stats` sets the default mode to "eval".
    """
    def __init__(self):
        super().__init__()
        self._has_stats = False
        self._pea_mode = "bare"
        self._align_handles: List = []

    def precompute_source_stats(self, source_loader, device, **kwargs):
        raise NotImplementedError

    def remove_align_hooks(self):
        for h in self._align_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._align_handles = []

    def reset(self):
        if hasattr(self, 'stats') and self.stats is not None:
            self.stats = self.stats.reset()
        if hasattr(self, '_cache'):
            self._cache.clear()
        if hasattr(self, 'norm_weights'):
            self.norm_weights = {}
        if hasattr(self, '_step'):
            self._step = 0


# ResNet wrapper

class PEAResNetWrapper(PEA):
    """Progressive Embedding Alignment for ResNet (CIFAR + torchvision + timm).

    Source stats are computed lazily on the first call to
    `precompute_source_stats`; before that, forward() falls back to the
    bare-model output (so accuracy probes can still run pre-stats).
    """
    _SKIP_THRESHOLD = 1e-3
    _MV_SWITCH_THRESHOLD = 1e-3  # ~0 -> effectively CORAL for most blocks

    def __init__(self, model: nn.Module, *,
                 group_size: int = 32,
                 update_every: int = 1,
                 eps: float = 1e-5,
                 ema_tile_h: int = 8,
                 coral_tile_h: int = 8,
                 coral_tile_w: int = 64,
                 use_augmentation: bool = True,
                 n_aug_max: int = 2,
                 center_crop_ratio: float = 0.9,
                 do_hflip: bool = True,
                 include_original: bool = True,
                 microbatch_variants: int = 1,
                 reset_on_entropy_spike: bool = True,
                 stats_kwargs: Optional[dict] = None):
        super().__init__()
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        self.shim = _ResNetShim(self.model)
        self.group_size = int(group_size)
        self.update_every = int(update_every)
        self.eps = float(eps)
        self.ema_tile_h = int(ema_tile_h)
        self.coral_tile_h = int(coral_tile_h)
        self.coral_tile_w = int(coral_tile_w)
        self.use_augmentation = bool(use_augmentation)
        self.n_aug_max = int(n_aug_max)
        self.center_crop_ratio = float(center_crop_ratio)
        self.do_hflip = bool(do_hflip)
        self.include_original = bool(include_original)
        self.microbatch_variants = int(microbatch_variants)
        self.reset_on_entropy_spike = bool(reset_on_entropy_spike)
        self.stats = PEAStatsResNet(**(stats_kwargs or {}))

        self.source_means: Dict[Tuple[str, int], torch.Tensor] = {}
        self.source_vars: Dict[Tuple[str, int], torch.Tensor] = {}
        self.src_cov_groups: Dict[Tuple[str, int], List[torch.Tensor]] = {}
        self._src_dev_cache: Dict[Tuple[Tuple[str, int], torch.device], dict] = {}
        self.norm_weights: Dict[Tuple[str, int], float] = {}
        self._step = 0

    # Source-stats precomputation

    @torch.no_grad()
    def precompute_source_stats(self, source_loader, device,
                                max_samples: int = 2000,
                                max_batches: Optional[int] = None):
        # `max_batches` is converted to a sample count via the loader's batch_size.
        if max_batches is not None:
            bs = getattr(source_loader, 'batch_size', None) or 64
            max_samples = int(max_batches) * int(bs)
        means, vars_, cov_sqrts = compute_source_stats_resnet(
            self.model, source_loader, device, max_samples=max_samples
        )
        self.source_means = {k: v.float() for k, v in means.items()}
        self.source_vars = {k: v.float() for k, v in vars_.items()}
        # Pre-group Σ_src^{1/2}
        self.src_cov_groups = {}
        for key, cov_sqrt in cov_sqrts.items():
            cov_sqrt = cov_sqrt.float()
            cov = cov_sqrt @ cov_sqrt.T
            groups = []
            for s, e in _group_slices(cov.shape[0], self.group_size):
                groups.append(matrix_sqrt_fp32(cov[s:e, s:e].contiguous()))
            self.src_cov_groups[key] = groups
        self._has_stats = True
        # Hooks are live once source stats exist; "eval" aligns without updating the EMA stats.
        self._register_align_hooks()
        self._pea_mode = "eval"

    # Core helpers

    def _ensure_src_cached(self, key, device):
        tag = (key, device)
        if tag in self._src_dev_cache:
            return self._src_dev_cache[tag]
        mu = self.source_means[key].to(device, non_blocking=True).view(1, -1).contiguous()
        var = self.source_vars[key].to(device, non_blocking=True).view(1, -1).contiguous()
        cov_groups = [g.to(device, non_blocking=True) for g in self.src_cov_groups[key]]
        bundle = {"mu": mu, "var": var, "cov_groups": cov_groups}
        self._src_dev_cache[tag] = bundle
        return bundle

    def _register_align_hooks(self):
        """Register one mode-aware forward hook per residual block of the inner ResNet."""
        if self._align_handles:
            return  # idempotent

        def make_hook(key):
            def hook(_m, _i, out):
                if not self._has_stats or self._pea_mode == "bare":
                    return out
                update_stats = (self._pea_mode == "adapt")
                return self._align_block(out, key, update_stats=update_stats)
            return hook

        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            seq = getattr(self.model, layer_name)
            for idx, blk in enumerate(seq):
                h = blk.register_forward_hook(make_hook((layer_name, idx)))
                self._align_handles.append(h)

    @torch.no_grad()
    def _align_block(self, feat: torch.Tensor, key,
                     update_stats: bool = True) -> torch.Tensor:
        w = float(self.norm_weights.get(key, 1.0))
        if w < self._SKIP_THRESHOLD:
            return feat
        if update_stats:
            self.stats.update_from_feat4d(key, feat, tile_h=self.ema_tile_h)
        elif key not in self.stats.mu:
            # Eval mode with unseeded EMA stats for this block: nothing to align against.
            return feat

        B, C, H, W = feat.shape
        device = feat.device
        src = self._ensure_src_cached(key, device)
        mu_src = src["mu"]
        var_src = src["var"]

        # MV branch (threshold ~0, effectively never taken)
        if w < self._MV_SWITCH_THRESHOLD:
            mu_dom = self.stats.mu[key].to(device).view(1, -1, 1, 1)
            var_dom = torch.diag(self.stats.cov[key].to(device)).clamp_min(self.eps).view(1, -1, 1, 1)
            scale = torch.sqrt((var_src.view(1, -1, 1, 1) + self.eps) / (var_dom + self.eps))
            bias = mu_src.view(1, -1, 1, 1) - mu_dom * scale
            corr = feat * (scale - 1.0) + bias
            feat.add_(corr, alpha=w)
            return feat

        # CORAL branch
        cov_dom_full = self.stats.cov[key].to(device).float()
        mu_dom_full = self.stats.mu[key].to(device)
        th = self.coral_tile_h
        tw = self.coral_tile_w

        for (s, e), cov_src_g_sqrt in zip(
            _group_slices(C, self.group_size), src["cov_groups"]
        ):
            g = e - s
            Fg = feat[:, s:e, :, :]
            mu_s = mu_src[:, s:e]
            mu_d = mu_dom_full[:, s:e]
            cov_g = cov_dom_full[s:e, s:e].contiguous()
            inv_sqrt_g = invsqrt_spd(cov_g, eps=self.eps)

            max_N = B * min(th, H) * min(tw, W)
            tmp1 = torch.empty((max_N, g), device=device, dtype=feat.dtype)
            tmp2 = torch.empty((max_N, g), device=device, dtype=feat.dtype)

            for h0 in range(0, H, th):
                h1 = min(h0 + th, H)
                for w0 in range(0, W, tw):
                    w1 = min(w0 + tw, W)
                    sl = Fg[:, :, h0:h1, w0:w1]
                    X = sl.permute(0, 2, 3, 1).reshape(-1, g)
                    N = X.shape[0]
                    X.sub_(mu_d)
                    torch.mm(X, inv_sqrt_g, out=tmp1[:N, :])
                    torch.mm(tmp1[:N, :], cov_src_g_sqrt, out=tmp2[:N, :])
                    tmp2[:N, :].add_(mu_s)
                    Y = tmp2[:N, :].view(sl.shape[0], sl.shape[2], sl.shape[3], g)
                    Y = Y.permute(0, 3, 1, 2).contiguous()
                    sl.mul_(1.0 - w).add_(Y, alpha=w)
        return feat

    @torch.no_grad()
    def _forward_with_alignment(self, x: torch.Tensor) -> torch.Tensor:
        """Pass-2 forward with the alignment hooks in "adapt" mode."""
        self._step += 1
        x = x.float()
        prev_mode = self._pea_mode
        try:
            self._pea_mode = "adapt"
            return self.model(x)
        finally:
            self._pea_mode = prev_mode

    @torch.no_grad()
    def _forward_bare(self, x: torch.Tensor) -> torch.Tensor:
        """Used both as the pre-stats fallback and for pass-1 weight computation."""
        return self.model(x)

    # Pass-1 block weights

    @torch.no_grad()
    def _compute_block_weights(self, imgs: torch.Tensor) -> Dict[Tuple[str, int], float]:
        device = imgs.device
        names = _select_tta_names(
            use_augmentation=self.use_augmentation,
            n_aug_max=self.n_aug_max,
            center_crop_ratio=self.center_crop_ratio,
            do_hflip=self.do_hflip,
            include_original=self.include_original,
        )
        reducers: Dict[Tuple[str, int], _BlockMVReducer] = {}

        def hook_factory(key):
            def hook(_m, _i, out):
                feat = out.detach().float()
                C = feat.shape[1]
                if key not in reducers:
                    reducers[key] = _BlockMVReducer(C=C, device=feat.device)
                reducers[key].update(feat)
            return hook

        handles = []
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            seq = getattr(self.model, layer_name)
            for idx in range(len(seq)):
                handles.append(seq[idx].register_forward_hook(hook_factory((layer_name, idx))))

        # Pass-1 reads unaligned features ("bare" mode) so the per-block discrepancy is non-degenerate.
        prev_mode = self._pea_mode
        try:
            self._pea_mode = "bare"
            for name in names:
                xvar = _apply_variant(imgs, name, self.center_crop_ratio)
                _ = self.model(xvar)
        finally:
            self._pea_mode = prev_mode

        for h in handles:
            h.remove()

        keys: List[Tuple[str, int]] = []
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            seq = getattr(self.model, layer_name)
            for idx in range(len(seq)):
                keys.append((layer_name, idx))

        scores = torch.empty(len(keys), device=device, dtype=torch.float32)
        for i, key in enumerate(keys):
            mu_dom, var_dom = reducers[key].finalize()
            mu_src = self.source_means[key].to(device)
            var_src = self.source_vars[key].to(device)
            diff_mu = (mu_src - mu_dom).reshape(-1)
            diff_var = (var_src - var_dom).reshape(-1)
            scores[i] = (
                torch.linalg.vector_norm(diff_mu, ord=2) +
                torch.linalg.vector_norm(diff_var, ord=2)
            )

        vmin, vmax = scores.min(), scores.max()
        if float(vmax) > float(vmin):
            normed = (scores - vmin) / (vmax - vmin)
        else:
            normed = torch.zeros_like(scores)
        return {k: float(normed[i]) for i, k in enumerate(keys)}

    # Public forward (two-pass)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Before precompute_source_stats the wrapper behaves as the bare model.
        if not self._has_stats:
            return self._forward_bare(x)

        # Pass-1: per-block discrepancy to normalised weights
        self.norm_weights = self._compute_block_weights(x)

        # Pass-2: averaged logits over geometry TTA variants
        names = _select_tta_names(
            use_augmentation=self.use_augmentation,
            n_aug_max=self.n_aug_max,
            center_crop_ratio=self.center_crop_ratio,
            do_hflip=self.do_hflip,
            include_original=self.include_original,
        )
        K = len(names)
        B = x.shape[0]
        mb = max(1, int(self.microbatch_variants))

        logits_sum: Optional[torch.Tensor] = None
        processed = 0
        for start in range(0, K, mb):
            chunk = names[start:start + mb]
            x_list = [_apply_variant(x, n, self.center_crop_ratio) for n in chunk]
            x_batch = torch.cat(x_list, dim=0)
            logits_chunk = self._forward_with_alignment(x_batch)
            logits_chunk = logits_chunk.view(len(chunk), B, -1)
            chunk_sum = logits_chunk.sum(dim=0)
            logits_sum = chunk_sum if logits_sum is None else (logits_sum + chunk_sum)
            processed += len(chunk)

        logits = logits_sum / max(1, processed)

        # An entropy spike resets the target stats
        if self.reset_on_entropy_spike:
            with torch.no_grad():
                p = torch.softmax(logits.float(), dim=1)
                e = float(-(p * p.clamp_min(1e-12).log()).sum(dim=1).mean())
            if self.stats.entropy_spike(e):
                self.stats = self.stats.reset()
                self._src_dev_cache.clear()
            self.stats.update_entropy(e)

        return logits

    def predict(self, x):
        return self.forward(x)


# ViT wrapper

class PEAViTWrapper(PEA):
    """Progressive Embedding Alignment for timm ViT-Base."""
    _SKIP_THRESHOLD = 1e-3
    _MV_SWITCH_THRESHOLD = 1e-3

    def __init__(self, model: nn.Module, *,
                 use_cls_only: bool = False,
                 group_size: int = 768,
                 update_every: int = 1,
                 eps: float = 1e-5,
                 use_augmentation: bool = True,
                 n_aug_max: int = 2,
                 center_crop_ratio: float = 0.9,
                 do_hflip: bool = True,
                 include_original: bool = True,
                 microbatch_variants: int = 1,
                 weight_use_cls_only: bool = True,
                 reset_on_entropy_spike: bool = True,
                 stats_kwargs: Optional[dict] = None,
                 imagenet_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        self.use_cls_only = bool(use_cls_only)
        self.group_size = int(group_size)
        self.update_every = int(update_every)
        self.eps = float(eps)
        self.use_augmentation = bool(use_augmentation)
        self.n_aug_max = int(n_aug_max)
        self.center_crop_ratio = float(center_crop_ratio)
        self.do_hflip = bool(do_hflip)
        self.include_original = bool(include_original)
        self.microbatch_variants = int(microbatch_variants)
        self.weight_use_cls_only = bool(weight_use_cls_only)
        self.reset_on_entropy_spike = bool(reset_on_entropy_spike)
        self.imagenet_mask = imagenet_mask
        self.stats = PEAStatsViT(**(stats_kwargs or {}))

        self.source_means: Dict[str, torch.Tensor] = {}
        self.source_vars: Dict[str, torch.Tensor] = {}
        self.src_cov_groups: Dict[str, List[torch.Tensor]] = {}
        self.norm_weights: Dict[str, float] = {}
        self._cache: Dict[str, dict] = {}
        self._step = 0

    @torch.no_grad()
    def precompute_source_stats(self, source_loader, device,
                                max_batches: Optional[int] = None):
        means, vars_, cov_sqrts = compute_source_stats_vit(
            self.model, source_loader, device,
            use_cls_only=self.use_cls_only, max_batches=max_batches
        )
        self.source_means = {k: v.float() for k, v in means.items()}
        self.source_vars = {k: v.float() for k, v in vars_.items()}
        groups: Dict[str, List[torch.Tensor]] = {}
        for key, cov_sqrt in cov_sqrts.items():
            cov = cov_sqrt.float() @ cov_sqrt.float().T
            groups[key] = [
                matrix_sqrt_fp32(cov[s:e, s:e].contiguous())
                for s, e in _group_slices(cov.shape[0], self.group_size)
            ]
        self.src_cov_groups = groups
        self._has_stats = True
        # Hooks are live once source stats exist; "eval" aligns without updating the EMA stats.
        self._register_align_hooks()
        self._pea_mode = "eval"

    @torch.no_grad()
    def _tokens(self, feat: torch.Tensor) -> torch.Tensor:
        return feat[:, 0, :] if self.use_cls_only else feat.reshape(-1, feat.shape[-1])

    def _register_align_hooks(self):
        """Register one mode-aware forward hook per transformer block of the inner ViT."""
        if self._align_handles:
            return  # idempotent

        def make_hook(key):
            def hook(_m, _i, out):
                if not self._has_stats or self._pea_mode == "bare":
                    return out
                update_stats = (self._pea_mode == "adapt")
                return self._align_block(out, key, update_stats=update_stats)
            return hook

        for i, blk in enumerate(self.model.blocks):
            h = blk.register_forward_hook(make_hook(f"block{i}"))
            self._align_handles.append(h)

    @torch.no_grad()
    def _align_block(self, feat: torch.Tensor, key: str,
                     update_stats: bool = True) -> torch.Tensor:
        w = float(self.norm_weights.get(key, 1.0))
        if w < self._SKIP_THRESHOLD:
            return feat
        X = self._tokens(feat)
        if update_stats:
            spike = self.stats.update(key, X)
        elif key not in self.stats.mu:
            # Eval mode with unseeded EMA stats for this block: nothing to align against.
            return feat
        else:
            # "eval" mode: do not drift EMA target stats but still consult
            # the running target mean / covariance for the alignment.
            spike = False
        mu_src = self.source_means[key].to(feat.device)
        var_src = self.source_vars[key].to(feat.device)
        mu_dom = self.stats.mu[key].to(feat.device)

        if w < self._MV_SWITCH_THRESHOLD:
            var_dom = torch.diag(self.stats.cov[key].to(feat.device)).view(1, -1)
            scale = torch.sqrt((var_src + self.eps) / var_dom.clamp_min(self.eps))
            Y = (X - mu_dom) * scale + mu_src
        else:
            if spike or (key not in self._cache) or ((self._step % self.update_every) == 0):
                inv_sqrts: List[torch.Tensor] = []
                cov_dom = self.stats.cov[key].to(feat.device).float()
                D = cov_dom.shape[0]
                for s, e in _group_slices(D, self.group_size):
                    inv_sqrts.append(invsqrt_spd(cov_dom[s:e, s:e].contiguous(), eps=self.eps))
                self._cache[key] = {"inv_sqrts": inv_sqrts, "mu_dom": mu_dom}
            Xc = X - self._cache[key]["mu_dom"]
            Y = torch.empty_like(X)
            for (s, e), inv_sqrt_g, cov_src_g_sqrt in zip(
                _group_slices(Xc.shape[-1], self.group_size),
                self._cache[key]["inv_sqrts"],
                self.src_cov_groups[key],
            ):
                Y[:, s:e] = Xc[:, s:e] @ inv_sqrt_g @ cov_src_g_sqrt.to(feat.device)
            Y.add_(mu_src)

        correction = w * (Y - X)
        if self.use_cls_only:
            feat[:, 0, :].add_(correction)
        else:
            feat.add_(correction.view_as(feat))
        return feat

    @torch.no_grad()
    def _forward_with_alignment(self, x: torch.Tensor) -> torch.Tensor:
        """Pass-2 forward with the alignment hooks in "adapt" mode."""
        self._step += 1
        x = x.float()
        prev_mode = self._pea_mode
        try:
            self._pea_mode = "adapt"
            return self.model(x)
        finally:
            self._pea_mode = prev_mode

    @torch.no_grad()
    def _compute_block_weights(self, imgs: torch.Tensor) -> Dict[str, float]:
        device = imgs.device
        names = _select_tta_names(
            use_augmentation=self.use_augmentation,
            n_aug_max=self.n_aug_max,
            center_crop_ratio=self.center_crop_ratio,
            do_hflip=self.do_hflip,
            include_original=self.include_original,
        )
        reducers: Dict[str, _TokenMVReducer] = {}

        def hook_factory(key):
            def hook(_m, _i, out):
                feat = out.detach().float()
                X = feat[:, 0, :] if self.weight_use_cls_only else feat.reshape(-1, feat.shape[-1])
                if key not in reducers:
                    reducers[key] = _TokenMVReducer(D=X.shape[1], device=X.device)
                reducers[key].update(X)
            return hook

        handles = [blk.register_forward_hook(hook_factory(f"block{i}"))
                   for i, blk in enumerate(self.model.blocks)]
        # Pass-1 reads unaligned features ("bare" mode) so the per-block discrepancy is non-degenerate.
        prev_mode = self._pea_mode
        try:
            self._pea_mode = "bare"
            for name in names:
                xvar = _apply_variant(imgs, name, self.center_crop_ratio)
                _ = self.model(xvar)
        finally:
            self._pea_mode = prev_mode
        for h in handles:
            h.remove()

        keys: List[str] = [f"block{i}" for i in range(len(self.model.blocks))]
        scores = torch.empty(len(keys), device=device, dtype=torch.float32)
        for i, key in enumerate(keys):
            mu_dom, var_dom = reducers[key].finalize()
            mu_src = self.source_means[key].to(device)
            var_src = self.source_vars[key].to(device)
            diff_mu = (mu_src - mu_dom).reshape(-1)
            diff_var = (var_src - var_dom).reshape(-1)
            scores[i] = (
                torch.linalg.vector_norm(diff_mu, ord=2) +
                torch.linalg.vector_norm(diff_var, ord=2)
            )
        vmin, vmax = scores.min(), scores.max()
        normed = (scores - vmin) / (vmax - vmin) if float(vmax) > float(vmin) else torch.zeros_like(scores)
        return {k: float(normed[i]) for i, k in enumerate(keys)}

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._has_stats:
            out = self.model(x)
            if self.imagenet_mask is not None:
                out = out[:, self.imagenet_mask]
            return out

        self.norm_weights = self._compute_block_weights(x)
        names = _select_tta_names(
            use_augmentation=self.use_augmentation,
            n_aug_max=self.n_aug_max,
            center_crop_ratio=self.center_crop_ratio,
            do_hflip=self.do_hflip,
            include_original=self.include_original,
        )
        K = len(names)
        B = x.shape[0]
        mb = max(1, int(self.microbatch_variants))
        logits_sum: Optional[torch.Tensor] = None
        processed = 0
        for start in range(0, K, mb):
            chunk = names[start:start + mb]
            x_list = [_apply_variant(x, n, self.center_crop_ratio) for n in chunk]
            x_batch = torch.cat(x_list, dim=0)
            logits_chunk = self._forward_with_alignment(x_batch)
            logits_chunk = logits_chunk.view(len(chunk), B, -1)
            chunk_sum = logits_chunk.sum(dim=0)
            logits_sum = chunk_sum if logits_sum is None else (logits_sum + chunk_sum)
            processed += len(chunk)
        logits = logits_sum / max(1, processed)
        if self.imagenet_mask is not None:
            logits = logits[:, self.imagenet_mask]

        if self.reset_on_entropy_spike:
            with torch.no_grad():
                p = torch.softmax(logits.float(), dim=1)
                e = float(-(p * p.clamp_min(1e-12).log()).sum(dim=1).mean())
            if self.stats.entropy_spike(e):
                self.stats = self.stats.reset()
                self._cache.clear()
            self.stats.update_entropy(e)

        return logits

    def predict(self, x):
        return self.forward(x)


# Registry factories

@register('pea_resnet18')
def pea_resnet18(model, optimizer=None, **kwargs):
    """Factory for PEA on ResNet-18 (BP-free).

    Args:
        model: ResNet-18 (CIFAR `ResNet_Custom`/`ResNet` or torchvision).
        optimizer: ignored (PEA is BP-free).
        **kwargs: forwarded to `PEAResNetWrapper.__init__`.
    """
    return PEAResNetWrapper(model=model, **kwargs)


@register('pea_vit')
def pea_vit(model, optimizer=None, **kwargs):
    """Factory for PEA on timm ViT-Base (BP-free)."""
    return PEAViTWrapper(model=model, **kwargs)


def configure_model(model):
    """PEA requires no parameter mutation; freeze everything in eval mode."""
    model.eval()
    model.requires_grad_(False)
    return model


def check_model(model):
    has_blocks = hasattr(model, 'blocks')
    has_resnet_layers = all(hasattr(model, f'layer{i}') for i in range(1, 5))
    assert has_blocks or has_resnet_layers, (
        "PEA needs either timm-style .blocks (ViT) or .layer1..4 (ResNet)."
    )
