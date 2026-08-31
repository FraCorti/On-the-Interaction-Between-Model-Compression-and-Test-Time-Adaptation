"""
LAME, Laplacian Adjusted Maximum-likelihood Estimation (Boudiaf et al., CVPR 2022).
Adapted from the official code: https://github.com/fiveai/LAME

Backpropagation-free TTA: refines per-sample class assignments through a
Laplacian-regularized maximum-likelihood objective on a kNN graph over the
L2-normalized penultimate features of the test batch. The last nn.Linear is
detected automatically and a forward pre-hook captures its input.
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .learner import register

__all__ = ['lame', 'LAME']


@register('lame')
def lame(model, optimizer=None, knn=5, sigma=1.0, affinity='kNN',
         force_symmetry=True, max_steps=100, bound_lambda=1.0,
         feature_module_name=None, imagenet_mask=None, **kwargs):
    """Factory for LAME (BP-free).

    Args:
        model: pretrained classifier.
        optimizer: ignored (LAME is BP-free).
        knn: number of nearest neighbours for the kNN affinity (default 5,
            matches Boudiaf's official `configs/method/best/lame.yaml`).
        sigma: RBF bandwidth (used only for `affinity='rbf'`; overridden in
            place by the mean k-th distance per Boudiaf's implementation).
        affinity: one of {'kNN', 'rbf', 'linear'}.
        force_symmetry: symmetrize the affinity as (W + W^T) / 2.
        max_steps: cap on the fixed-point iterations (typically converges <10).
        bound_lambda: weight λ of the Laplacian pairwise term.
        feature_module_name: dotted name of the module whose INPUT will be
            captured as penultimate features. If None, LAME auto-detects the
            last nn.Linear in the model.
        imagenet_mask: optional 1k-of-K logit slice (ImageNet-R / -A subsets).
    """
    return LAME(model=model, knn=knn, sigma=sigma, affinity=affinity,
                force_symmetry=force_symmetry, max_steps=max_steps,
                bound_lambda=bound_lambda,
                feature_module_name=feature_module_name,
                imagenet_mask=imagenet_mask)


class AffinityMatrix:
    def __call__(self, X):
        raise NotImplementedError


class kNN_affinity(AffinityMatrix):
    def __init__(self, knn, **kwargs):
        self.knn = knn

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.knn + 1, N)
        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]
        W = torch.zeros(N, N, device=X.device, dtype=X.dtype)
        W.scatter_(dim=-1, index=knn_index, value=1.0)
        return W


class rbf_affinity(AffinityMatrix):
    def __init__(self, sigma, knn, **kwargs):
        self.sigma = sigma
        self.k = knn

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)
        n_neighbors = min(self.k, N)
        kth_dist = dist.topk(k=n_neighbors, dim=-1, largest=False).values[:, -1]
        sigma = kth_dist.mean()
        return torch.exp(- dist ** 2 / (2 * sigma ** 2))


class linear_affinity(AffinityMatrix):
    def __call__(self, X):
        return torch.matmul(X, X.t())


def _entropy_energy(Y, unary, pairwise, bound_lambda):
    return (unary * Y - bound_lambda * pairwise * Y +
            Y * torch.log(Y.clip(1e-20))).sum()


def laplacian_optimization(unary, kernel, bound_lambda=1.0, max_steps=100,
                           tol=1e-8):
    """Closed-form fixed-point optimizer (Boudiaf et al. 2022, Eq. 4)."""
    oldE = float('inf')
    Y = (-unary).softmax(-1)
    for i in range(max_steps):
        pairwise = bound_lambda * kernel.matmul(Y)
        Y = (-unary + pairwise).softmax(-1)
        E = _entropy_energy(Y, unary, pairwise, bound_lambda).item()
        if i > 1 and abs(E - oldE) <= tol * abs(oldE):
            break
        oldE = E
    return Y


def _find_final_linear(model):
    """Return the dotted name of the last nn.Linear module reachable.

    Walks model.named_modules() in registration order; the last Linear is
    the classifier head for all the architectures used in this project
    (ResNet_Custom.fc, ResNet.linear, torchvision resnet18.fc, timm
    VisionTransformer.head).
    """
    last_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
    if last_name is None:
        raise ValueError(
            "LAME could not find an nn.Linear classifier head in the model. "
            "Pass feature_module_name explicitly to .lame(...)."
        )
    return last_name


def _resolve_module(model, dotted_name):
    mod = model
    for part in dotted_name.split('.'):
        mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
    return mod


class LAME(nn.Module):
    """LAME wrapper. Forward returns class probabilities (not logits); argmax
    over the last dim still yields the predicted class, as in the official
    `format_result` convention.
    """

    def __init__(self, model, knn=5, sigma=1.0, affinity='kNN',
                 force_symmetry=True, max_steps=100, bound_lambda=1.0,
                 feature_module_name=None, imagenet_mask=None):
        super().__init__()
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)

        self.knn = int(knn)
        self.sigma = float(sigma)
        self.affinity_name = affinity
        self.force_symmetry = bool(force_symmetry)
        self.max_steps = int(max_steps)
        self.bound_lambda = float(bound_lambda)
        self.imagenet_mask = imagenet_mask

        if affinity == 'kNN':
            self.affinity = kNN_affinity(knn=self.knn)
        elif affinity == 'rbf':
            self.affinity = rbf_affinity(sigma=self.sigma, knn=self.knn)
        elif affinity == 'linear':
            self.affinity = linear_affinity()
        else:
            raise ValueError(f"Unknown affinity '{affinity}'. "
                             "Use one of: kNN, rbf, linear.")

        self._feature_module_name = (
            feature_module_name if feature_module_name is not None
            else _find_final_linear(self.model)
        )
        target = _resolve_module(self.model, self._feature_module_name)
        self._captured_features = None
        self._hook_handle = target.register_forward_pre_hook(self._capture_hook)

        self.model_state = deepcopy(self.model.state_dict())

    def _capture_hook(self, module, inputs):
        feat = inputs[0]
        if feat.dim() > 2:
            feat = feat.flatten(1)
        self._captured_features = feat.detach()
        return None

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self._captured_features = None

    def predict(self, x):
        return self.forward(x)

    @torch.no_grad()
    def forward(self, x):
        self._captured_features = None
        logits = self.model(x)
        if self.imagenet_mask is not None:
            logits = logits[:, self.imagenet_mask]

        probas = logits.softmax(dim=-1)
        unary = -torch.log(probas + 1e-10)

        feats = self._captured_features
        if feats is None:
            raise RuntimeError(
                f"LAME: forward pre-hook on '{self._feature_module_name}' "
                f"did not fire. Ensure the captured module is reachable on "
                f"the forward path."
            )
        feats = F.normalize(feats.to(logits.device, dtype=logits.dtype),
                            p=2, dim=-1)
        kernel = self.affinity(feats)
        if self.force_symmetry:
            kernel = 0.5 * (kernel + kernel.t())

        Y = laplacian_optimization(
            unary=unary, kernel=kernel,
            bound_lambda=self.bound_lambda, max_steps=self.max_steps,
        )
        return Y

    def __del__(self):
        try:
            self._hook_handle.remove()
        except Exception:
            pass


def configure_model(model):
    """LAME requires no parameter mutation; freeze everything in eval mode."""
    model.eval()
    model.requires_grad_(False)
    return model


def check_model(model):
    """Confirm there is at least one nn.Linear that LAME can hook."""
    has_linear = any(isinstance(m, nn.Linear) for m in model.modules())
    assert has_linear, "LAME needs an nn.Linear classifier head to hook"
