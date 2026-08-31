"""Matrix-based entropy of a normalized positive semi-definite matrix.

Implements the matrix-based Renyi entropy of order alpha (Giraldo et al., 2015),
used in the activation-map entropy of the paper following the definition of
Skean et al. (2025): with lambda_i the eigenvalues of a trace-one PSD matrix A,
H_alpha(A) = log(sum_i lambda_i^alpha) / (1 - alpha), and H_1(A) = -sum_i lambda_i log lambda_i.
"""
import torch


def matrixAlphaEntropy(A, alpha=1.0):
    """Return the matrix-based Renyi entropy of order `alpha` of the PSD matrix `A` (natural log).

    Eigenvalues are clamped at 1e-12 and renormalized to sum to one so that small
    negative values produced by finite-precision eigendecompositions are ignored.
    """
    eig_vals = torch.linalg.eigvalsh(A)
    eig_vals = torch.clamp(eig_vals, min=1e-12)
    eig_vals = eig_vals / eig_vals.sum()
    if abs(alpha - 1.0) < 1e-6:
        return -torch.sum(eig_vals * torch.log(eig_vals + 1e-12))
    return 1.0 / (1.0 - alpha) * torch.log(torch.sum(eig_vals ** alpha) + 1e-12)
