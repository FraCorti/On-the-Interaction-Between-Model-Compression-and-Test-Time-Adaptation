"""
Multi-checkpoint aggregation utilities.

Aggregates per-corruption, per-ratio accuracy results across multiple
checkpoints (different initializations) and computes mean ± std statistics.
"""

import copy
import numpy as np


def parse_checkpoint_paths(args):
    """Return list of checkpoint paths from --checkpoints or --checkpoint."""
    if hasattr(args, 'checkpoints') and args.checkpoints:
        return [p.strip() for p in args.checkpoints.split(',') if p.strip()]
    return [args.checkpoint]


def extract_accuracy_matrix(acc_dict, compression_ratios, corruptions):
    """Convert nested dict {ratio: [corr_lists]} to numpy array.

    Returns:
        np.ndarray of shape (n_ratios, n_corruptions) where each entry
        is the mean accuracy across the inner list (usually length 1).
    """
    n_ratios = len(compression_ratios)
    n_corr = len(corruptions)
    matrix = np.full((n_ratios, n_corr), np.nan)
    for i, ratio in enumerate(compression_ratios):
        for j in range(n_corr):
            vals = acc_dict[ratio][j]
            if vals:
                matrix[i, j] = np.mean(vals)
    return matrix


def aggregate_accuracy_matrices(all_matrices):
    """Aggregate a list of (n_ratios, n_corruptions) matrices.

    Args:
        all_matrices: list of np.ndarray, one per checkpoint.

    Returns:
        mean_matrix, std_matrix: both (n_ratios, n_corruptions).
    """
    stacked = np.stack(all_matrices, axis=0)  # (n_ckpt, n_ratios, n_corr)
    mean_matrix = np.nanmean(stacked, axis=0)
    std_matrix = np.nanstd(stacked, axis=0, ddof=0)
    return mean_matrix, std_matrix


def aggregate_corruption_averaged(all_matrices):
    """Aggregate corruption-averaged accuracy across checkpoints:
    mean accuracy ± std across checkpoints, averaged over corruptions.

    Args:
        all_matrices: list of (n_ratios, n_corruptions) arrays.

    Returns:
        mean_per_ratio: (n_ratios,) mean across checkpoints of corruption-averaged acc
        std_per_ratio: (n_ratios,) std across checkpoints of corruption-averaged acc
    """
    corr_averaged = [np.nanmean(m, axis=1) for m in all_matrices]
    stacked = np.stack(corr_averaged, axis=0)  # (n_ckpt, n_ratios)
    mean_per_ratio = np.mean(stacked, axis=0)
    std_per_ratio = np.std(stacked, axis=0, ddof=0)
    return mean_per_ratio, std_per_ratio


def aggregate_sparsity(all_sparsity_dicts, compression_ratios):
    """Average sparsity across checkpoints (should be nearly identical)."""
    result_mean = {}
    result_std = {}
    for ratio in compression_ratios:
        vals = [d[ratio] for d in all_sparsity_dicts if d[ratio] is not None]
        if vals:
            result_mean[ratio] = np.mean(vals)
            result_std[ratio] = np.std(vals, ddof=0)
        else:
            result_mean[ratio] = None
            result_std[ratio] = None
    return result_mean, result_std


def print_aggregated_results(all_results, compression_ratios, corruptions,
                              args, script_prefix):
    """Print aggregated mean ± std results across checkpoints.

    Args:
        all_results: list of dicts, each with keys:
            'acc_pre': compressed_corruptions_acc_per_ratio
            'acc_post': results_per_ratio
            'sparsity': sparsity_per_ratio
        compression_ratios: list of float
        corruptions: list of str
        args: parsed arguments
        script_prefix: e.g. "resnet18_cifar10" or "resnet18_imagenet"
    """
    n_ckpt = len(all_results)
    print(f"\n{'='*70}")
    print(f"AGGREGATED RESULTS ACROSS {n_ckpt} CHECKPOINTS")
    print(f"{'='*70}")

    # Pre-TTA accuracy
    pre_matrices = [
        extract_accuracy_matrix(r['acc_pre'], compression_ratios, corruptions)
        for r in all_results
    ]
    pre_mean, pre_std = aggregate_accuracy_matrices(pre_matrices)
    pre_corr_mean, pre_corr_std = aggregate_corruption_averaged(pre_matrices)

    print(f"\n--- Pre-TTA Accuracy (mean ± std across {n_ckpt} checkpoints) ---")
    print(f"{script_prefix}_{args.method}_multi_ckpt_pre_tta_mean_per_ratio = {{")
    for i, ratio in enumerate(compression_ratios):
        print(f"  {ratio}: {pre_corr_mean[i]:.4f} ± {pre_corr_std[i]:.4f},")
    print("}")

    # Post-TTA accuracy
    post_matrices = [
        extract_accuracy_matrix(r['acc_post'], compression_ratios, corruptions)
        for r in all_results
    ]
    post_mean, post_std = aggregate_accuracy_matrices(post_matrices)
    post_corr_mean, post_corr_std = aggregate_corruption_averaged(post_matrices)

    print(f"\n--- Post-TTA Accuracy (mean ± std across {n_ckpt} checkpoints) ---")
    print(f"{script_prefix}_{args.method}_multi_ckpt_post_tta_mean_per_ratio = {{")
    for i, ratio in enumerate(compression_ratios):
        print(f"  {ratio}: {post_corr_mean[i]:.4f} ± {post_corr_std[i]:.4f},")
    print("}")

    # Gap: Oracle - TTA
    gap_matrices = [post - pre for post, pre in zip(post_matrices, pre_matrices)]
    gap_mean, gap_std = aggregate_accuracy_matrices(gap_matrices)
    gap_corr_mean, gap_corr_std = aggregate_corruption_averaged(gap_matrices)

    print(f"\n--- Oracle-TTA Gap (mean ± std across {n_ckpt} checkpoints) ---")
    print(f"{script_prefix}_{args.method}_multi_ckpt_gap_per_ratio = {{")
    for i, ratio in enumerate(compression_ratios):
        print(f"  {ratio}: {gap_corr_mean[i]:.4f} ± {gap_corr_std[i]:.4f},")
    print("}")

    # Per-corruption aggregated results (full matrix)
    print(f"\n--- Per-corruption Pre-TTA Mean (matrix: ratios x corruptions) ---")
    print(f"{script_prefix}_{args.method}_multi_ckpt_pre_tta_per_corruption_mean = {{")
    for i, ratio in enumerate(compression_ratios):
        corr_str = ", ".join(f"{pre_mean[i,j]:.4f}" for j in range(len(corruptions)))
        print(f"  {ratio}: [{corr_str}],")
    print("}")

    print(f"\n--- Per-corruption Pre-TTA Std (matrix: ratios x corruptions) ---")
    print(f"{script_prefix}_{args.method}_multi_ckpt_pre_tta_per_corruption_std = {{")
    for i, ratio in enumerate(compression_ratios):
        corr_str = ", ".join(f"{pre_std[i,j]:.4f}" for j in range(len(corruptions)))
        print(f"  {ratio}: [{corr_str}],")
    print("}")

    print(f"\n--- Per-corruption Post-TTA Mean ---")
    print(f"{script_prefix}_{args.method}_multi_ckpt_post_tta_per_corruption_mean = {{")
    for i, ratio in enumerate(compression_ratios):
        corr_str = ", ".join(f"{post_mean[i,j]:.4f}" for j in range(len(corruptions)))
        print(f"  {ratio}: [{corr_str}],")
    print("}")

    print(f"\n--- Per-corruption Post-TTA Std ---")
    print(f"{script_prefix}_{args.method}_multi_ckpt_post_tta_per_corruption_std = {{")
    for i, ratio in enumerate(compression_ratios):
        corr_str = ", ".join(f"{post_std[i,j]:.4f}" for j in range(len(corruptions)))
        print(f"  {ratio}: [{corr_str}],")
    print("}")

    # Sparsity
    all_sparsity = [r['sparsity'] for r in all_results]
    sp_mean, sp_std = aggregate_sparsity(all_sparsity, compression_ratios)
    print(f"\n--- Sparsity (mean ± std) ---")
    print(f"{script_prefix}_{args.method}_multi_ckpt_sparsity = {{")
    for ratio in compression_ratios:
        if sp_mean[ratio] is not None:
            print(f"  {ratio}: {sp_mean[ratio]:.4f} ± {sp_std[ratio]:.4f},")
    print("}")


def _aggregate_formatted_maps(maps):
    """Aggregate list of formatted metric maps across checkpoints.

    Each map has structure: {corruption: [value_at_ratio_0, value_at_ratio_1, ...]}
    where value is either a scalar float, a list of floats (vector), or None.

    Per-seed values may be ``None`` (metric undefined for that seed) or vectors
    of differing length; they are NaN-padded to a common shape and reduced with
    ``np.nanmean`` / ``np.nanstd``.
    """
    mean_map = {}
    std_map = {}

    all_corrs = set()
    for m in maps:
        all_corrs.update(m.keys())

    def _to_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    for corr in sorted(all_corrs):
        all_ratio_values = [m[corr] for m in maps if corr in m]
        if not all_ratio_values:
            continue

        n_ratios = max(len(v) for v in all_ratio_values)
        mean_vals = []
        std_vals = []

        for r_idx in range(n_ratios):
            ckpt_values = [v[r_idx] for v in all_ratio_values if r_idx < len(v)]
            ckpt_values = [c for c in ckpt_values if c is not None]
            if not ckpt_values:
                mean_vals.append(None)
                std_vals.append(None)
                continue

            any_listlike = any(isinstance(c, (list, tuple)) for c in ckpt_values)

            if not any_listlike:
                arr = np.array([_to_float(c) for c in ckpt_values], dtype=float)
                if np.all(np.isnan(arr)):
                    mean_vals.append(None)
                    std_vals.append(None)
                else:
                    mean_vals.append(round(float(np.nanmean(arr)), 4))
                    std_vals.append(round(float(np.nanstd(arr, ddof=0)), 4))
                continue

            # Vector branch: NaN-pad each per-seed value to a common length.
            max_len = 0
            for c in ckpt_values:
                if isinstance(c, (list, tuple)):
                    max_len = max(max_len, len(c))
                else:
                    max_len = max(max_len, 1)

            rows = []
            for c in ckpt_values:
                if isinstance(c, (list, tuple)):
                    row = [_to_float(x) for x in c]
                else:
                    row = [_to_float(c)]
                if len(row) < max_len:
                    row = row + [float("nan")] * (max_len - len(row))
                rows.append(row)

            stacked = np.array(rows, dtype=float)
            if np.all(np.isnan(stacked)):
                mean_vals.append(None)
                std_vals.append(None)
            else:
                mean_vals.append([round(float(v), 4) for v in np.nanmean(stacked, axis=0)])
                std_vals.append([round(float(v), 4) for v in np.nanstd(stacked, axis=0, ddof=0)])

        mean_map[corr] = mean_vals
        std_map[corr] = std_vals

    return mean_map, std_map


def print_aggregated_diagnostic_metrics(all_results, corruptions, args,
                                         script_prefix):
    """Aggregate and print diagnostic metrics (CKA, AME, GA) across checkpoints.

    Args:
        all_results: list of dicts from _run_for_checkpoint(), each containing
            'diagnostic_metrics' key with formatted metric maps.
        corruptions: list of corruption names
        args: parsed arguments
        script_prefix: e.g. "resnet18_imagenet" or "vit_imagenet"
    """
    n_ckpt = len(all_results)

    metric_keys = set()
    for r in all_results:
        dm = r.get('diagnostic_metrics', {})
        metric_keys.update(dm.keys())

    if not metric_keys:
        return

    print(f"\n{'='*70}")
    print(f"AGGREGATED DIAGNOSTIC METRICS ACROSS {n_ckpt} CHECKPOINTS")
    print(f"{'='*70}")

    metric_labels = {
        'cka_pre': 'CKA Dense vs Pruned (Pre-TTA)',
        'cka_post': 'CKA Pruned Pre vs Post-TTA',
        'ame_pre': 'Activation Map Entropy (Pre-TTA)',
        'ame_post': 'Activation Map Entropy (Post-TTA)',
        'ga_pre': 'Gradient Cosine Similarity (Pre-TTA)',
        'ga_post': 'Gradient Cosine Similarity (Post-TTA)',
        'gn_tta_pre': 'Gradient Norm TTA (Pre-TTA)',
        'gn_oracle_pre': 'Gradient Norm Oracle (Pre-TTA)',
        'gn_tta_post': 'Gradient Norm TTA (Post-TTA)',
        'gn_oracle_post': 'Gradient Norm Oracle (Post-TTA)',
    }

    for key in sorted(metric_keys):
        maps = []
        for r in all_results:
            dm = r.get('diagnostic_metrics', {})
            if key in dm:
                maps.append(dm[key])

        if len(maps) < 2:
            continue

        label = metric_labels.get(key, key)
        mean_map, std_map = _aggregate_formatted_maps(maps)

        print(f"\n--- {label} (mean ± std across {n_ckpt} checkpoints) ---")
        print(f"{script_prefix}_{args.method}_multi_ckpt_{key}_mean = {mean_map}")
        print(f"{script_prefix}_{args.method}_multi_ckpt_{key}_std = {std_map}")
