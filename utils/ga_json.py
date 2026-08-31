"""Per-(arch, method, ratio, corruption, seed) JSON persistence for the
matched-oracle gradient-alignment outputs.

One file per (arch, tta_method, compression_method, severity, calibration, seed),
holding cosine + L2 norms for every (ratio, corruption) pair, so downstream
analyses read structured JSON instead of re-parsing run stdout.
"""

import json
import os
import subprocess
import time
from datetime import datetime, timezone


def _git_sha():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _seed_from_checkpoint(ckpt_path):
    """Best-effort seed extraction from checkpoint filename.

    Matches the seeded-checkpoint naming:
      vit_base_patch16_224_augreg2_seed{1,2,3}.pth
      resnet18_{cifar10,imagenet}_seed{0,1,2}.pth
    Returns None if no seed is encoded in the filename.
    """
    if not ckpt_path:
        return None
    base = os.path.basename(ckpt_path)
    import re
    m = re.search(r'seed(\d+)', base)
    return int(m.group(1)) if m else None


def _matrix(metric_dict, compression_ratios, corruptions):
    """Turn the driver's nested {ratio: [per-corruption-list-of-values]} into
    a 2D list of float-or-null suitable for JSON. Empty lists -> null."""
    out = []
    for ratio in compression_ratios:
        row = []
        per_corr = metric_dict.get(ratio, [])
        for j, _ in enumerate(corruptions):
            vals = per_corr[j] if j < len(per_corr) else []
            if isinstance(vals, list):
                if len(vals) == 0:
                    row.append(None)
                else:
                    finite = [v for v in vals if v is not None]
                    row.append(sum(finite) / len(finite) if finite else None)
            else:
                row.append(vals)
        out.append(row)
    return out


def dump_ga_json(*, output_dir, arch, tta_method, oracle_method,
                 compression_method, calibration, severity, phase,
                 checkpoint_path, compression_ratios, corruptions,
                 grad_align, grad_norm_tta, grad_norm_oracle):
    """Write one matched-oracle GA JSON file for a single (arch, tta, method,
    severity, calib, seed, phase) combination.

    Args mirror the driver-side variables. ``grad_align`` / ``grad_norm_tta`` /
    ``grad_norm_oracle`` are the nested dicts the drivers already populate
    (e.g. grad_align_pre_corr). ``phase`` is 'pre_tta' or 'post_tta'.

    Returns the absolute path of the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    seed = _seed_from_checkpoint(checkpoint_path)
    seed_tag = f"seed{seed}" if seed is not None else "seedNA"
    fname = (f"{arch}_{tta_method}_{compression_method}_sev{severity}"
             f"_calib{calibration}_{seed_tag}_{phase}.json")
    path = os.path.join(output_dir, fname)

    payload = {
        'arch': arch,
        'tta_method': tta_method,
        'oracle_method': oracle_method,
        'compression_method': compression_method,
        'calibration': calibration,
        'severity': severity,
        'phase': phase,
        'checkpoint': checkpoint_path,
        'seed': seed,
        'ratios': list(compression_ratios),
        'corruptions': list(corruptions),
        'ga_cos':      _matrix(grad_align,       compression_ratios, corruptions),
        'norm_tta':    _matrix(grad_norm_tta,    compression_ratios, corruptions),
        'norm_oracle': _matrix(grad_norm_oracle, compression_ratios, corruptions),
        'git_sha': _git_sha(),
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    return path
