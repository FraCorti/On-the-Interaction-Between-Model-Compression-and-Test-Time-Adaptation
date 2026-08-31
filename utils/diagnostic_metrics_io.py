"""Ingestion and multi-seed aggregation of diagnostic-metric JSONs (CKA, AME, MSA, PE).

Each JSON filename encodes (arch, dataset, scheme, adapt, method, ratio, seed,
severity, phase). Cells are validated, grouped by (scheme, adapt, method, ratio,
severity, corruption) and averaged across seeds. AME values are taken from the
`_ameonly` companion JSON when a complete one exists; the dense per-layer AME
reference for the L2-to-dense distance is read from `dense_ame_references/`.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# Filename pattern: arch_dataset_scheme_adaptX_methodY_rR_seedS_sevV_phaseP[_ameonly].json
_FNAME_RE = re.compile(
    r"^(?P<arch>rn18|vit)_(?P<dataset>cifar10|imagenet)_"
    r"(?P<scheme>dense_prune_tta|dense_prune_finetune_in_tta|smaller_dense_tta)_"
    r"adapt(?P<adapt>NONE|SAR|PEA|SPA|FOA|oracle|oracle_spa)_"
    r"method(?P<method>fold|taylor|hessian|wanda|mag-l2)_"
    r"r(?P<ratio>[0-9.]+?)_seed(?P<seed>\d+)_sev(?P<sev>\d+)_"
    r"phase(?P<phase>PRE|POST)(?P<ameonly>_ameonly)?\.json$"
)

# Result files with an mtime at or before this cut-off are ignored (0.0 disables the filter).
IMAGENET_MTIME_CUTOFF = 0.0

QUARANTINE_PREFIXES = ()


def parse_filename(fname: str) -> Optional[dict]:
    m = _FNAME_RE.match(fname)
    if m is None:
        return None
    d = m.groupdict()
    d["ratio"] = float(d["ratio"])
    d["seed"] = int(d["seed"])
    d["sev"] = int(d["sev"])
    d["ameonly"] = bool(d["ameonly"])
    return d


def _quarantined(path: str) -> bool:
    bn = os.path.basename(os.path.dirname(path))
    return any(bn.startswith(p) for p in QUARANTINE_PREFIXES)


def _approx_eq(a, b) -> bool:
    if a is None or b is None:
        return False
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def _build_pre_index(roots: Iterable[str]) -> Dict[str, str]:
    """Index every PRE-phase full JSON by basename so a POST cell can find
    its matched PRE in O(1) for the silent-PRE check."""
    out: Dict[str, str] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for fname in os.listdir(root):
            meta = parse_filename(fname)
            if meta is None or meta["phase"] != "PRE" or meta["ameonly"]:
                continue
            out.setdefault(fname, os.path.join(root, fname))
    return out


def _matched_pre_name(post_fname: str) -> str:
    base = re.sub(r"_adapt(SPA|PEA|SAR|oracle|oracle_spa|FOA)_",
                  "_adaptNONE_", post_fname)
    return base.replace("_phasePOST", "_phasePRE")


def _is_silent_pre(post_path: str, pre_path: Optional[str]) -> bool:
    if pre_path is None:
        return False
    try:
        with open(post_path) as f:
            post = json.load(f)
        with open(pre_path) as f:
            pre = json.load(f)
    except Exception:
        return False
    post_v = post.get("per_corruption", {}).get("gaussian_noise", {}).get("pred_entropy_mean")
    pre_v = pre.get("per_corruption", {}).get("gaussian_noise", {}).get("pred_entropy_mean")
    return _approx_eq(post_v, pre_v)


# Ingestion: walk roots, apply validity rules, group cells.

def _ameonly_payload_complete(payload: dict, corruptions: Iterable[str]) -> bool:
    pc = payload.get("per_corruption", {})
    for c in corruptions:
        entry = pc.get(c)
        if not isinstance(entry, dict):
            return False
        per_layer = entry.get("ame_per_layer")
        if not per_layer:
            return False
    return True


def load_diagnostic_cells(
    roots: Iterable[str],
    arch: str,
    dataset: str,
    enforce_silent_pre_check: bool = True,
    prefer_ameonly: bool = True,
) -> Tuple[Dict[Tuple, dict], dict]:
    """Walk `roots`, keep only cells matching (arch, dataset), and apply the
    validity gates. Returns:

        cells : {(scheme, adapt, method, ratio, seed, sev, phase): per_corruption_dict}
        stats : {key: int}    counters for the ingestion report.

    When `prefer_ameonly` is True the AME fields (`ame_mean`, `ame_min`,
    `ame_per_layer`) are taken from the matched `_ameonly` JSON whenever
    that file exists and contains all 15 corruptions; the other metrics
    come from the full JSON.
    """
    pre_index = _build_pre_index(roots)
    stats = defaultdict(int)
    cells: Dict[Tuple, dict] = {}

    # `_repair.json` cells (REPAIR applied after compression) supersede their
    # no-REPAIR counterpart and carry all fields, so they get no AME overlay.
    repair_index: Dict[str, str] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for fname in os.listdir(root):
            if fname.endswith("_repair.json"):
                base = fname.replace("_repair.json", ".json")
                repair_index.setdefault(base, os.path.join(root, fname))

    # Index ameonly JSONs by their full-driver filename; `_ameonly_paper.json`
    # takes precedence over `_ameonly.json`.
    ameonly_index: Dict[str, str] = {}
    if prefer_ameonly:
        ameonly_paper_index: Dict[str, str] = {}
        ameonly_pooled_index: Dict[str, str] = {}
        for root in roots:
            if not os.path.isdir(root):
                continue
            for fname in os.listdir(root):
                if fname.endswith("_ameonly_paper.json"):
                    full_fname = fname.replace("_ameonly_paper.json", ".json")
                    ameonly_paper_index.setdefault(full_fname, os.path.join(root, fname))
                    continue
                meta = parse_filename(fname)
                if meta is None or not meta["ameonly"]:
                    continue
                if meta["arch"] != arch or meta["dataset"] != dataset:
                    continue
                full_fname = fname.replace("_ameonly.json", ".json")
                ameonly_pooled_index.setdefault(full_fname, os.path.join(root, fname))
        # Paper recipe wins on every key it covers; pooled fills the rest.
        ameonly_index = dict(ameonly_pooled_index)
        ameonly_index.update(ameonly_paper_index)

    for root in roots:
        if not os.path.isdir(root):
            stats["missing_root"] += 1
            continue
        for fname in sorted(os.listdir(root)):
            # Parse `_repair.json` under its canonical cell filename.
            is_repair = fname.endswith("_repair.json")
            if is_repair:
                fname_for_parse = fname.replace("_repair.json", ".json")
            else:
                fname_for_parse = fname
            meta = parse_filename(fname_for_parse)
            if meta is None:
                continue
            if meta["arch"] != arch or meta["dataset"] != dataset:
                continue
            if meta["ameonly"]:
                # Companion file, already indexed above.
                stats["skip_ameonly_indexed"] += 1
                continue

            fpath = os.path.join(root, fname)
            # A `_repair.json` for the same cell supersedes this file.
            if (not is_repair) and (fname in repair_index):
                stats["skip_superseded_by_repair"] += 1
                continue
            if _quarantined(fpath):
                stats["skip_quarantine"] += 1
                continue
            if dataset == "imagenet":
                try:
                    mt = os.path.getmtime(fpath)
                except OSError:
                    stats["skip_stat_fail"] += 1
                    continue
                if mt <= IMAGENET_MTIME_CUTOFF:
                    stats["skip_imagenet_mtime"] += 1
                    continue
            if meta["phase"] == "POST" and enforce_silent_pre_check:
                pre_fname = _matched_pre_name(fname_for_parse)
                pre_path = pre_index.get(pre_fname)
                if _is_silent_pre(fpath, pre_path):
                    stats["skip_silent_pre"] += 1
                    continue

            try:
                with open(fpath) as f:
                    payload = json.load(f)
            except Exception:
                stats["skip_parse_fail"] += 1
                continue

            # `_repair.json` cells need no AME overlay.
            if is_repair:
                stats["repair_cell_loaded"] += 1
            elif prefer_ameonly and fname in ameonly_index:
                try:
                    with open(ameonly_index[fname]) as f:
                        ameonly_payload = json.load(f)
                except Exception:
                    ameonly_payload = None
                if ameonly_payload is not None and _ameonly_payload_complete(
                        ameonly_payload, CORRUPTIONS_15):
                    for corr, ame_entry in ameonly_payload.get("per_corruption", {}).items():
                        if not isinstance(ame_entry, dict):
                            continue
                        target = payload.setdefault("per_corruption", {}).setdefault(corr, {})
                        for k in ("ame_mean", "ame_min", "ame_per_layer"):
                            if k in ame_entry:
                                target[k] = ame_entry[k]
                    stats["ame_overlay_applied"] += 1
                else:
                    stats["ame_overlay_incomplete_skipped"] += 1

            key = (meta["scheme"], meta["adapt"], meta["method"],
                   meta["ratio"], meta["seed"], meta["sev"], meta["phase"])
            # Duplicate cells across roots: keep the first (roots order is caller-controlled).
            if key in cells:
                stats["skip_duplicate"] += 1
                continue
            cells[key] = {
                "per_corruption": payload.get("per_corruption", {}),
                "context": payload.get("context", {}),
            }
            stats["loaded"] += 1
    return cells, dict(stats)


# Dense AME reference loading and L2-to-dense computation

def load_dense_ame_references(
    arch: str, dataset: str, seeds: Iterable[int],
    references_dir: str = "dense_ame_references",
) -> Dict[Tuple[int, int, str], Dict[str, float]]:
    """Load dense-model per-layer AME vectors keyed by `(seed, severity, corruption)`.

    Reference files have the shape
    `{context: {...}, per_severity: {sev_str: {corruption: {layer_name: ame}}}}`.
    A missing seed-specific file falls back to the seed-0 file.
    """
    ref: Dict[Tuple[int, int, str], Dict[str, float]] = {}
    for seed in seeds:
        path = os.path.join(
            references_dir,
            f"dense_ame_{arch}_{dataset}_seed{seed}.json",
        )
        if not os.path.isfile(path):
            # Single-seed proxy fallback
            fallback = os.path.join(
                references_dir,
                f"dense_ame_{arch}_{dataset}_seed0.json",
            )
            if os.path.isfile(fallback):
                path = fallback
            else:
                continue
        try:
            with open(path) as f:
                payload = json.load(f)
        except Exception:
            continue
        for sev_str, corr_map in payload.get("per_severity", {}).items():
            try:
                sev = int(sev_str)
            except (TypeError, ValueError):
                continue
            for corr, layer_map in corr_map.items():
                if not isinstance(layer_map, dict) or not layer_map:
                    continue
                ref[(seed, sev, corr)] = {k: float(v) for k, v in layer_map.items()}
    return ref


def ame_l2_to_dense(
    compressed_per_layer: dict,
    dense_per_layer: Optional[dict],
) -> float:
    """`||AME_compressed - AME_dense||_2` over matched per-layer keys.

    `ame_per_layer` stores effective ranks exp(entropy), so the log is taken
    before differencing to obtain entropies in nats; values <= 0 are clipped
    to 1.0 (log(1)=0). Returns NaN if either side is missing or no layer
    overlap exists.
    """
    if not isinstance(compressed_per_layer, dict) or not compressed_per_layer:
        return math.nan
    if not isinstance(dense_per_layer, dict) or not dense_per_layer:
        return math.nan
    shared = sorted(set(compressed_per_layer.keys()) & set(dense_per_layer.keys()))
    if not shared:
        return math.nan
    acc = 0.0
    for k in shared:
        try:
            c = float(compressed_per_layer[k])
            d = float(dense_per_layer[k])
        except (TypeError, ValueError):
            return math.nan
        # AME = entropy (nats) = log(eff_rank). Clip min to 1.0 so a
        # collapsed layer (eff_rank=1) maps to entropy 0.
        c_ent = math.log(max(c, 1.0))
        d_ent = math.log(max(d, 1.0))
        diff = c_ent - d_ent
        acc += diff * diff
    return math.sqrt(acc)


# Metric extraction and aggregation

def _safe_float(x) -> float:
    try:
        v = float(x)
        if math.isnan(v):
            return math.nan
        return v
    except (TypeError, ValueError):
        return math.nan


def metric_cka_min(corr_entry: dict) -> float:
    """Worst-layer CKA (`cka_diag_min` written by `metric_cka_to_dense`)."""
    return _safe_float(corr_entry.get("cka_diag_min"))


def metric_ame_mean(corr_entry: dict) -> float:
    """Layer-mean Activation-Map Entropy: mean over `ame_per_layer`."""
    return _safe_float(corr_entry.get("ame_mean"))


def metric_msa_dsr_mean(corr_entry: dict) -> float:
    """Mode-Switching Activation distance d_SR (mean over MSA n_samples)."""
    return _safe_float(corr_entry.get("msa_dsr_mean"))


def metric_pred_entropy_mean(corr_entry: dict) -> float:
    return _safe_float(corr_entry.get("pred_entropy_mean"))


CORRUPTIONS_15 = (
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)


def aggregate_seed_mean(
    cells: Dict[Tuple, dict],
    scheme: str,
    adapt: str,
    phase: str,
    severity: int,
    ratios: List[float],
    methods: List[str],
    metric_fn: Callable[..., float],
    corruptions: Iterable[str] = CORRUPTIONS_15,
    seeds: Iterable[int] = (0, 1, 2),
) -> Dict[str, Dict[str, List[float]]]:
    """Return `{method: {corruption: [val_per_ratio]}}` with each value =
    NaN-aware seed-mean across the configured seeds. Methods with no
    valid seeds for a given (ratio, corruption) get NaN at that index.

    `metric_fn` may have signature `metric_fn(corr_entry)` (legacy) or
    `metric_fn(corr_entry, seed=..., sev=..., corruption=...)` when the
    reduction needs context (e.g. dense-AME L2).
    """
    out: Dict[str, Dict[str, List[float]]] = {}
    for method in methods:
        corr_map: Dict[str, List[float]] = {}
        for corr in corruptions:
            row: List[float] = []
            for r in ratios:
                seed_vals: List[float] = []
                for seed in seeds:
                    key = (scheme, adapt, method, r, seed, severity, phase)
                    cell = cells.get(key)
                    if cell is None:
                        continue
                    entry = cell["per_corruption"].get(corr)
                    if entry is None:
                        continue
                    try:
                        v = metric_fn(entry, seed=seed, sev=severity,
                                      corruption=corr)
                    except TypeError:
                        v = metric_fn(entry)
                    if not math.isnan(v):
                        seed_vals.append(v)
                if seed_vals:
                    row.append(sum(seed_vals) / len(seed_vals))
                else:
                    row.append(math.nan)
            corr_map[corr] = row
        out[method] = corr_map
    return out


def make_ame_l2_metric(
    dense_ref: Dict[Tuple[int, int, str], Dict[str, float]],
) -> Callable[..., float]:
    """Closure that returns `metric_fn(entry, seed=..., sev=..., corruption=...)`
    -> `||ame_per_layer_compressed - ame_per_layer_dense||_2`.

    Designed to be passed to `aggregate_seed_mean` / `hero_dict_for_group`."""
    def _metric(entry: dict, **ctx) -> float:
        seed = ctx.get("seed")
        sev = ctx.get("sev")
        corruption = ctx.get("corruption")
        compressed = entry.get("ame_per_layer") if isinstance(entry, dict) else None
        dense = dense_ref.get((seed, sev, corruption))
        if dense is None and seed != 0:
            # Single-seed reference fallback: reuse seed-0 dense AME for
            # seeds 1/2 if a seed-specific reference isn't available.
            dense = dense_ref.get((0, sev, corruption))
        return ame_l2_to_dense(compressed, dense)
    return _metric


def filter_nan_methods(method_dict: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    """Drop methods whose every (corruption, ratio) cell is NaN.
    Used to remove smaller_dense_tta from AME / CKA outputs on ImageNet."""
    kept = {}
    for m, corr_map in method_dict.items():
        any_valid = any(
            (not math.isnan(v))
            for vec in corr_map.values()
            for v in vec
        )
        if any_valid:
            kept[m] = corr_map
    return kept


# Display-name / method-token shims shared by the plotting scripts.

# Display name -> filesystem method token mapping.
HERO_NAME_TO_FILE_TOKEN = {
    "Wanda": "wanda",
    "Taylor": "taylor",
    "OBD": "hessian",          # OBD's filename token is `hessian`
    "Fold": "fold",
    "Magnitude (L2)": "mag-l2",
}
FILE_TOKEN_TO_HERO_NAME = {v: k for k, v in HERO_NAME_TO_FILE_TOKEN.items()}

DATA_BASED_HERO_METHODS = ["Wanda", "Taylor", "OBD"]
DATA_FREE_HERO_METHODS = ["Fold", "Magnitude (L2)"]


def hero_dict_for_group(
    cells: Dict[Tuple, dict],
    scheme: str,
    adapt: str,
    phase: str,
    severity: int,
    ratios: List[float],
    group_methods: List[str],
    metric_fn: Callable[..., float],
) -> Dict[str, Dict[str, List[float]]]:
    """Same as `aggregate_seed_mean` but keyed by display names
    (Wanda / Taylor / OBD / Fold / Magnitude (L2)) instead of file tokens.

    For Oracle pairs pass `adapt='oracle'` and prefix the returned keys with `'Oracle '`.
    """
    file_tokens = [HERO_NAME_TO_FILE_TOKEN[m] for m in group_methods]
    by_token = aggregate_seed_mean(
        cells=cells, scheme=scheme, adapt=adapt, phase=phase,
        severity=severity, ratios=ratios, methods=file_tokens,
        metric_fn=metric_fn,
    )
    return {FILE_TOKEN_TO_HERO_NAME[t]: by_token[t] for t in file_tokens}
