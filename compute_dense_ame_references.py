from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from torchvision.models import resnet18 as tv_resnet18
from model_cifar10.resnet import ResNet18_Custom, BasicBlock_Custom
from torchvision.models.resnet import BasicBlock as TVBasicBlock
from utils.effective_rank import (
    compute_layerwise_effective_ranks,
    compute_layerwise_ame_paper_recipe,
)
from utils.seed import set_seed


# Same constants as the diagnostic driver.

CORRUPTIONS_15 = (
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)


def post_residual_hook_targets(arch: str):
    if arch == "rn18":
        return (BasicBlock_Custom, TVBasicBlock)
    raise ValueError(f"Unsupported arch for AME reference: {arch}")


def build_dense_model(arch: str, dataset: str) -> nn.Module:
    if arch == "rn18":
        if dataset == "cifar10":
            return ResNet18_Custom(num_classes=10)
        return tv_resnet18(weights=None, num_classes=1000)
    raise ValueError(f"Unsupported (arch, dataset): ({arch}, {dataset})")


def load_state_dict_from_payload(model: nn.Module, ckpt_path: str, device) -> nn.Module:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    model.load_state_dict(state, strict=True)
    return model.to(device)


# Per-corruption data loaders

def build_cifar10c_loader(corruptions_root: str, corruption: str,
                          severity: int, batch_size: int, num_workers: int,
                          device: torch.device):
    """CIFAR-10-C loader with the same severity slicing as
    `utils.cifar10_c` (five bands of 10K images) but CPU-side normalisation,
    so it also runs without a GPU."""
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    from torchvision.transforms import v2

    npy_path = os.path.join(corruptions_root, f"{corruption}.npy")
    labels_path = os.path.join(corruptions_root, "labels.npy")
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(npy_path)

    start_idx = (severity - 1) * 10000
    end_idx = severity * 10000
    imgs_all = np.load(npy_path, mmap_mode="r")
    lbls_all = np.load(labels_path)

    # CIFAR-10 normalisation (same constants as utils/cifar10_c.py).
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2471, 0.2435, 0.2616]).view(3, 1, 1)

    class _NpyDataset(Dataset):
        def __init__(self, imgs, lbls):
            self.imgs = imgs
            self.lbls = lbls

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, i):
            img = np.array(self.imgs[i])      # (32, 32, 3) uint8
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img = (img - mean) / std
            return img, int(self.lbls[i])

    ds = _NpyDataset(imgs_all[start_idx:end_idx], lbls_all[start_idx:end_idx])
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )


def build_imagenetc_loader(corruptions_root: str, corruption: str,
                           severity: int, batch_size: int, num_workers: int):
    from utils.imagenet_c import (
        ImageNet_C, set_imagenet_corruptions_data_handlers, set_individual_corruption,
    )

    class _Args:
        pass

    args = _Args()
    args.corruptions_root_imagenet = corruptions_root
    args.encoder_name = "resnet18"
    set_imagenet_corruptions_data_handlers(
        corruptions=[corruption], severity=severity, root_path=corruptions_root,
    )
    set_individual_corruption(corruption_name=corruption)
    ds_handle = ImageNet_C(encoder_name="resnet18", args=args)
    return ds_handle.get_corruption_data_loader(
        corruption=corruption, batch_size=batch_size,
        shuffle=False, persistent_workers=False, pin_memory=True,
        num_workers=num_workers,
    )


# Optional ImageNet-C tar extraction.

def extract_imagenet_c_tar(tar_path: str, extract_to: str):
    if not os.path.isfile(tar_path):
        raise FileNotFoundError(f"Tar not found: {tar_path}")
    os.makedirs(extract_to, exist_ok=True)
    print(f"[extract] {tar_path} -> {extract_to}", flush=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(extract_to)
    print(f"[extract] done", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=("rn18",), default="rn18")
    p.add_argument("--dataset", choices=("cifar10", "imagenet"), required=True)
    p.add_argument("--checkpoint_pattern", required=True,
                   help="Path template. Use `{seed}` to interpolate the seed. "
                        "If the template contains no `{seed}`, the same "
                        "checkpoint is reused for every requested seed.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--severities", type=int, nargs="+", default=[3, 5],
                   choices=[1, 2, 3, 4, 5])
    p.add_argument("--corruptions_root", type=str, default="",
                   help="Path to CIFAR-10-C / ImageNet-C root.")
    p.add_argument("--imagenet_tar", type=str, default="",
                   help="(Optional, ImageNet only) Path to an ImageNet-C "
                        "severity-X tar that we should extract into "
                        "`--imagenet_root` before running.")
    p.add_argument("--imagenet_root", type=str, default="",
                   help="Extraction target for `--imagenet_tar`. If unset and "
                        "`--imagenet_tar` is given, defaults to "
                        "$SCRATCH/imagenet-c-extracted.")
    p.add_argument("--output_dir", type=str, default="dense_ame_references")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--num_batches", type=int, default=20,
                   help="Mini-batches aggregated into the channel-wise Gram "
                        "matrix.")
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu (default: cuda if available, else cpu).")
    p.add_argument("--corruptions", nargs="+", default=list(CORRUPTIONS_15))
    p.add_argument("--skip_if_exists", action="store_true")
    p.add_argument("--ame_recipe", choices=("pooled", "paper"), default="pooled",
                   help="AME aggregator, as in the diagnostic driver: `pooled` "
                        "(pooled covariance) or `paper` (per-batch entropy, "
                        "then mean across batches).")
    args = p.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"[ctx] arch={args.arch} dataset={args.dataset} device={device} "
          f"seeds={args.seeds} severities={args.severities} "
          f"corruptions={len(args.corruptions)} num_batches={args.num_batches}",
          flush=True)

    # Optional: extract ImageNet-C tar before computing.
    if args.dataset == "imagenet" and args.imagenet_tar:
        target = args.imagenet_root or os.path.expandvars(
            "$SCRATCH/imagenet-c-extracted")
        extract_imagenet_c_tar(args.imagenet_tar, target)
        args.corruptions_root = target

    if not args.corruptions_root:
        sys.exit("--corruptions_root is required.")
    os.makedirs(args.output_dir, exist_ok=True)

    targets = post_residual_hook_targets(args.arch)

    for seed in args.seeds:
        recipe_suffix = "_paper" if args.ame_recipe == "paper" else ""
        out_path = os.path.join(
            args.output_dir,
            f"dense_ame_{args.arch}_{args.dataset}_seed{seed}{recipe_suffix}.json",
        )
        if args.skip_if_exists and os.path.isfile(out_path):
            print(f"[skip] {out_path} exists", flush=True)
            continue

        set_seed(seed)
        model = build_dense_model(args.arch, args.dataset)
        ckpt_path = args.checkpoint_pattern.replace("{seed}", str(seed))
        if not os.path.isfile(ckpt_path):
            print(f"[warn] checkpoint missing for seed {seed}: {ckpt_path} - "
                  f"skipping", flush=True)
            continue
        model = load_state_dict_from_payload(model, ckpt_path, device)
        model.eval()
        print(f"\n[seed {seed}] loaded {ckpt_path}", flush=True)

        per_severity = {}
        t_start = time.time()
        for sev in args.severities:
            print(f"  severity {sev}:", flush=True)
            per_corruption = {}
            for corr_idx, corr in enumerate(args.corruptions):
                t0 = time.time()
                if args.dataset == "cifar10":
                    loader = build_cifar10c_loader(
                        args.corruptions_root, corr, sev,
                        args.batch_size, args.num_workers, device=device,
                    )
                else:
                    loader = build_imagenetc_loader(
                        args.corruptions_root, corr, sev,
                        args.batch_size, args.num_workers,
                    )
                if args.ame_recipe == "paper":
                    ranks = compute_layerwise_ame_paper_recipe(
                        model, loader, device=str(device),
                        num_batches=args.num_batches,
                        target_layer_types=targets,
                    )
                else:
                    ranks = compute_layerwise_effective_ranks(
                        model, loader, device=str(device),
                        num_batches=args.num_batches,
                        target_layer_types=targets,
                    )
                per_corruption[corr] = {k: float(v) for k, v in ranks.items()}
                dt = time.time() - t0
                vals = list(per_corruption[corr].values())
                print(f"    [{corr_idx+1:>2d}/{len(args.corruptions)}] {corr:>20s} "
                      f"mean={sum(vals)/len(vals):>6.2f} min={min(vals):>6.2f} "
                      f"({dt:>5.1f}s)", flush=True)
            per_severity[str(sev)] = per_corruption

        payload = {
            "context": {
                "arch": args.arch, "dataset": args.dataset, "seed": seed,
                "severities": args.severities,
                "num_batches": args.num_batches,
                "checkpoint": ckpt_path,
                "wall_seconds": time.time() - t_start,
            },
            "per_severity": per_severity,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[wrote] {out_path}  ({time.time()-t_start:.0f}s total)",
              flush=True)


if __name__ == "__main__":
    main()
