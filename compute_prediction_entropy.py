import os
import argparse
import math
import json
import copy
import torch
import torch.nn.functional as F
import gc

from model_cifar10.resnet import ResNet18_Custom
from compression.fold import ResNet18_ModelFolding
from compression.mag_prune import ResNet18_MagnitudePruning
from compression.wanda import ResNet18_WandaPruning
from compression.rand_prune import ResNet18_RandomPruning
from utils.custom_structural_pruning import one_shot_structural_pruning_resnet18
from utils.utils import load_model
from utils.cifar10_c import set_cifar10_corruptions_data_handlers, Cifar10_C
from utils.datasets import get_cifar10
from utils.seed import set_seed


# Helpers

def set_gpu(gpu):
    if gpu == -1 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def compute_softmax_prediction_entropy(model, loader, device, max_batches=None):
    """
    Compute mean prediction-level softmax entropy H(p) over a dataset.

    H(p) = -sum_c  p_c * log(p_c)   (per sample, then averaged)

    Returns
    -------
    mean_entropy : float
        Mean softmax entropy across all samples.
    std_entropy : float
        Std of per-sample entropy.
    num_samples : int
        Number of samples evaluated.
    """
    model.eval()
    all_entropies = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)

        logits = model(x)
        per_sample_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        all_entropies.append(per_sample_entropy.cpu())

    all_entropies = torch.cat(all_entropies, dim=0)
    return (
        all_entropies.mean().item(),
        all_entropies.std().item(),
        all_entropies.shape[0],
    )


# REPAIR (re-estimate BN running stats from training data)

def apply_repair(model, train_loader, device, num_batches=4):
    """Reset and re-estimate BN running stats on training data."""
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None

    model.train()
    for idx, (x, _) in enumerate(train_loader):
        model(x.to(device))
        if idx >= num_batches - 1:
            break
    model.eval()


# Main

def _run_for_checkpoint(args):
    device = set_gpu(args.cuda_device)
    set_seed(2020)

    corruptions = [
        "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
        "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
        "brightness", "contrast", "elastic_transform", "pixelate",
        "jpeg_compression",
    ]
    if args.debug:
        corruptions = ["shot_noise"]

    set_cifar10_corruptions_data_handlers(
        severity=args.severity, root_path=args.corruptions_root,
        corruptions=corruptions, encoder_name=args.encoder_name,
    )

    compression_ratios = [
        0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.35, 0.45,
        0.55, 0.65, 0.75, 0.85, 0.95,
    ]
    if args.debug:
        compression_ratios = [0.0, 0.5, 0.95]

    pruner_map = {
        "mag-l2": lambda m, r: ResNet18_MagnitudePruning(m, compression_ratio=r, p=2),
        "wanda":  lambda m, r: ResNet18_WandaPruning(m, compression_ratio=r),
        "fold":   lambda m, r: ResNet18_ModelFolding(m, compression_ratio=r),
    }

    # Load pretrained model
    model = ResNet18_Custom()
    load_model(model=model, path=args.checkpoint, device=device)
    origin_param = count_parameters(model)
    model = model.to(device)

    cifar10_c_dataset = Cifar10_C(encoder_name=args.encoder_name, args=args)
    train_loader = get_cifar10(train=True)

    # Results storage
    results = {
        "method": args.method,
        "severity": args.severity,
        "encoder": "resnet18",
        "dataset": "cifar10-c",
        "sparsities": {},
        "prediction_entropy": {},
        "prediction_entropy_std": {},
    }

    num_classes = 10
    uniform_entropy = math.log(num_classes)
    print(f"Uniform entropy (H_max = log({num_classes})): {uniform_entropy:.4f}")

    for ratio in compression_ratios:

        # Compress
        if ratio == 0.0:
            compressed_model = copy.deepcopy(model).to(device)
        else:
            if args.method not in ("taylor", "hessian"):
                pruner = pruner_map[args.method](copy.deepcopy(model), ratio)
                if isinstance(pruner, ResNet18_WandaPruning):
                    pruner.run_calibration(
                        train_loader, device,
                        num_batches=50 if not args.debug else 2,
                    )
                compressed_model = pruner.apply().to(device)
            else:
                compressed_model = one_shot_structural_pruning_resnet18(
                    model=copy.deepcopy(model),
                    compression_method=args.method,
                    train_loader=train_loader,
                    compression_ratio=ratio,
                    device=device,
                    num_batches=50 if not args.debug else 2,
                )

        # REPAIR
        pruned_params = count_parameters(compressed_model)
        sparsity = 100.0 * (1.0 - pruned_params / origin_param)
        results["sparsities"][str(ratio)] = round(sparsity, 2)

        if ratio != 0.0:
            apply_repair(compressed_model, train_loader, device, num_batches=4)
        compressed_model.eval()

        print(f"\n{'='*60}")
        print(f"  Ratio={ratio}  Sparsity={sparsity:.1f}%  Method={args.method}")
        print(f"{'='*60}")

        results["prediction_entropy"][str(ratio)] = {}
        results["prediction_entropy_std"][str(ratio)] = {}

        # Evaluate on training data (baseline)
        mean_h_train, std_h_train, n_train = compute_softmax_prediction_entropy(
            compressed_model, train_loader, device,
            max_batches=args.max_batches,
        )
        results["prediction_entropy"][str(ratio)]["clean_train"] = round(mean_h_train, 6)
        results["prediction_entropy_std"][str(ratio)]["clean_train"] = round(std_h_train, 6)
        print(f"  Clean train:  H(p)={mean_h_train:.4f} ± {std_h_train:.4f}  "
              f"(H_max={uniform_entropy:.4f}, ratio={mean_h_train/uniform_entropy:.3f})")

        # Evaluate on each corruption
        for corruption in corruptions:
            corr_loader = cifar10_c_dataset.get_corruption_data_loader(
                corruption=corruption, batch_size=128, shuffle=False,
                persistent_workers=False, pin_memory=True,
            )

            mean_h, std_h, n_samples = compute_softmax_prediction_entropy(
                compressed_model, corr_loader, device,
                max_batches=args.max_batches,
            )

            results["prediction_entropy"][str(ratio)][corruption] = round(mean_h, 6)
            results["prediction_entropy_std"][str(ratio)][corruption] = round(std_h, 6)

            print(f"  {corruption:20s}:  H(p)={mean_h:.4f} ± {std_h:.4f}  "
                  f"(ratio_to_uniform={mean_h/uniform_entropy:.3f})")

            del corr_loader
            gc.collect()

        del compressed_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save per-checkpoint results
    out_dir = "logs"
    os.makedirs(out_dir, exist_ok=True)
    seed_tag = getattr(args, "seed_tag", "") or ""
    seed_suffix = f"_seed{seed_tag}" if seed_tag != "" else ""
    out_path = os.path.join(
        out_dir,
        f"prediction_entropy_{args.method}_sev{args.severity}{seed_suffix}.json",
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def main(args):
    import copy as _copy
    import numpy as _np

    checkpoint_paths = []
    if hasattr(args, 'checkpoints') and args.checkpoints:
        checkpoint_paths = [p.strip() for p in args.checkpoints.split(',') if p.strip()]
    else:
        checkpoint_paths = [args.checkpoint]

    all_results = []
    for ckpt_idx, ckpt_path in enumerate(checkpoint_paths):
        if len(checkpoint_paths) > 1:
            print(f"\n{'='*70}")
            print(f"CHECKPOINT {ckpt_idx+1}/{len(checkpoint_paths)}: {ckpt_path}")
            print(f"{'='*70}\n")

        args.checkpoint = ckpt_path
        if len(checkpoint_paths) > 1:
            args.seed_tag = str(ckpt_idx)
        result = _run_for_checkpoint(args)
        if result is not None:
            all_results.append(_copy.deepcopy(result))

    if len(checkpoint_paths) > 1 and len(all_results) > 1:
        n_ckpt = len(all_results)
        print(f"\n{'='*70}")
        print(f"AGGREGATED PREDICTION ENTROPY ACROSS {n_ckpt} CHECKPOINTS")
        print(f"{'='*70}")

        agg_mean = {}
        agg_std = {}

        all_ratios = sorted(all_results[0]["prediction_entropy"].keys())
        for ratio_str in all_ratios:
            agg_mean[ratio_str] = {}
            agg_std[ratio_str] = {}
            corr_keys = all_results[0]["prediction_entropy"][ratio_str].keys()
            for corr in corr_keys:
                vals = [r["prediction_entropy"][ratio_str][corr]
                        for r in all_results
                        if ratio_str in r["prediction_entropy"]
                        and corr in r["prediction_entropy"][ratio_str]]
                arr = _np.array(vals, dtype=float)
                agg_mean[ratio_str][corr] = round(float(_np.mean(arr)), 6)
                agg_std[ratio_str][corr] = round(float(_np.std(arr, ddof=0)), 6)

        print(f"\nresnet18_cifar10_{args.method}_multi_ckpt_prediction_entropy_mean = {{")
        for ratio_str in all_ratios:
            print(f"  '{ratio_str}': {agg_mean[ratio_str]},")
        print("}")

        print(f"\nresnet18_cifar10_{args.method}_multi_ckpt_prediction_entropy_std = {{")
        for ratio_str in all_ratios:
            print(f"  '{ratio_str}': {agg_std[ratio_str]},")
        print("}")

        out_dir = "logs"
        os.makedirs(out_dir, exist_ok=True)
        agg_path = os.path.join(
            out_dir,
            f"prediction_entropy_cifar10_{args.method}_sev{args.severity}_multi_ckpt.json",
        )
        with open(agg_path, "w") as f:
            json.dump({"mean": agg_mean, "std": agg_std, "n_checkpoints": n_ckpt}, f, indent=2)
        print(f"\nAggregated results saved to {agg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute prediction-level softmax entropy for compressed models"
    )
    parser.add_argument("--method", required=True,
                        choices=["fold", "taylor", "hessian", "mag-l2", "wanda"])
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--cuda_device", type=int, default=-1)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit batches per evaluation (None=full dataset)")
    parser.add_argument("--checkpoint", type=str,
                        default="pretrained/resnet18_cifar10.pth")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated list of checkpoint paths for multi-seed")
    parser.add_argument("--seed_tag", type=str, default="",
                        help="Optional seed tag (e.g. '0','1','2') appended to the "
                             "output JSON filename so multi-seed runs save per-seed "
                             "files instead of overwriting each other. In multi-"
                             "checkpoint mode this is auto-set to the checkpoint "
                             "index unless explicitly provided.")
    parser.add_argument("--corruptions_root", type=str,
                        default="{}/Few-Shot-Adaptation-Learning/datasets/CIFAR-10-C/{}/{}")
    parser.add_argument("--encoder_name", type=str, default="resnet18")
    parser.add_argument("--prune_data", default="train", choices=["train", "test"])
    args = parser.parse_args()
    main(args)
