from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

# Model loading and corruption loaders come from the diagnostic driver.
import run_diagnostic_metrics as drv
from utils.learner import sar as sarmod
from utils.optimizers import sam_collect_params

CORRUPTIONS_15 = drv.CORRUPTIONS_15


def _flat_grad(loss, params):
    """Flattened mean-gradient of `loss` w.r.t. `params` (autograd.grad, graph kept)."""
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                      for gi, p in zip(g, params)])


def _alignment_for_mask(logits, y, params, mask):
    """cos(g_TTA, g_Oracle), ||g_TTA||, ||g_Oracle||, with both gradients taken
    over the samples in `mask`."""
    n = int(mask.sum().item())
    if n < 2:
        return {"cos": float("nan"), "norm_tta": float("nan"),
                "norm_oracle": float("nan"), "n": n}
    lg = logits[mask]
    tta_loss = sarmod.softmax_entropy(lg).mean()          # unsupervised (entropy)
    oracle_loss = F.cross_entropy(lg, y[mask])            # supervised CE, same samples
    g_tta = _flat_grad(tta_loss, params)
    g_orc = _flat_grad(oracle_loss, params)
    nt, no = g_tta.norm(), g_orc.norm()
    cos = float((g_tta @ g_orc) / (nt * no + 1e-12)) if (nt > 0 and no > 0) else float("nan")
    return {"cos": cos, "norm_tta": float(nt), "norm_oracle": float(no), "n": n}


def load_compressed_model(args, device):
    """Post-compression, pre-adaptation model phi_0, loaded per scheme as in
    the diagnostic driver."""
    ckpt_path, action = drv.resolve_scheme_checkpoint_path(args)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    train_loader = drv.build_clean_train_loader(args, device)
    if action == "load_and_compress":                      # scheme A: dense -> prune
        model = drv.build_dense_model(args.arch, args.dataset)
        drv.load_state_dict_from_payload(model, ckpt_path, device)
        model = drv.compress_on_the_fly(model, args.arch, args.method,
                                        args.compression_ratio, train_loader, device,
                                        num_calib_batches=args.num_calib_batches)
    elif args.scheme == "smaller_dense_tta":               # scheme C: narrow from scratch
        model = drv.build_narrow_model(args.arch, args.dataset, args.compression_ratio)
        drv.load_state_dict_from_payload(model, ckpt_path, device)
    else:                                                  # scheme B: pruned shell + finetuned weights
        model = drv.build_dense_model(args.arch, args.dataset).to(device)
        model = drv.compress_on_the_fly(model, args.arch, args.method,
                                        args.compression_ratio, train_loader, device,
                                        num_calib_batches=args.num_calib_batches)
        drv.load_state_dict_from_payload(model, ckpt_path, device)
    return model.to(device)


@torch.enable_grad()
def ablation_one_corruption(model, params, loader, device, margin_e0, n_total):
    """Gather up to n_total samples, compute the logits once, then evaluate the
    alignment for each mask from the same forward graph."""
    xs, ys = [], []
    seen = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        xs.append(x); ys.append(y); seen += x.size(0)
        if seen >= n_total:
            break
    X = torch.cat(xs)[:n_total].to(device).float()
    Y = torch.cat(ys)[:n_total].to(device)
    logits = model(X)                                      # grad-tracking
    with torch.no_grad():
        ent = sarmod.softmax_entropy(logits.detach())
        reliable = ent < margin_e0
        correct = logits.detach().argmax(1) == Y
    out = {
        "all":              _alignment_for_mask(logits, Y, params, torch.ones_like(reliable)),
        "reliable":         _alignment_for_mask(logits, Y, params, reliable),
        "reliable_correct": _alignment_for_mask(logits, Y, params, reliable & correct),
        "frac_reliable":    float(reliable.float().mean().item()),
        "frac_confident_wrong": float((reliable & ~correct).float().mean().item()),
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--scheme", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--compression_ratio", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--checkpoints_root", default=".")
    p.add_argument("--data_root_cifar", default="./data")
    p.add_argument("--data_root_imagenet", default="./imagenet")
    p.add_argument("--corruptions_root_cifar", default="./CIFAR-10-C")
    p.add_argument("--corruptions_root_imagenet", default="./imagenet_c")
    p.add_argument("--num_calib_batches", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_total", type=int, default=512)
    p.add_argument("--encoder_name", default=None)
    p.add_argument("--corruptions", default="")
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--skip_if_exists", action="store_true")
    args = p.parse_args()
    args.phase = "POST"; args.adapt = "SAR"   # for path helpers / naming parity

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    drv.set_seed(args.seed)
    num_classes = 10 if args.dataset == "cifar10" else 1000
    margin_e0 = 0.4 * math.log(num_classes)               # SAR E0 = 0.4 ln C (per dataset)

    fname = (f"ga_ablation_{args.arch}_{args.dataset}_{args.scheme}_method{args.method}"
             f"_r{args.compression_ratio}_seed{args.seed}_sev{args.severity}.json")
    fpath = os.path.join(args.output_dir, fname)
    if args.skip_if_exists and os.path.isfile(fpath):
        print(f"[skip] {fpath}"); return
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_compressed_model(args, device)
    net = sarmod.configure_model(model)                   # BN affine trainable, batch stats
    params, _ = sam_collect_params(net, freeze_top=True)  # the SAR update subspace
    net.eval()
    n_params = sum(q.numel() for q in model.parameters())

    if args.dataset == "cifar10":
        drv.set_cifar10_corruptions_data_handlers(
            severity=args.severity, root_path=args.corruptions_root_cifar,
            corruptions=list(CORRUPTIONS_15), encoder_name=args.encoder_name)
    else:
        drv.set_imagenet_corruptions_data_handlers(
            corruptions=list(CORRUPTIONS_15), severity=args.severity,
            root_path=args.corruptions_root_imagenet)
    corrs = args.corruptions.split(",") if args.corruptions else list(CORRUPTIONS_15)

    per_corruption = {}
    for c in corrs:
        try:
            loader = drv.build_corruption_loader(args, c)
            res = ablation_one_corruption(net, params, loader, device, margin_e0, args.n_total)
            per_corruption[c] = res
            print(f"[{c}] reliable cos={res['reliable']['cos']:.3f}  "
                  f"reliable_correct cos={res['reliable_correct']['cos']:.3f}  "
                  f"conf-wrong frac={res['frac_confident_wrong']:.3f}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            per_corruption[c] = {"error": f"{type(e).__name__}: {e}"}

    payload = {"context": {"arch": args.arch, "dataset": args.dataset,
                           "scheme": args.scheme, "method": args.method,
                           "compression_ratio": args.compression_ratio,
                           "seed": args.seed, "severity": args.severity,
                           "margin_e0": margin_e0, "n_params": n_params,
                           "n_total": args.n_total},
               "per_corruption": per_corruption}
    with open(fpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[output] {fpath}")


if __name__ == "__main__":
    main()
