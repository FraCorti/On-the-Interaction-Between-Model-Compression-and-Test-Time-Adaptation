import os
import copy
import math
import gc
from contextlib import contextmanager
from collections import OrderedDict
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from thop import profile

from torchvision.models import resnet18

from model.resnet import (
    merge_channel_ResNet18_big_clustering,
    merge_channel_ResNet18_big_clustering_approx_repair,
)
from utils.format_results import format_metric_map, format_numpy_floats_map
from utils.imagenet_c import (
    set_individual_corruption,
    set_imagenet_corruptions_data_handlers,
    ImageNet_C,
)
from utils.seed import set_seed
from utils.utils import eval_model, load_model, fuse_bnorms_resnet
from utils.utils import DI_REPAIR, NO_REPAIR, REPAIR, DF_REPAIR
from utils.datasets import get_imagenet
from utils.compute_hessian_vision import pack_to_gpu_from_loader, get_hessian, compute_fisher_trace_subspace, compute_ece, compute_gradient_snr, compute_layerwise_gradient_snr, get_sorted_layer_snr_list, compute_layerwise_hessian_trace, get_sorted_layer_hessian_trace_list
from utils.gradient_alignment import compute_gradient_alignment
from utils.fisher_layerwise import compute_layerwise_fisher, get_sorted_layer_fisher_list
from utils.loss_barrier import get_id_barrier, get_ood_barrier, inflate_state_dict, restore_bn_running_stats
from cka import CKACalculator
from utils import learner, losses
from utils.learner.set_optimizer import get_adaptation_optimizer_configure_model

import wandb
import sharpness


# Prompt entropy utilities
def _shannon_entropy_from_gram(G: torch.Tensor) -> float:
    """
    G: (C x C) positive semi-definite Gram matrix on CPU (float32/64).
    Returns Shannon entropy of normalized eigenvalues.
    """
    G = 0.5 * (G + G.t())
    evals = torch.linalg.eigvalsh(G.to(torch.float64))
    evals = torch.clamp(evals, min=0.0)
    tot = float(evals.sum().item())
    if tot <= 0.0:
        return float("nan")
    p = evals / tot
    p = p[p > 0]
    H = -float(torch.sum(p * torch.log(p)).item())
    return H


@contextmanager
def _layer_gram_accumulator(model: torch.nn.Module, layer_names):
    """
    Register forward hooks on `layer_names` and accumulate Z^T Z per layer.
    For 4D activations (N,C,H,W): einsum over spatial positions.
    For 2D activations (N,C): A^T A.
    Returns dict: name -> Gram (C x C) on CPU.
    """
    grams = {}
    handles = []
    name2mod = dict(model.named_modules())

    def make_hook(layer_name):
        def hook(_m, _inp, out):
            act = out
            if isinstance(act, (list, tuple)):
                act = act[0]
            if not torch.is_tensor(act):
                return
            with torch.no_grad():
                if act.dim() == 4:
                    N, C, H, W = act.shape
                    A = act.detach().to(torch.float32).view(N, C, H * W)
                    Gb = torch.einsum("ncs,nks->ck", A, A)
                elif act.dim() == 2:
                    A = act.detach().to(torch.float32)
                    Gb = A.t().mm(A)
                else:
                    return
                Gb = Gb.to("cpu")
                if layer_name not in grams:
                    grams[layer_name] = Gb
                else:
                    grams[layer_name] += Gb

        return hook

    for ln in layer_names:
        if ln in name2mod:
            handles.append(name2mod[ln].register_forward_hook(make_hook(ln)))

    try:
        yield grams
    finally:
        for h in handles:
            h.remove()


def get_resnet18_layernames(
        model,
        granularity: str = "block",
        include_stem: bool = False,
        include_maxpool: bool = False,
        include_avgpool: bool = False,
):
    """
    Enumerate layers to hook for torchvision ResNet-18.
    - 'block': hook BasicBlock modules => 'layer{k}.{i}' (output is post-add, post-ReLU)
    - 'conv' : hook '...conv1' and '...conv2' inside each BasicBlock
    Optional: stem 'relu', 'maxpool', 'avgpool'.
    """
    names = []
    if include_stem and hasattr(model, "relu"):
        names.append("relu")
    if include_maxpool and hasattr(model, "maxpool"):
        names.append("maxpool")
    for k in range(1, 5):
        layer = getattr(model, f"layer{k}")
        for i in range(len(layer)):
            if granularity == "block":
                names.append(f"layer{k}.{i}")
            elif granularity == "conv":
                names.append(f"layer{k}.{i}.conv1")
                names.append(f"layer{k}.{i}.conv2")
            else:
                raise ValueError("granularity must be 'block' or 'conv'")
    if include_avgpool and hasattr(model, "avgpool"):
        names.append("avgpool")  # (N,C,1,1); valid for Gram/eigs
    return names


def compute_prompt_entropy(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        layer_names,
        max_batches: int = 20,
        eps: float = 1e-6,
        debug: bool = False,
        return_std: bool = False,
):
    """
    Batch-averaged prompt entropy per hooked layer.

    For each batch:
      1) build a per-layer Gram G_b,l from that batch alone
      2) compute Shannon entropy H_b,l from normalized eigvals of G_b,l
    Then average H_b,l across batches b (up to `max_batches`) to get H_l.

    Args:
        model: network to hook (ResNet-18).
        loader: DataLoader over the corruption split (bs=256 recommended).
        device: torch.device.
        layer_names: iterable of module names to hook (e.g., ["layer1.0", ..., "layer4.1"]).
        max_batches: number of batches to aggregate (e.g., 20).
        eps: small ridge (scaled by trace/C) added to G for numeric stability.
        debug: if True, stop after the first batch.
        return_std: if True, also return per-layer std across batches.

    Returns:
        entropies: OrderedDict {layer_name -> mean H_l across batches}
        eranks:    OrderedDict {layer_name -> exp(mean H_l)}
        (optional) entropies_std: OrderedDict {layer_name -> std(H_b,l) across batches}
    """
    model.eval()
    name2mod = dict(model.named_modules())

    # Running stats across batches
    sum_H = {ln: 0.0 for ln in layer_names}
    sumsq_H = {ln: 0.0 for ln in layer_names}
    cnt_H = {ln: 0 for ln in layer_names}

    # Hooks are registered once; batch_grams is reset for every batch
    batch_grams = None

    def _shannon_entropy_from_gram(G: torch.Tensor) -> float:
        G = 0.5 * (G + G.t())
        evals = torch.linalg.eigvalsh(G.to(torch.float64))
        evals = torch.clamp(evals, min=0.0)
        tot = float(evals.sum().item())
        if tot <= 0.0:
            return float("nan")
        p = (evals / tot)
        p = p[p > 0]
        return -float(torch.sum(p * torch.log(p)).item())

    def make_hook(layer_name):
        def hook(_m, _inp, out):
            nonlocal batch_grams
            act = out[0] if isinstance(out, (list, tuple)) else out
            if not torch.is_tensor(act):
                return
            with torch.no_grad():
                if act.dim() == 4:
                    N, C, H, W = act.shape
                    A = act.detach().to(torch.float32).view(N, C, H * W)
                    Gb = torch.einsum('ncs,nks->ck', A, A)  # CxC
                elif act.dim() == 2:
                    A = act.detach().to(torch.float32)
                    Gb = A.t().mm(A)  # CxC
                else:
                    return
                if batch_grams is None:
                    batch_grams = {}
                batch_grams[layer_name] = batch_grams.get(layer_name, 0) + Gb.to('cpu')

        return hook

    # Register hooks once
    handles = []
    for ln in layer_names:
        if ln in name2mod:
            handles.append(name2mod[ln].register_forward_hook(make_hook(ln)))

    try:
        with torch.no_grad():
            for b_idx, (x, _y) in enumerate(loader):
                batch_grams = {}

                _ = model(x.to(device, non_blocking=('cuda' in device.type)))

                # compute batch entropies per layer and update running stats
                for ln in layer_names:
                    G = batch_grams.get(ln, None)
                    if G is None:
                        continue
                    if eps > 0.0:
                        C = G.shape[0]
                        tr = float(torch.trace(G))
                        ridge = (eps * (tr / max(C, 1))) if C > 0 else 0.0
                        if ridge > 0.0:
                            G = G + ridge * torch.eye(C, dtype=G.dtype, device=G.device)
                    H = _shannon_entropy_from_gram(G)
                    if H == H:  # not NaN
                        sum_H[ln] += float(H)
                        sumsq_H[ln] += float(H) * float(H)
                        cnt_H[ln] += 1

                if (max_batches is not None and (b_idx + 1) >= max_batches) or debug:
                    break
    finally:
        for h in handles:
            h.remove()

    # Finalize
    entropies = OrderedDict()
    eranks = OrderedDict()
    entropies_std = OrderedDict() if return_std else None
    for ln in layer_names:
        if cnt_H[ln] > 0:
            mean_H = sum_H[ln] / cnt_H[ln]
            entropies[ln] = mean_H
            eranks[ln] = math.exp(mean_H)
            if return_std:
                var = max(sumsq_H[ln] / cnt_H[ln] - mean_H ** 2, 0.0)
                entropies_std[ln] = math.sqrt(var)
        else:
            entropies[ln] = float("nan")
            eranks[ln] = float("nan")
            if return_std:
                entropies_std[ln] = float("nan")

    return (entropies, eranks, entropies_std) if return_std else (entropies, eranks)


def _aggregate_pe_lists(pe_dict_list, ordered_names):
    """
    Convert a list of {layer_name: entropy} dicts into an ordered list of entropies
    (averaging across repeats), aligned with `ordered_names`.
    """
    if len(pe_dict_list) == 0:
        return [float("nan")] * len(ordered_names)
    vals = []
    for name in ordered_names:
        v = np.nanmean([d.get(name, np.nan) for d in pe_dict_list])
        vals.append(float(v) if not np.isnan(v) else float("nan"))
    return vals


def test_merge(origin_model, checkpoint, dataloader, train_loader, max_ratio, method, repair, di_samples_path,
               eval=True, measure_variance=False, device=None, repair_batch=4):
    input = torch.randn(1, 3, 224, 224).to(device=device)
    origin_model.to(device=device)
    origin_model.eval()

    origin_flop, origin_param = profile(origin_model, inputs=(input,), verbose=False)

    model = method(copy.deepcopy(origin_model), checkpoint, max_ratio=max_ratio, hooks=None, device=device)
    model.to(device=device)
    model.eval()
    flop, folded_param = profile(model, inputs=(input,), verbose=False)

    if repair != NO_REPAIR and repair != DF_REPAIR:
        if repair == DI_REPAIR:
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.reset_running_stats()
                    module.momentum = None

            model.train()
            model(torch.load(di_samples_path).to(device=device))
            model.eval()

        elif repair == REPAIR:
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.reset_running_stats()
                    module.momentum = None

            model.train()
            idx = 0
            for x, _ in train_loader:
                model(x.to(device=device))
                idx += 1
                if idx == repair_batch:
                    break
            model.eval()

    if eval is True:
        acc, loss = eval_model(model=model, dataloader=dataloader, device=device)
        return model, acc, 100.0 * (1.0 - (folded_param / origin_param))

    return model, 0.0, 100.0 * (1.0 - (folded_param / origin_param))


def set_gpu(gpu):
    if gpu == -1:
        print("Using CPU")
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu))
    else:
        print("Using CPU")
        device = torch.device("cpu")
    return device


def main(args):
    device = set_gpu(args.cuda_device)

    sar_entropy_loss = losses.SAR_EntropyLoss(margin_e0=0.4 * math.log(1000), fallback_on_empty=False)

    if args.debug:
        corruptions_robustbench = ["shot_noise"]
        set_individual_corruption(corruption_name="shot_noise")
    else:
        corruptions_robustbench = [
            "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
            "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
            "elastic_transform", "pixelate", "jpeg_compression"
        ]

    desc = {"experiment": args.exp_name.format(
        "pre_tta_metrics_" if args.pre_tta_metrics_computation else "post_tta_metrics_", )}
    if 'cuda' in device.type and not args.debug:
        wandb.login(key=args.wandb_key)
        wandb.init(
            project="{}{}_resnet18_tta_imagenetc".format("DEBUG_" if args.debug else "",
                                                         os.environ.get("SLURM_JOB_ID")),
            config=desc,
            name="{}{}_{}".format("DEBUG_" if args.debug else "", args.method, ""),
            group="resnet18_tta_imagenetc_folded",
        )

    set_imagenet_corruptions_data_handlers(
        severity=args.severity,
        corruptions=corruptions_robustbench,
        encoder_name=args.encoder_name
        , root_path=args.corruptions_root)

    compression_ratios = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

    results_per_ratio = {r: [[] for _ in range(len(corruptions_robustbench))] for r in compression_ratios}
    compressed_corruptions_acc_per_ratio = {r: [[] for _ in range(len(corruptions_robustbench))] for r in
                                            compression_ratios}
    compressed_val_set_acc_per_ratio = {r: [] for r in compression_ratios}
    sparsity_per_ratio = {r: None for r in compression_ratios}

    avg_loss_sharpness_pre_tta_per_ratio_corrupted_data = {ratio: [[] for _ in range(len(corruptions_robustbench))] for
                                                           ratio in
                                                           compression_ratios}
    avg_err_sharpness_pre_tta_per_ratio_corrupted_data = {ratio: [[] for _ in range(len(corruptions_robustbench))] for
                                                          ratio in
                                                          compression_ratios}

    avg_loss_sharpness_post_tta_per_ratio_corrupted_data = {ratio: [[] for _ in range(len(corruptions_robustbench))] for
                                                            ratio in
                                                            compression_ratios}
    avg_err_sharpness_post_tta_per_ratio_corrupted_data = {ratio: [[] for _ in range(len(corruptions_robustbench))] for
                                                           ratio in
                                                           compression_ratios}

    avg_loss_sharpness_pre_tta_per_ratio_training_data = {ratio: [[] for _ in range(len(corruptions_robustbench))] for
                                                          ratio in
                                                          compression_ratios}

    avg_err_sharpness_pre_tta_per_ratio_training_data = {ratio: [[] for _ in range(len(corruptions_robustbench))] for
                                                         ratio
                                                         in
                                                         compression_ratios}

    avg_loss_sharpness_post_tta_per_ratio_training_data = {ratio: [[] for _ in range(len(corruptions_robustbench))] for
                                                           ratio in
                                                           compression_ratios}
    avg_err_sharpness_post_tta_per_ratio_training_data = {ratio: [[] for _ in range(len(corruptions_robustbench))] for
                                                          ratio in
                                                          compression_ratios}

    # Prompt entropy storage (per-layer dicts)
    prompt_entropy_layers_pre = {r: [[] for _ in range(len(corruptions_robustbench))] for r in compression_ratios}
    prompt_entropy_layers_post = {r: [[] for _ in range(len(corruptions_robustbench))] for r in compression_ratios}
    prompt_entropy_layers_pre_train = {r: [] for r in compression_ratios}
    prompt_entropy_layers_post_train = {r: [[] for _ in range(len(corruptions_robustbench))] for r in
                                        compression_ratios}

    # Hessian and Fisher storage
    hess_pre_train_top2 = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}
    hess_post_train_top2 = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}
    hess_pre_corr_top2 = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}
    hess_post_corr_top2 = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}

    # ECE storage
    ece_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}
    ece_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}

    # Gradient SNR storage
    snr_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}
    snr_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}

    # Layer-wise Gradient SNR storage
    layerwise_snr_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                              compression_ratios}
    layerwise_snr_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                               compression_ratios}

    # Layer-wise Hessian Trace storage
    layerwise_hessian_trace_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                        compression_ratios}
    layerwise_hessian_trace_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                         compression_ratios}

    fisher_pre_train = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}
    fisher_post_train = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}
    fisher_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}
    fisher_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}

    # Fisher trace per-parameter (normalized by |U|)
    fisher_norm_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}
    fisher_norm_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}

    # Layer-wise Fisher storage
    layerwise_fisher_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}
    layerwise_fisher_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}

    # Gradient alignment storage
    grad_align_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                           compression_ratios}
    grad_align_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                            compression_ratios}

    # Gradient L2 norms (TTA and Oracle)
    grad_norm_tta_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                              compression_ratios}
    grad_norm_oracle_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                 compression_ratios}
    grad_norm_tta_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                               compression_ratios}
    grad_norm_oracle_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                  compression_ratios}

    # CKA storage
    cka_dense_vs_pruned_pre_tta_train = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                         compression_ratios}
    cka_dense_vs_pruned_pre_tta_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                        compression_ratios}
    cka_pruned_pre_vs_post_tta_train = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                        compression_ratios}
    cka_pruned_pre_vs_post_tta_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                       compression_ratios}

    dense_adapted_models_cka = []

    # Loss barrier storage (ID = clean validation, OOD = corruption data)
    # Max barrier values
    id_barrier_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                           compression_ratios}
    id_barrier_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                            compression_ratios}
    ood_barrier_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                            compression_ratios}
    ood_barrier_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                             compression_ratios}
    # Full barrier profiles (barriers_at_alphas - list of barrier values at each alpha in [0, 0.1, ..., 1.0])
    id_barrier_profile_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                   compression_ratios}
    id_barrier_profile_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                    compression_ratios}
    ood_barrier_profile_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                    compression_ratios}
    ood_barrier_profile_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                     compression_ratios}
    # Cache dense TTA state dicts per corruption for OOD barrier anchor
    dense_tta_state_dicts = {}

    print("Pretrained model: ", args.encoder_name, "TTA method: ", args.tta_method, "Method: ", args.method)
    print("Corruptions name and test-time evaluation order: ", corruptions_robustbench)
    print("Repair method: ", args.repair)
    print("Sharpness evaluation: ", args.sharpness)
    if args.sharpness:
        print("Number of batches for sharpness evaluation: ", args.eval_batches_num)
        print("Batch size for sharpness evaluation: ", args.m_sharpness)
        print("Prompt entropy evaluation: ", args.prompt_entropy)
    if args.prompt_entropy:
        print("PE granularity: ", args.pe_granularity, "Include stem: ", args.pe_include_stem)
        print("PE max batches: ", args.pe_max_batches)
    print("Hessian analysis: ", args.hessian_analysis)
    print("CKA analysis: ", args.cka_analysis)
    print("Prune data: ", args.prune_data)
    print("Fisher analysis: ", args.fisher_analysis)
    print("Gradient alignment: ", args.gradient_alignment)
    print("Pre-TTA metrics computation: ", args.pre_tta_metrics_computation)
    print("Post-TTA metrics computation: ", args.post_tta_metrics_computation)
    print("Debug mode: ", args.debug)

    if args.prune_data == 'test':
        if args.method not in ['wanda', 'taylor', 'hessian']:
            raise ValueError("Pruning with test data is only supported for Wanda, Taylor, and Hessian methods.")
        print("---------------- Pruning using TEST data (Corruptions) ----------------")
    else:
        print("---------------- Pruning using TRAIN data ----------------")

    for seed in [2020]:
        set_seed(seed=seed)

        model = resnet18(num_classes=1000).to(device=device)
        load_model(model, args.checkpoint, device=device)

        imagenet_c_dataset = ImageNet_C(encoder_name=args.encoder_name, args=args)
        model = model.to(device=device)

        test_loader = get_imagenet(datadir=args.dataset_root.format(os.path.dirname(os.getcwd())), train=False,
                                   bs=128)
        train_loader = get_imagenet(datadir=args.dataset_root.format(os.path.dirname(os.getcwd())), train=True,
                                    bs=64)

        method = merge_channel_ResNet18_big_clustering
        if args.repair == DF_REPAIR:
            fuse_bnorms_resnet(model, [2, 2, 2, 2], override=False)
            method = merge_channel_ResNet18_big_clustering_approx_repair

        for ratio in compression_ratios:

            if args.prune_data == 'train':
                if ratio == 0.0:
                    compressed_model = copy.deepcopy(model)
                    sparsity = 0.0
                    acc, _ = eval_model(model=compressed_model, dataloader=test_loader, device=device)
                else:
                    compressed_model, acc, sparsity = test_merge(
                        origin_model=copy.deepcopy(model),
                        checkpoint=copy.deepcopy(model).state_dict(),
                        dataloader=test_loader,
                        train_loader=train_loader,
                        max_ratio=ratio,
                        method=method,
                        repair=args.repair,
                        di_samples_path=args.di_samples_path,
                        device=device,
                        eval=True if not args.debug else False
                    )
                if sparsity_per_ratio[ratio] is None:
                    sparsity_per_ratio[ratio] = sparsity
                if acc is not None:
                    compressed_val_set_acc_per_ratio[ratio].append(acc * 100)
            else:
                compressed_model = copy.deepcopy(model)

            # Pre-adaptation (no TTA): prompt entropy on training data, then accuracy per corruption
            if args.prompt_entropy:
                ordered_names_train = get_resnet18_layernames(
                    compressed_model,
                    granularity=args.pe_granularity,
                    include_stem=args.pe_include_stem,
                    include_maxpool=False,
                    include_avgpool=args.pe_include_avgpool
                )
                pe_loader_train = get_imagenet(
                    datadir=args.dataset_root.format(os.path.dirname(os.getcwd())),
                    train=True,
                    bs=args.m_sharpness
                )

                ent_pre_train, _ = compute_prompt_entropy(
                    model=compressed_model,
                    loader=pe_loader_train,
                    device=device,
                    layer_names=tuple(ordered_names_train),
                    max_batches=args.pe_max_batches,
                    eps=1e-6
                )
                prompt_entropy_layers_pre_train[ratio].append(ent_pre_train)

            for corruption_idx, corruption in enumerate(corruptions_robustbench):
                if args.prune_data == 'test':
                    if ratio == 0.0:
                        compressed_model = copy.deepcopy(model)
                        sparsity = 0.0
                    else:
                        corruption_loader = imagenet_c_dataset.get_corruption_data_loader(
                            corruption=corruption, batch_size=args.m_sharpness, shuffle=True
                        )

                        # Prune using corruption data
                        compressed_model, acc, sparsity = test_merge(
                            origin_model=copy.deepcopy(model),
                            checkpoint=copy.deepcopy(model).state_dict(),
                            dataloader=test_loader,
                            train_loader=corruption_loader,
                            max_ratio=ratio,
                            method=method,
                            repair=args.repair,
                            di_samples_path=args.di_samples_path,
                            device=device,
                            eval=False
                        )

                    if sparsity_per_ratio[ratio] is None:
                        sparsity_per_ratio[ratio] = sparsity

                top1acc, top5_acc = imagenet_c_dataset.evaluate_model_all_corruption_data(
                    model=copy.deepcopy(compressed_model), device=device, batch_size=64,
                    corruption=corruption, args=args, compute_acc=True
                )
                compressed_corruptions_acc_per_ratio[ratio][corruption_idx].append(top1acc)

            # TTA per corruption, followed by post-TTA metrics
            for corruption_idx, corruption in enumerate(corruptions_robustbench):
                corruption_adapted_folded_model = copy.deepcopy(compressed_model)

                # Pre-TTA sharpness
                if args.sharpness and args.pre_tta_metrics_computation:
                    try:
                        pre_tta_model = copy.deepcopy(corruption_adapted_folded_model)
                        sharpness.configure_model_for_sar_sharpness(pre_tta_model)
                        pre_tta_model.eval()

                        batches_sharpness = imagenet_c_dataset.get_corruption_data_loader(
                            corruption=corruption, batch_size=args.m_sharpness, shuffle=False
                        )

                        sharpness_obj_pre_tta_corrupted_data, sharpness_err_pre_tta_corrupted_data, _, output_pre_tta = sharpness.eval_APGD_sharpness(
                            model=copy.deepcopy(pre_tta_model), device=device, batches=batches_sharpness,
                            loss_f=sar_entropy_loss, rho=0.002, n_iters=20, n_restarts=1, step_size_mult=1.0,
                            debug=args.debug, rand_init=False, no_grad_norm=False,
                            eval_batches_num=args.eval_batches_num if not args.debug else 2, verbose=False,
                            return_output=True, adaptive=True, version='default', norm='linf'
                        )
                        train_loader_sharpness = get_imagenet_sharpness_resnet18(
                            datadir=args.dataset_root.format(os.path.dirname(os.getcwd())),
                            train=True,
                            bs=args.m_sharpness)
                        sharpness_obj_pre_tta_training_data, sharpness_err_pre_tta_training_data, _, output_pre_tta_training_data = sharpness.eval_APGD_sharpness(
                            model=copy.deepcopy(pre_tta_model), device=device, batches=train_loader_sharpness,
                            loss_f=lambda logits, y: F.cross_entropy(logits, y, reduction='mean'),
                            rho=0.002, n_iters=20, n_restarts=1,
                            step_size_mult=1.0,
                            debug=args.debug,
                            rand_init=False, no_grad_norm=False,
                            eval_batches_num=args.eval_batches_num if not args.debug else 2,
                            verbose=False, return_output=True, adaptive=True, version='default',
                            norm='linf')
                        avg_loss_sharpness_pre_tta_per_ratio_training_data[ratio][corruption_idx].append(
                            float(sharpness_obj_pre_tta_training_data))
                        avg_err_sharpness_pre_tta_per_ratio_training_data[ratio][corruption_idx].append(
                            float(sharpness_err_pre_tta_training_data))
                        del train_loader_sharpness
                        avg_loss_sharpness_pre_tta_per_ratio_corrupted_data[ratio][corruption_idx].append(
                            float(sharpness_obj_pre_tta_corrupted_data))
                        avg_err_sharpness_pre_tta_per_ratio_corrupted_data[ratio][corruption_idx].append(
                            float(sharpness_err_pre_tta_corrupted_data))

                        del batches_sharpness
                        gc.collect()
                    except Exception as e:
                        print(f"Error computing pre-TTA sharpness: {e}")

                # Pre-TTA Hessian and Fisher
                if (args.hessian_analysis or args.fisher_analysis) and args.pre_tta_metrics_computation:
                    # Create loader before try block to ensure .transform is available
                    corruption_loader_pre_tta = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption,
                        batch_size=args.m_sharpness,
                        shuffle=True
                    )
                    try:
                        # Corruption data Hessian/Fisher

                        ds_corruption = pack_to_gpu_from_loader(
                            loader=corruption_loader_pre_tta,
                            max_batches=args.eval_batches_num if not args.debug else 2,
                            device=device
                        )

                        model_to_analyze_corr = copy.deepcopy(corruption_adapted_folded_model).to(device)
                        sharpness.configure_model_for_sar_sharpness(model_to_analyze_corr)
                        model_to_analyze_corr.eval()

                        # Enable grads only for normalization layers (affine) for Hessian/Fisher
                        for n, p in model_to_analyze_corr.named_parameters():
                            if "bn" in n or "norm" in n:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False

                        if args.hessian_analysis:
                            evals_corr, _ = get_hessian(device=device, model=copy.deepcopy(model_to_analyze_corr),
                                                        dataset=ds_corruption,
                                                        loss_fn=sar_entropy_loss,
                                                        neigs=1,
                                                        physical_batch_size=args.m_sharpness, exclude_ln=False, args=args,
                                                        preprocess=corruption_loader_pre_tta.transform)
                            hess_val_corr = evals_corr[0].item()
                            hess_pre_corr_top2[ratio][corruption_idx].append(hess_val_corr)


                        fisher_result_corr = compute_fisher_trace_subspace(
                            device=device, model=copy.deepcopy(model_to_analyze_corr),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(
                                corruption=corruption,
                                batch_size=args.m_sharpness,
                                shuffle=False
                            ).dataset,
                            loss_fn=sar_entropy_loss,
                            physical_batch_size=args.m_sharpness,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)

                        fisher_pre_corr[ratio][corruption_idx].append(fisher_result_corr["raw"])
                        fisher_norm_pre_corr[ratio][corruption_idx].append(fisher_result_corr["normalized"])

                        # Layer-wise Fisher (Pre-TTA)
                        lw_fisher_pre = compute_layerwise_fisher(
                            device=device, model=copy.deepcopy(corruption_adapted_folded_model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(
                                corruption=corruption,
                                batch_size=args.m_sharpness,
                                shuffle=False
                            ).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=500 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        layerwise_fisher_pre_corr[ratio][corruption_idx].append(
                            get_sorted_layer_fisher_list(lw_fisher_pre))

                        del ds_corruption, model_to_analyze_corr
                        gc.collect()

                        
                        # Corruption-data metrics
                        # Layer-wise Fisher (Pre-TTA)
                        lw_fisher_pre = compute_layerwise_fisher(
                            device=device, model=copy.deepcopy(compressed_model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(
                                corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                                persistent_workers=False, pin_memory=True, num_workers=7 if args.method != "fold" else 1
                            ).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=500 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        layerwise_fisher_pre_corr[ratio][corruption_idx].append(
                            get_sorted_layer_fisher_list(lw_fisher_pre))

                        # ECE (Pre-TTA)
                        ece_val = compute_ece(
                            device=device, model=copy.deepcopy(compressed_model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(
                                corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                                persistent_workers=False, pin_memory=True, num_workers=7 if args.method != "fold" else 1
                            ).dataset,
                            max_samples=500 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        ece_pre_corr[ratio][corruption_idx].append(ece_val)

                        # Gradient SNR (Pre-TTA)
                        snr_val = compute_gradient_snr(
                            device=device, model=copy.deepcopy(compressed_model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(
                                corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                                persistent_workers=False, pin_memory=True, num_workers=7 if args.method != "fold" else 1
                            ).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        snr_pre_corr[ratio][corruption_idx].append(snr_val)

                        # Layer-wise Gradient SNR (Pre-TTA)
                        lw_snr_val = compute_layerwise_gradient_snr(
                            device=device, model=copy.deepcopy(compressed_model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(
                                corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                                persistent_workers=False, pin_memory=True, num_workers=7 if args.method != "fold" else 1
                            ).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        layerwise_snr_pre_corr[ratio][corruption_idx].append(
                            get_sorted_layer_snr_list(lw_snr_val))

                        # Layer-wise Hessian Trace (Pre-TTA)
                        lw_hess_trace_val = compute_layerwise_hessian_trace(
                            device=device, model=copy.deepcopy(compressed_model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(
                                corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                                persistent_workers=False, pin_memory=True, num_workers=7 if args.method != "fold" else 1
                            ).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            n_hutchinson_iters=10 if not args.debug else 2,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        layerwise_hessian_trace_pre_corr[ratio][corruption_idx].append(
                            get_sorted_layer_hessian_trace_list(lw_hess_trace_val))

                        del corruption_loader_pre_tta
                        gc.collect()
                    except Exception as e:
                        print(f"Error computing pre-TTA Hessian/Fisher: {e}")

                # Pre-TTA gradient alignment (cosine similarity between TTA and oracle gradients)
                if args.gradient_alignment and args.pre_tta_metrics_computation:
                    corruption_loader_ga = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption,
                        batch_size=args.m_sharpness,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=1
                    )
                    try:
                        model_for_ga = copy.deepcopy(corruption_adapted_folded_model).to(device)
                        sharpness.configure_model_for_sar_sharpness(model_for_ga)

                        ga_result = compute_gradient_alignment(
                            device=device,
                            model=model_for_ga,
                            dataset=corruption_loader_ga.dataset,
                            tta_loss_fn=sar_entropy_loss,
                            oracle_loss_fn=F.cross_entropy,
                            physical_batch_size=args.m_sharpness,
                            max_samples=None,
                            args=args,
                            preprocess=corruption_loader_ga.transform,
                            match_tta_bn=True)
                        ga_val = ga_result['cosine_sim']
                        if math.isnan(ga_val):
                            grad_align_pre_corr[ratio][corruption_idx].append([])
                        else:
                            grad_align_pre_corr[ratio][corruption_idx].append(ga_val)

                        # Store gradient L2 norms
                        norm_tta_val = ga_result['norm_tta']
                        norm_oracle_val = ga_result['norm_oracle']
                        if math.isnan(norm_tta_val):
                            grad_norm_tta_pre_corr[ratio][corruption_idx].append([])
                        else:
                            grad_norm_tta_pre_corr[ratio][corruption_idx].append(norm_tta_val)
                        if math.isnan(norm_oracle_val):
                            grad_norm_oracle_pre_corr[ratio][corruption_idx].append([])
                        else:
                            grad_norm_oracle_pre_corr[ratio][corruption_idx].append(norm_oracle_val)

                        del corruption_loader_ga, model_for_ga
                        gc.collect()

                    except Exception as e:
                        print(f"Error computing pre-TTA Gradient Alignment: {e}")

                # Pre-TTA loss barriers (ID and OOD)
                if args.loss_barrier and args.pre_tta_metrics_computation:
                    try:
                        # Get dense pretrained state dict (θ₀)
                        dense_pretrained_sd = copy.deepcopy(model.state_dict())
                        # Get pruned pre-TTA state dict (φ_pre)
                        pruned_pre_tta_sd = copy.deepcopy(corruption_adapted_folded_model.state_dict())

                        # Compute dense TTA for this corruption (once per corruption, cached)
                        if corruption not in dense_tta_state_dicts:
                            dense_model_for_tta = copy.deepcopy(model).to(device)
                            optimizer_dense, _ = get_adaptation_optimizer_configure_model(
                                tta_method_name=args.tta_method,
                                model=dense_model_for_tta,
                                learning_rate=0.001 if args.tta_method == "tent" else 0.00025,
                                args=args, device=device
                            )
                            adapted_dense_model = learner.make(
                                name=args.tta_method,
                                model=dense_model_for_tta,
                                optimizer=optimizer_dense
                            )
                            _ = imagenet_c_dataset.evaluate_model_all_corruption_data(
                                model=adapted_dense_model, device=device,
                                batch_size=128 if args.tta_method == "tent" else 64,
                                corruption=corruption, args=args
                            )
                            # Restore BN stats for dense TTA model using corruption data
                            # (same data distribution the model was adapted on)
                            bn_calib_loader = imagenet_c_dataset.get_corruption_data_loader(
                                corruption=corruption, batch_size=args.m_sharpness, shuffle=True
                            )
                            restore_bn_running_stats(
                                model=adapted_dense_model.model,
                                calibration_loader=bn_calib_loader,
                                device=device,
                                num_batches=50
                            )
                            dense_tta_state_dicts[corruption] = copy.deepcopy(adapted_dense_model.model.state_dict())
                            del dense_model_for_tta, optimizer_dense, adapted_dense_model, bn_calib_loader
                            gc.collect()

                        # ID Barrier: B_ID(θ₀, θ̂_pre) on clean validation data
                        id_barrier_val, id_barrier_details = get_id_barrier(
                            model_dense=copy.deepcopy(model),
                            sd_dense_pretrained=dense_pretrained_sd,
                            sd_pruned=pruned_pre_tta_sd,
                            val_dataloader=test_loader,
                            loss_fn=F.cross_entropy,
                            n_alphas=11,
                            device=device,
                            max_batches=args.eval_batches_num if not args.debug else 2
                        )
                        id_barrier_pre_corr[ratio][corruption_idx].append(id_barrier_val)
                        id_barrier_profile_pre_corr[ratio][corruption_idx].append(id_barrier_details['barriers_at_alphas'])

                        # OOD Barrier: B_Qc(θ^post_dense(c), θ̂_pre) on corruption data
                        corruption_loader_barrier = imagenet_c_dataset.get_corruption_data_loader(
                            corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                            persistent_workers=False, pin_memory=True, num_workers=7
                        )
                        ood_barrier_val, ood_barrier_details = get_ood_barrier(
                            model_dense=copy.deepcopy(model),
                            sd_dense_tta=dense_tta_state_dicts[corruption],
                            sd_pruned=pruned_pre_tta_sd,
                            corruption_dataloader=corruption_loader_barrier,
                            loss_fn=F.cross_entropy,
                            n_alphas=11,
                            device=device,
                            max_batches=args.eval_batches_num if not args.debug else 2
                        )
                        ood_barrier_pre_corr[ratio][corruption_idx].append(ood_barrier_val)
                        ood_barrier_profile_pre_corr[ratio][corruption_idx].append(ood_barrier_details['barriers_at_alphas'])

                        del corruption_loader_barrier
                        gc.collect()

                    except Exception as e:
                        print(f"Error computing pre-TTA Loss Barrier: {e}")

                # Pre-TTA CKA
                if args.cka_analysis and args.pre_tta_metrics_computation:
                    try:
                        # Dense vs Pruned (Pre-TTA) on Corruption Data
                        cka_calc_corr = CKACalculator(model1=copy.deepcopy(model),
                                                      model2=copy.deepcopy(corruption_adapted_folded_model),
                                                      dataloader=imagenet_c_dataset.get_corruption_data_loader(
                                                          corruption=corruption,
                                                          batch_size=args.m_sharpness,
                                                          shuffle=False),
                                                      device=device,
                                                      num_epochs=1,
                                                      debug=args.debug,
                                                      is_main_process=True)
                        cka_matrix_corr = cka_calc_corr.calculate_cka_matrix()
                        cka_diagonal_corr = cka_matrix_corr.diagonal().tolist()
                        cka_dense_vs_pruned_pre_tta_corr[ratio][corruption_idx].append(cka_diagonal_corr)
                        cka_calc_corr.reset()
                        del cka_calc_corr
                        gc.collect()
                    except Exception as e:
                        print(f"Error computing pre-TTA CKA: {e}")

                # Pre-TTA prompt entropy
                if args.prompt_entropy:
                    prompt_entropy_model = copy.deepcopy(corruption_adapted_folded_model)
                    ordered_names = get_resnet18_layernames(
                        prompt_entropy_model,
                        granularity=args.pe_granularity,
                        include_stem=args.pe_include_stem,
                        include_maxpool=args.pe_include_maxpool,
                        include_avgpool=args.pe_include_avgpool,
                    )
                    pe_loader = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption, batch_size=256, shuffle=False
                    )
                    prompt_entropy_model.eval()
                    ent_pre, _ = compute_prompt_entropy(
                        model=prompt_entropy_model,
                        loader=pe_loader,
                        device=device,
                        layer_names=tuple(ordered_names),
                        max_batches=args.pe_max_batches,
                        eps=1e-6
                    )
                    prompt_entropy_layers_pre[ratio][corruption_idx].append(ent_pre)

                if args.pre_tta_only:
                    continue  # skip TTA adaptation and all post-TTA computation

                optimizer_corruption, spa_model = get_adaptation_optimizer_configure_model(
                    tta_method_name=args.tta_method,
                    model=corruption_adapted_folded_model,
                    learning_rate=0.00025,
                    args=args, device=device
                )

                # NORM is BP-free and already wrapped by get_adaptation_optimizer_configure_model
                if args.tta_method == "norm":
                    adapted_model_corruption = spa_model if spa_model is not None else corruption_adapted_folded_model
                else:
                    adapted_model_corruption = learner.make(
                        name=args.tta_method, model=corruption_adapted_folded_model, optimizer=optimizer_corruption
                    )
                top1acc, _ = imagenet_c_dataset.evaluate_model_all_corruption_data(
                    model=adapted_model_corruption, device=device, batch_size=64,
                    corruption=corruption, args=args
                )

                results_per_ratio[ratio][corruption_idx].append(top1acc)
                # Post-TTA sharpness
                if args.sharpness and args.post_tta_metrics_computation:
                    try:
                        post_tta_compressed_model = copy.deepcopy(adapted_model_corruption.model)
                        sharpness.configure_model_for_sar_sharpness(post_tta_compressed_model)
                        post_tta_compressed_model.eval()

                        batches_sharpness = imagenet_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                          batch_size=args.m_sharpness,
                                                                                          shuffle=False)

                        sharpness_obj_post_tta, sharpness_err_post_tta, _, output_post_tta = sharpness.eval_APGD_sharpness(
                            model=copy.deepcopy(post_tta_compressed_model), batches=batches_sharpness,
                            loss_f=sar_entropy_loss,
                            device=device,
                            rho=0.002, n_iters=20, n_restarts=1,
                            step_size_mult=1.0,
                            debug=args.debug,
                            eval_batches_num=args.eval_batches_num if not args.debug else 2,
                            rand_init=False, no_grad_norm=False,
                            verbose=False, return_output=True, adaptive=True, version='default',
                            norm='linf')

                        train_loader_sharpness = get_imagenet_sharpness_resnet18(
                            datadir=args.dataset_root.format(os.path.dirname(os.getcwd())),
                            train=True,
                            bs=args.m_sharpness)

                        sharpness_obj_post_tta_training_data, sharpness_err_post_tta_training_data, _, output_post_tta_training_data = sharpness.eval_APGD_sharpness(
                            model=copy.deepcopy(post_tta_compressed_model), device=device,
                            batches=train_loader_sharpness,
                            loss_f=lambda logits, y: F.cross_entropy(logits, y, reduction='mean'),
                            rho=0.002, n_iters=20, n_restarts=1,
                            step_size_mult=1.0,
                            debug=args.debug,
                            eval_batches_num=args.eval_batches_num if not args.debug else 2,
                            rand_init=False, no_grad_norm=False,
                            verbose=False, return_output=True, adaptive=True, version='default',
                            norm='linf')
                        avg_loss_sharpness_post_tta_per_ratio_training_data[ratio][corruption_idx].append(
                            float(sharpness_obj_post_tta_training_data))
                        avg_err_sharpness_post_tta_per_ratio_training_data[ratio][corruption_idx].append(
                            float(sharpness_err_post_tta_training_data))
                        del train_loader_sharpness
                        avg_loss_sharpness_post_tta_per_ratio_corrupted_data[ratio][corruption_idx].append(
                            float(sharpness_obj_post_tta))
                        avg_err_sharpness_post_tta_per_ratio_corrupted_data[ratio][corruption_idx].append(
                            float(sharpness_err_post_tta))

                        del batches_sharpness
                        gc.collect()
                    except Exception as e:
                        print(f"Error computing post-TTA sharpness: {e}")

                # Post-TTA Hessian and Fisher
                if (args.hessian_analysis or args.fisher_analysis) and args.post_tta_metrics_computation:
                    # Create loader before try block to ensure .transform is available
                    corruption_loader_post_tta = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption,
                        batch_size=args.m_sharpness,
                        shuffle=True
                    )
                    try:
                        # Corruption data Hessian/Fisher

                        ds_corruption = pack_to_gpu_from_loader(
                            loader=corruption_loader_post_tta,
                            max_batches=args.eval_batches_num if not args.debug else 2,
                            device=device
                        )

                        model_to_analyze_corr = copy.deepcopy(adapted_model_corruption.model).to(device)
                        sharpness.configure_model_for_sar_sharpness(model_to_analyze_corr)
                        model_to_analyze_corr.eval()

                        # Enable grads only for normalization layers (affine) for Hessian/Fisher
                        for n, p in model_to_analyze_corr.named_parameters():
                            if "bn" in n or "norm" in n:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False

                        if args.hessian_analysis:
                            evals_corr, _ = get_hessian(device=device, model=copy.deepcopy(model_to_analyze_corr),
                                                        dataset=ds_corruption,
                                                        loss_fn=sar_entropy_loss,
                                                        neigs=1,
                                                        physical_batch_size=args.m_sharpness, exclude_ln=False, args=args,
                                                        preprocess=corruption_loader_post_tta.transform)
                            hess_val_corr = evals_corr[0].item()
                            hess_post_corr_top2[ratio][corruption_idx].append(hess_val_corr)


                        fisher_result_corr = compute_fisher_trace_subspace(
                            device=device, model=copy.deepcopy(model_to_analyze_corr),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                  batch_size=args.m_sharpness,
                                                                                  shuffle=False).dataset,
                            loss_fn=sar_entropy_loss,
                            physical_batch_size=args.m_sharpness,
                            args=args,
                            preprocess=corruption_loader_post_tta.transform)

                        fisher_post_corr[ratio][corruption_idx].append(fisher_result_corr["raw"])
                        fisher_norm_post_corr[ratio][corruption_idx].append(fisher_result_corr["normalized"])

                        # Layer-wise Fisher (Post-TTA)
                        lw_fisher_post = compute_layerwise_fisher(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                  batch_size=args.m_sharpness,
                                                                                  shuffle=False).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=500 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_post_tta.transform)
                        layerwise_fisher_post_corr[ratio][corruption_idx].append(
                            get_sorted_layer_fisher_list(lw_fisher_post))

                        # ECE (Post-TTA)
                        ece_val_post = compute_ece(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                  batch_size=args.m_sharpness,
                                                                                  shuffle=False).dataset,
                            max_samples=500 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_post_tta.transform)
                        ece_post_corr[ratio][corruption_idx].append(ece_val_post)

                        # Gradient SNR (Post-TTA)
                        snr_val_post = compute_gradient_snr(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                  batch_size=args.m_sharpness,
                                                                                  shuffle=False).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_post_tta.transform)
                        snr_post_corr[ratio][corruption_idx].append(snr_val_post)

                        # Layer-wise Gradient SNR (Post-TTA)
                        lw_snr_val_post = compute_layerwise_gradient_snr(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(
                                corruption=corruption, batch_size=args.m_sharpness, shuffle=False).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_post_tta.transform)
                        layerwise_snr_post_corr[ratio][corruption_idx].append(
                            get_sorted_layer_snr_list(lw_snr_val_post))

                        # Layer-wise Hessian Trace (Post-TTA)
                        lw_hess_trace_val_post = compute_layerwise_hessian_trace(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(
                                corruption=corruption, batch_size=args.m_sharpness, shuffle=False).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            n_hutchinson_iters=10 if not args.debug else 2,
                            args=args,
                            preprocess=corruption_loader_post_tta.transform)
                        layerwise_hessian_trace_post_corr[ratio][corruption_idx].append(
                            get_sorted_layer_hessian_trace_list(lw_hess_trace_val_post))

                        del corruption_loader_post_tta, ds_corruption, model_to_analyze_corr
                        gc.collect()
                    except Exception as e:
                        print(f"Error computing post-TTA Hessian/Fisher: {e}")

                # Post-TTA gradient alignment (cosine similarity between TTA and oracle gradients)
                if args.gradient_alignment and args.post_tta_metrics_computation:
                    corruption_loader_ga_post = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption,
                        batch_size=args.m_sharpness,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=1
                    )
                    try:
                        model_for_ga_post = copy.deepcopy(adapted_model_corruption.model).to(device)
                        sharpness.configure_model_for_sar_sharpness(model_for_ga_post)

                        ga_result_post = compute_gradient_alignment(
                            device=device,
                            model=model_for_ga_post,
                            dataset=corruption_loader_ga_post.dataset,
                            tta_loss_fn=sar_entropy_loss,
                            oracle_loss_fn=F.cross_entropy,
                            physical_batch_size=args.m_sharpness,
                            max_samples=None,
                            args=args,
                            preprocess=corruption_loader_ga_post.transform,
                            match_tta_bn=True)
                        ga_val_post = ga_result_post['cosine_sim']
                        if math.isnan(ga_val_post):
                            grad_align_post_corr[ratio][corruption_idx].append([])
                        else:
                            grad_align_post_corr[ratio][corruption_idx].append(ga_val_post)

                        # Store gradient L2 norms
                        norm_tta_val_post = ga_result_post['norm_tta']
                        norm_oracle_val_post = ga_result_post['norm_oracle']
                        if math.isnan(norm_tta_val_post):
                            grad_norm_tta_post_corr[ratio][corruption_idx].append([])
                        else:
                            grad_norm_tta_post_corr[ratio][corruption_idx].append(norm_tta_val_post)
                        if math.isnan(norm_oracle_val_post):
                            grad_norm_oracle_post_corr[ratio][corruption_idx].append([])
                        else:
                            grad_norm_oracle_post_corr[ratio][corruption_idx].append(norm_oracle_val_post)

                        del corruption_loader_ga_post, model_for_ga_post
                        gc.collect()

                    except Exception as e:
                        print(f"Error computing post-TTA Gradient Alignment: {e}")

                # Post-TTA loss barriers (ID and OOD)
                if args.loss_barrier and args.post_tta_metrics_computation:
                    try:
                        # Get dense pretrained state dict (θ₀)
                        dense_pretrained_sd = copy.deepcopy(model.state_dict())
                        
                        # Re-estimate BN running statistics of the adapted model on corruption data
                        model_for_bn_restore = copy.deepcopy(adapted_model_corruption.model).to(device)
                        bn_calib_loader_pruned = imagenet_c_dataset.get_corruption_data_loader(
                            corruption=corruption, batch_size=args.m_sharpness, shuffle=True
                        )
                        restore_bn_running_stats(
                            model=model_for_bn_restore,
                            calibration_loader=bn_calib_loader_pruned,
                            device=device,
                            num_batches=50
                        )
                        # Get pruned post-TTA state dict (φ_post) with restored BN stats
                        pruned_post_tta_sd = copy.deepcopy(model_for_bn_restore.state_dict())
                        del model_for_bn_restore, bn_calib_loader_pruned
                        gc.collect()

                        # Dense TTA should already be cached from pre-TTA computation
                        if corruption not in dense_tta_state_dicts:
                            dense_model_for_tta = copy.deepcopy(model).to(device)
                            optimizer_dense, _ = get_adaptation_optimizer_configure_model(
                                tta_method_name=args.tta_method,
                                model=dense_model_for_tta,
                                learning_rate=0.001 if args.tta_method == "tent" else 0.00025,
                                args=args, device=device
                            )
                            adapted_dense_model = learner.make(
                                name=args.tta_method,
                                model=dense_model_for_tta,
                                optimizer=optimizer_dense
                            )
                            _ = imagenet_c_dataset.evaluate_model_all_corruption_data(
                                model=adapted_dense_model, device=device,
                                batch_size=128 if args.tta_method == "tent" else 64,
                                corruption=corruption, args=args
                            )
                            # Restore BN stats for dense TTA model using corruption data
                            bn_calib_loader_dense = imagenet_c_dataset.get_corruption_data_loader(
                                corruption=corruption, batch_size=args.m_sharpness, shuffle=True
                            )
                            restore_bn_running_stats(
                                model=adapted_dense_model.model,
                                calibration_loader=bn_calib_loader_dense,
                                device=device,
                                num_batches=50
                            )
                            dense_tta_state_dicts[corruption] = copy.deepcopy(adapted_dense_model.model.state_dict())
                            del dense_model_for_tta, optimizer_dense, adapted_dense_model, bn_calib_loader_dense
                            gc.collect()

                        # ID Barrier: B_ID(θ₀, θ̂_post) on clean validation data
                        id_barrier_val, id_barrier_details = get_id_barrier(
                            model_dense=copy.deepcopy(model),
                            sd_dense_pretrained=dense_pretrained_sd,
                            sd_pruned=pruned_post_tta_sd,
                            val_dataloader=test_loader,
                            loss_fn=F.cross_entropy,
                            n_alphas=11,
                            device=device,
                            max_batches=args.eval_batches_num if not args.debug else 2
                        )
                        id_barrier_post_corr[ratio][corruption_idx].append(id_barrier_val)
                        id_barrier_profile_post_corr[ratio][corruption_idx].append(id_barrier_details['barriers_at_alphas'])

                        # OOD Barrier: B_Qc(θ^post_dense(c), θ̂_post) on corruption data
                        corruption_loader_barrier = imagenet_c_dataset.get_corruption_data_loader(
                            corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                            persistent_workers=False, pin_memory=True, num_workers=7
                        )
                        ood_barrier_val, ood_barrier_details = get_ood_barrier(
                            model_dense=copy.deepcopy(model),
                            sd_dense_tta=dense_tta_state_dicts[corruption],
                            sd_pruned=pruned_post_tta_sd,
                            corruption_dataloader=corruption_loader_barrier,
                            loss_fn=F.cross_entropy,
                            n_alphas=11,
                            device=device,
                            max_batches=args.eval_batches_num if not args.debug else 2
                        )
                        ood_barrier_post_corr[ratio][corruption_idx].append(ood_barrier_val)
                        ood_barrier_profile_post_corr[ratio][corruption_idx].append(ood_barrier_details['barriers_at_alphas'])

                        del corruption_loader_barrier
                        gc.collect()

                    except Exception as e:
                        print(f"Error computing post-TTA Loss Barrier: {e}")

                # Post-TTA CKA
                if args.cka_analysis and args.post_tta_metrics_computation:
                    try:

                        if ratio == 0.0:
                            dense_adapted_models_cka.append(copy.deepcopy(adapted_model_corruption.model))

                        # Pruned Pre-TTA vs Post-TTA on Corruption Data
                        cka_calc_corr = CKACalculator(model1=copy.deepcopy(dense_adapted_models_cka[corruption_idx]),
                                                      model2=copy.deepcopy(adapted_model_corruption.model),
                                                      dataloader=imagenet_c_dataset.get_corruption_data_loader(
                                                          corruption=corruption,
                                                          batch_size=args.m_sharpness,
                                                          shuffle=False),
                                                      device=device,
                                                      num_epochs=1,
                                                      debug=args.debug,
                                                      is_main_process=True)
                        cka_matrix_corr = cka_calc_corr.calculate_cka_matrix()
                        cka_diagonal_corr = cka_matrix_corr.diagonal().tolist()
                        cka_pruned_pre_vs_post_tta_corr[ratio][corruption_idx].append(cka_diagonal_corr)
                        cka_calc_corr.reset()
                        del cka_calc_corr
                        gc.collect()
                    except Exception as e:
                        print(f"Error computing post-TTA CKA: {e}")

                # Post-TTA prompt entropy

                if args.prompt_entropy:
                    adapted_model_corruption.model.eval()
                    pe_loader = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption, batch_size=args.m_sharpness, shuffle=False
                    )
                    ordered_names = get_resnet18_layernames(
                        adapted_model_corruption.model,
                        granularity=args.pe_granularity,
                        include_stem=args.pe_include_stem,
                        include_maxpool=args.pe_include_maxpool,
                        include_avgpool=args.pe_include_avgpool,
                    )
                    ent_post, _ = compute_prompt_entropy(
                        model=adapted_model_corruption.model,
                        loader=pe_loader,
                        device=device,
                        layer_names=tuple(ordered_names),
                        max_batches=args.pe_max_batches,
                        eps=1e-6
                    )
                    prompt_entropy_layers_post[ratio][corruption_idx].append(ent_post)

    # Aggregate and print results
    try:
        acc_pre_map = format_metric_map(compressed_corruptions_acc_per_ratio, compression_ratios,
                                        corruptions_robustbench)
        print(
            f"resnet18_imagenet_{args.method}_calibration{args.prune_data}_pre_adaptations_map_accuracy_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(acc_pre_map)}")

        acc_post_map = format_metric_map(results_per_ratio, compression_ratios, corruptions_robustbench)
        print(
            f"resnet18_imagenet_{args.method}_calibration{args.prune_data}_post_adaptations_map_accuracy_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(acc_post_map)}")
    except Exception as e:
        print(f"Error printing accuracy maps: {e}")

    # Sharpness
    if args.sharpness and args.pre_tta_metrics_computation:
        try:
            # Pre-TTA Loss
            sharp_loss_pre_map = format_metric_map(avg_loss_sharpness_pre_tta_per_ratio_corrupted_data,
                                                   compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration{args.prune_data}_pre_adaptations_map_sharpness_loss_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(sharp_loss_pre_map)}")

            # Pre-TTA Loss Training
            sharp_loss_training_pre_map = format_metric_map(avg_loss_sharpness_pre_tta_per_ratio_training_data,
                                                            compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration{args.prune_data}_pre_adaptations_map_sharpness_training_loss_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(sharp_loss_training_pre_map)}")

        except Exception as e:
            print(f"Error printing pre-TTA sharpness maps: {e}")

    if args.sharpness and args.post_tta_metrics_computation:
        try:
            # Post-TTA Loss
            sharp_loss_post_map = format_metric_map(avg_loss_sharpness_post_tta_per_ratio_corrupted_data,
                                                    compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration{args.prune_data}_post_adaptations_map_sharpness_loss_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(sharp_loss_post_map)}")

            # Post-TTA Loss Training
            sharp_loss_training_post_map = format_metric_map(avg_loss_sharpness_post_tta_per_ratio_training_data,
                                                             compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration{args.prune_data}_post_adaptations_map_sharpness_training_loss_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(sharp_loss_training_post_map)}")

        except Exception as e:
            print(f"Error printing post-TTA sharpness maps: {e}")

    # Prompt entropy
    if args.prompt_entropy and args.pre_tta_metrics_computation:
        try:
            # Pre-TTA (Corrupted)
            pe_pre_map = format_metric_map(prompt_entropy_layers_pre, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration{args.prune_data}_pre_adaptations_map_prompt_entropy_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(pe_pre_map)}")

        except Exception as e:
            print(f"Error printing pre-TTA prompt entropy maps: {e}")

    if args.prompt_entropy and args.post_tta_metrics_computation:
        try:
            # Post-TTA (Corrupted)
            pe_post_map = format_metric_map(prompt_entropy_layers_post, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration{args.prune_data}_post_adaptations_map_prompt_entropy_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(pe_post_map)}")

        except Exception as e:
            print(f"Error printing post-TTA prompt entropy maps: {e}")

    # Hessian and Fisher
    if args.hessian_analysis and args.pre_tta_metrics_computation:
        try:
            # Pre-TTA Hessian
            hess_pre_corr_map = format_metric_map(hess_pre_corr_top2, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_hessian_eigenvalues_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(hess_pre_corr_map)}")
        except Exception as e:
            print(f"Error printing pre-TTA Hessian maps: {e}")

    if args.fisher_analysis and args.pre_tta_metrics_computation:
        try:
            # Pre-TTA Fisher
            fisher_pre_corr_map = format_metric_map(fisher_pre_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_fisher_trace_in_U_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_pre_corr_map)}")

            # Pre-TTA Normalized Fisher (per-parameter)
            fisher_norm_pre_corr_map = format_metric_map(fisher_norm_pre_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_fisher_trace_normalized_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_norm_pre_corr_map)}")
            # Pre-TTA Layer-wise Fisher
            lw_fisher_pre_corr_map = format_metric_map(layerwise_fisher_pre_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_layerwise_fisher_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_fisher_pre_corr_map)}")

            # Pre-TTA ECE
            ece_pre_corr_map = format_metric_map(ece_pre_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_ece_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ece_pre_corr_map)}")

            # Pre-TTA Gradient SNR
            snr_pre_corr_map = format_metric_map(snr_pre_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_gradient_snr_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(snr_pre_corr_map)}")

            # Pre-TTA Layer-wise Gradient SNR
            lw_snr_pre_corr_map = format_metric_map(layerwise_snr_pre_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_layerwise_gradient_snr_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_snr_pre_corr_map)}")

            # Pre-TTA Layer-wise Hessian Trace
            lw_hess_trace_pre_corr_map = format_metric_map(layerwise_hessian_trace_pre_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_layerwise_hessian_trace_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_hess_trace_pre_corr_map)}")
        except Exception as e:
            print(f"Error printing pre-TTA Fisher maps: {e}")

    if args.gradient_alignment and args.pre_tta_metrics_computation:
        try:
            ga_pre_corr_map = format_metric_map(grad_align_pre_corr, compression_ratios,
                                                corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_gradient_alignment_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ga_pre_corr_map)}")
        except Exception as e:
            print(f"Error printing pre-TTA Gradient Alignment maps: {e}")

        try:
            gn_tta_pre_map = format_metric_map(grad_norm_tta_pre_corr, compression_ratios,
                                               corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_grad_norm_tta_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(gn_tta_pre_map)}")
            gn_oracle_pre_map = format_metric_map(grad_norm_oracle_pre_corr, compression_ratios,
                                                  corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_grad_norm_oracle_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(gn_oracle_pre_map)}")
        except Exception as e:
            print(f"Error printing pre-TTA Gradient Norm maps: {e}")

    if args.hessian_analysis and args.post_tta_metrics_computation:
        try:
            # Post-TTA Hessian
            hess_post_corr_map = format_metric_map(hess_post_corr_top2, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_hessian_eigenvalues_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(hess_post_corr_map)}")
        except Exception as e:
            print(f"Error printing post-TTA Hessian maps: {e}")

    if args.fisher_analysis and args.post_tta_metrics_computation:
        try:
            # Post-TTA Fisher
            fisher_post_corr_map = format_metric_map(fisher_post_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_fisher_trace_in_U_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_post_corr_map)}")

            # Post-TTA Normalized Fisher (per-parameter)
            fisher_norm_post_corr_map = format_metric_map(fisher_norm_post_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_fisher_trace_normalized_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_norm_post_corr_map)}")

            # Post-TTA Layer-wise Fisher
            lw_fisher_post_corr_map = format_metric_map(layerwise_fisher_post_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_layerwise_fisher_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_fisher_post_corr_map)}")
        except Exception as e:
            print(f"Error printing post-TTA Fisher maps: {e}")

        # Post-TTA ECE
        try:
            ece_post_corr_map = format_metric_map(ece_post_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_ece_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ece_post_corr_map)}")

            # Post-TTA Gradient SNR
            snr_post_corr_map = format_metric_map(snr_post_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_gradient_snr_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(snr_post_corr_map)}")

            # Post-TTA Layer-wise Gradient SNR
            lw_snr_post_corr_map = format_metric_map(layerwise_snr_post_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_layerwise_gradient_snr_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_snr_post_corr_map)}")

            # Post-TTA Layer-wise Hessian Trace
            lw_hess_trace_post_corr_map = format_metric_map(layerwise_hessian_trace_post_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_layerwise_hessian_trace_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_hess_trace_post_corr_map)}")
        except Exception as e:
            print(f"Error printing post-TTA ECE/SNR maps: {e}")

    if args.gradient_alignment and args.post_tta_metrics_computation:
        try:
            ga_post_corr_map = format_metric_map(grad_align_post_corr, compression_ratios,
                                                 corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_gradient_alignment_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ga_post_corr_map)}")
        except Exception as e:
            print(f"Error printing post-TTA Gradient Alignment maps: {e}")

        try:
            gn_tta_post_map = format_metric_map(grad_norm_tta_post_corr, compression_ratios,
                                                corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_grad_norm_tta_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(gn_tta_post_map)}")
            gn_oracle_post_map = format_metric_map(grad_norm_oracle_post_corr, compression_ratios,
                                                   corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_grad_norm_oracle_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(gn_oracle_post_map)}")
        except Exception as e:
            print(f"Error printing post-TTA Gradient Norm maps: {e}")

    # CKA
    if args.cka_analysis and args.pre_tta_metrics_computation:
        try:
            # Pre-TTA CKA
            cka_pre_corr_map = format_metric_map(cka_dense_vs_pruned_pre_tta_corr, compression_ratios,
                                                 corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_cka_dense_vs_pruned_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(cka_pre_corr_map)}")

        except Exception as e:
            print(f"Error printing pre-TTA CKA maps: {e}")

    if args.cka_analysis and args.post_tta_metrics_computation:
        try:
            # Post-TTA CKA
            cka_post_corr_map = format_metric_map(cka_pruned_pre_vs_post_tta_corr, compression_ratios,
                                                  corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_cka_pruned_pre_vs_post_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(cka_post_corr_map)}")

        except Exception as e:
            print(f"Error printing post-TTA CKA maps: {e}")

    # Loss barriers (ID and OOD)
    if args.loss_barrier:
        try:
            # Pre-TTA ID Barrier
            id_barrier_pre_map = format_metric_map(id_barrier_pre_corr, compression_ratios,
                                                   corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_id_barrier_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(id_barrier_pre_map)}")

            # Pre-TTA OOD Barrier
            ood_barrier_pre_map = format_metric_map(ood_barrier_pre_corr, compression_ratios,
                                                    corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_ood_barrier_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ood_barrier_pre_map)}")

            # Post-TTA ID Barrier
            id_barrier_post_map = format_metric_map(id_barrier_post_corr, compression_ratios,
                                                    corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_id_barrier_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(id_barrier_post_map)}")

            # Post-TTA OOD Barrier
            ood_barrier_post_map = format_metric_map(ood_barrier_post_corr, compression_ratios,
                                                     corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_ood_barrier_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ood_barrier_post_map)}")

            # Full Barrier Profiles (barriers at each alpha in [0, 0.1, ..., 1.0])
            # Pre-TTA ID Barrier Profile
            id_barrier_profile_pre_map = format_metric_map(id_barrier_profile_pre_corr, compression_ratios,
                                                           corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_id_barrier_profile_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(id_barrier_profile_pre_map)}")

            # Pre-TTA OOD Barrier Profile
            ood_barrier_profile_pre_map = format_metric_map(ood_barrier_profile_pre_corr, compression_ratios,
                                                            corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_ood_barrier_profile_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ood_barrier_profile_pre_map)}")

            # Post-TTA ID Barrier Profile
            id_barrier_profile_post_map = format_metric_map(id_barrier_profile_post_corr, compression_ratios,
                                                            corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_id_barrier_profile_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(id_barrier_profile_post_map)}")

            # Post-TTA OOD Barrier Profile
            ood_barrier_profile_post_map = format_metric_map(ood_barrier_profile_post_corr, compression_ratios,
                                                             corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_ood_barrier_profile_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ood_barrier_profile_post_map)}")

        except Exception as e:
            print(f"Error printing Loss Barrier maps: {e}")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet18 ImageNet TTA Clustering")
    parser.add_argument("--corruptions_root", default="{}/Few-Shot-Adaptation-Learning/datasets/imagenet-c/{}/{}",
                        type=str)
    parser.add_argument("--dataset_root", default="{}/REDS-Resource-Efficient-Deep-Subnetworks/datasets/imagenet",
                        type=str)
    parser.add_argument('--cuda_device', type=int, default=-1, help='gpu device number')
    parser.add_argument("--checkpoint", type=str, default="pretrained/resnet18_imagenet.pth")
    parser.add_argument("--repair", type=str, default="REPAIR", help="")
    parser.add_argument("--wandb_key", type=str, help="wandb key for login",
                        default=os.environ.get("WANDB_API_KEY"))
    parser.add_argument("--proj_name", type=str, default="{}_fold_resnet18_imagenet")
    parser.add_argument("--exp_name", type=str, default="fold_{}")
    parser.add_argument('--encoder_name', default='resnet18', type=str)
    parser.add_argument('--dataset_name', default='imagenet', type=str)
    parser.add_argument('--eval_batches_num', default=32, type=int)
    parser.add_argument('--m_sharpness', default=64, type=int)
    parser.add_argument('--severity', default=5, type=int)
    parser.add_argument('--no_adapt', default=False, action='store_true', help='Do not adapt the model at test time')
    parser.add_argument('--sharpness', default=False, action='store_true',
                        help='Compute sharpness of the loss landscape')
    parser.add_argument('--not_reset_statistics', default=False, action='store_true',
                        help='Do not reset BN stats after repair')
    parser.add_argument('--tta_method', default='tent', type=str)
    parser.add_argument('--repair_test_data', default=False, action='store_true', help='Apply repair using test data')
    parser.add_argument("--di_samples_path", type=str, default="resnet_imagenet_di.pt")
    parser.add_argument('--debug', default=False, action='store_true', help='Debug evaluation')

    # Prompt-entropy flags
    parser.add_argument('--prompt_entropy', default=False, action='store_true',
                        help='Compute Gram-based prompt entropy (pre/post TTA) with batch_size=256')
    parser.add_argument('--pe_max_batches', type=int, default=20,
                        help='Accumulate prompt-entropy over the first N batches (e.g., 20).')
    parser.add_argument('--pe_granularity', default='block', choices=['block', 'conv'],
                        help="Hook BasicBlock outputs ('block') or internal convs ('conv').")
    parser.add_argument('--pe_include_stem', default=False, action='store_true',
                        help='Include the stem output (model.relu).')
    parser.add_argument('--pe_include_maxpool', default=False, action='store_true',
                        help='Include maxpool activations.')
    parser.add_argument('--pe_include_avgpool', default=False, action='store_true',
                        help='Include avgpool activations.')
    parser.add_argument('--prune_data', default='train', choices=['train', 'test'],
                        help='Data to use for pruning calibration: "train" (default) or "test" (corruption data).')

    # Analysis control flags
    parser.add_argument('--method', type=str, default='fold', choices=['fold'],
                        help='Clustering/folding method to use')
    parser.add_argument('--hessian_analysis', default=False, action='store_true',
                        help='Compute Hessian eigenvalues')
    parser.add_argument('--fisher_analysis', default=False, action='store_true',
                        help='Compute Fisher analysis.')
    parser.add_argument('--loss_barrier', default=False, action='store_true',
                        help='Compute ID and OOD loss barriers for basin connectivity analysis')
    parser.add_argument('--gradient_alignment', default=False, action='store_true',
                        help='Compute gradient cosine similarity between TTA and Oracle gradients')
    parser.add_argument('--pre_tta_only', default=False, action='store_true',
                        help='Skip TTA adaptation and all post-TTA computation. '
                             'Use to compute only pre-TTA metrics (e.g., pre-TTA gradient alignment) '
                             'without the cost of running the full adaptation loop.')
    parser.add_argument('--cka_analysis', default=False, action='store_true',
                        help='Compute CKA (Centered Kernel Alignment) analysis')
    parser.add_argument('--pre_tta_metrics_computation', default=False, action='store_true',
                        help='Compute metrics before Test-Time Adaptation')
    parser.add_argument('--post_tta_metrics_computation', default=False, action='store_true',
                        help='Compute metrics after Test-Time Adaptation')

    args = parser.parse_args()
    main(args=args)
