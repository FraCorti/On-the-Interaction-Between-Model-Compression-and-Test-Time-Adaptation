import os
import argparse
import math
from collections import OrderedDict
import torch
import gc
from compression.fold import ResNet18_ModelFolding
from utils.custom_structural_pruning import one_shot_structural_pruning_resnet18
from utils.compute_hessian_vision import pack_to_gpu_from_loader, get_hessian, compute_fisher_trace_subspace, compute_ece, compute_gradient_snr, compute_layerwise_gradient_snr, get_sorted_layer_snr_list, compute_layerwise_hessian_trace, get_sorted_layer_hessian_trace_list
from utils.fisher_layerwise import compute_layerwise_fisher, get_sorted_layer_fisher_list
from utils.gradient_alignment import compute_gradient_alignment
from utils.loss_barrier import get_id_barrier, get_ood_barrier, inflate_state_dict, restore_bn_running_stats
from utils.format_results import format_numpy_floats_map, format_map_results, format_metric_map
from cka import CKACalculator, _HOOK_LAYER_TYPES
from utils.utils import eval_model, load_model
from model_cifar10.resnet import ResNet18_Custom, BasicBlock_Custom

# Extend default CKA hook types to include CIFAR-10 custom BasicBlock
CIFAR10_CKA_HOOK_LAYER_TYPES = _HOOK_LAYER_TYPES + (BasicBlock_Custom,)
from utils.seed import set_seed
from compression.mag_prune import ResNet18_MagnitudePruning
from utils.cifar10_c import set_individual_corruption, Cifar10_C, set_cifar10_corruptions_data_handlers
from compression.wanda import ResNet18_WandaPruning
from compression.rand_prune import ResNet18_RandomPruning
from compression.singleton import ResNet18_Singleton
import wandb
from utils.datasets import get_cifar10, get_cifar10_sharpness
import copy
import sharpness
from utils import learner, losses
import torch.nn.functional as F
from utils.learner.set_optimizer import get_adaptation_optimizer_configure_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


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


def get_resnet18_layernames(model, granularity="block",
                            include_stem=False):
    """
    Enumerate layers to hook for CIFAR-ResNet18 (custom) or similar.
    - 'block': hook the BasicBlock outputs (post-residual add + final ReLU).
    - 'conv' : hook conv1/conv2 inside each BasicBlock (pre-residual).
    Stem name for CIFAR model is 'relu_conv'; if absent, fallback to 'relu'.
    If the backbone is wrapped (e.g. SPA's BYOLWrapper exposes it as `.model`),
    hook names are prefixed with "model." to match the wrapper's `named_modules()` keys.
    """
    inner = model
    prefix = ""
    if (not hasattr(model, "layer1")) and hasattr(model, "model") \
            and hasattr(model.model, "layer1"):
        inner = model.model
        prefix = "model."

    names = []
    if include_stem:
        stem_name = "relu_conv" if hasattr(inner, "relu_conv") else "relu"
        names.append(f"{prefix}{stem_name}")

    for k in range(1, 5):
        layer = getattr(inner, f"layer{k}")
        for i in range(len(layer)):
            if granularity == "block":
                names.append(f"{prefix}layer{k}.{i}")  # BasicBlock output (post-add, post-ReLU)
            elif granularity == "conv":
                names.append(f"{prefix}layer{k}.{i}.conv1")
                names.append(f"{prefix}layer{k}.{i}.conv2")
            else:
                raise ValueError("granularity must be 'block' or 'conv'")
    # CIFAR variant uses functional avgpool; nothing to hook there
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
    Paper-style (batch_mean) prompt entropy per analyzed layer.

    For each batch:
      1) build a per-layer Gram G_b,l from that batch alone
      2) compute Shannon entropy H_b,l from normalized eigvals of G_b,l
    Then average H_b,l across batches b (up to `max_batches`) to get H_l.

    Args:
        model: CNN (custom ResNet18 for CIFAR-10).
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

    # Hooks are registered once; batch_grams is rebound for every batch
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

                # per-layer batch entropies and running stats
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


def _run_for_checkpoint(args):
    """Run full experiment for a single checkpoint. Returns result dicts for aggregation."""
    device = set_gpu(gpu=args.cuda_device)

    sar_entropy_loss = losses.SAR_EntropyLoss(margin_e0=0.4 * math.log(10), fallback_on_empty=False)

    if args.debug:
        corruptions_robustbench = ["shot_noise"]
        set_individual_corruption(corruption_name="shot_noise")
    else:
        corruptions_robustbench = [
            "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
            "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
            "elastic_transform", "pixelate", "jpeg_compression"
        ]

    if 'cuda' in device.type:
        wandb.login(key=args.wandb_key)

    desc = {"experiment": args.exp_name}
    if 'cuda' in device.type:
        wandb.init(
            project="{}{}_resnet18_cifar10_pruning_tta".format("DEBUG_" if args.debug else "",
                                                               os.environ.get("SLURM_JOB_ID")),
            config=desc,
            name="{}{}_{}".format("DEBUG_" if args.debug else "", args.method, args.prune_data),
            group="{}{}_{}".format("DEBUG_" if args.debug else "", args.exp_name, args.prune_data)
        )

    set_cifar10_corruptions_data_handlers(severity=args.severity, root_path=args.corruptions_root,
                                          corruptions=corruptions_robustbench, encoder_name=args.encoder_name)

    compression_ratios = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

    if args.debug:
        compression_ratios = [0.0, 0.5, 0.95]

    results_per_ratio = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}
    compressed_corruptions_acc_per_ratio = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                            compression_ratios}
    avg_loss_sharpness_pre_tta_per_ratio_corrupted_data = {ratio: [[] for _ in range(len(corruptions_robustbench))] for
                                                           ratio in
                                                           compression_ratios}

    avg_loss_sharpness_pre_tta_per_ratio_training_data = {ratio: [[] for _ in range(len(corruptions_robustbench))] for
                                                          ratio in
                                                          compression_ratios}

    compressed_val_set_acc_per_ratio = {ratio: [] for ratio in compression_ratios}
    sparsity_per_ratio = {ratio: None for ratio in compression_ratios}

    # Prompt entropy storage (per-layer dicts)
    activation_map_entropy_layers_pre = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                         compression_ratios}
    activation_map_entropy_layers_pre_train = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                               compression_ratios}

    # Post-TTA metric storage
    avg_loss_sharpness_post_tta_per_ratio_corrupted_data = {ratio: [[] for _ in range(len(corruptions_robustbench))] for
                                                            ratio in compression_ratios}
    avg_loss_sharpness_post_tta_per_ratio_training_data = {ratio: [[] for _ in range(len(corruptions_robustbench))] for
                                                           ratio in compression_ratios}

    activation_map_entropy_layers_post = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                          compression_ratios}

    activation_map_entropy_layers_post_train = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                                compression_ratios}

    hess_pre_train_top2 = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                           compression_ratios}
    hess_post_train_top2 = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                            compression_ratios}

    hess_pre_corr_top2 = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                          compression_ratios}
    hess_post_corr_top2 = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                           compression_ratios}

    fisher_pre_train = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                        compression_ratios}
    fisher_post_train = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                         compression_ratios}

    fisher_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                       compression_ratios}

    fisher_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                        compression_ratios}

    # Fisher trace per-parameter (normalized by |U|)
    fisher_norm_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                            compression_ratios}
    fisher_norm_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                             compression_ratios}
    fisher_norm_pre_train = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                             compression_ratios}
    fisher_norm_post_train = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                              compression_ratios}

    # Layer-wise Fisher storage
    layerwise_fisher_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                 compression_ratios}
    layerwise_fisher_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                  compression_ratios}

    # ECE (Expected Calibration Error) storage
    ece_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                    compression_ratios}
    ece_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                     compression_ratios}

    # Gradient SNR storage
    snr_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                    compression_ratios}
    snr_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                     compression_ratios}

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

    # Gradient Alignment (cosine similarity between TTA and Oracle gradients)
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

    cka_dense_vs_pruned_pre_tta_train = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                         compression_ratios}
    cka_dense_vs_pruned_pre_tta_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                        compression_ratios}
    cka_pruned_pre_vs_post_tta_train = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                        compression_ratios}
    cka_pruned_pre_vs_post_tta_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                       compression_ratios}

    dense_adapted_models_cka = []

    # Loss barrier storage (ID = clean test data, OOD = corruption data)
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

    pruner_map = {
        "mag-l1": lambda m, r: ResNet18_MagnitudePruning(m, compression_ratio=r, p=1),
        "mag-l2": lambda m, r: ResNet18_MagnitudePruning(m, compression_ratio=r, p=2),
        "wanda": lambda m, r: ResNet18_WandaPruning(m, compression_ratio=r),
        "fold": lambda m, r: ResNet18_ModelFolding(m, compression_ratio=r),
        "rand-prune": lambda m, r: ResNet18_RandomPruning(m, compression_ratio=r),
        "singleton": lambda m, r: ResNet18_Singleton(m, compression_ratio=r),
    }

    print("Pretrained model: ", args.encoder_name, "TTA method: ", args.tta_method)
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
    print("Debug mode: ", args.debug)

    if args.prune_data == 'test':
        if args.method not in ['wanda', 'taylor', 'hessian', 'fold', 'mag-l2', 'entropy_rank']:
            raise ValueError(
                "Pruning with test data is only supported for Wanda, Taylor, Hessian, Fold, and Mag-L2 methods.")
        print("---------------- Pruning using TEST data (Corruptions) ----------------")
    else:
        print("---------------- Pruning using TRAIN data ----------------")

    for seed in [2020]:
        set_seed(seed=seed)
        model = ResNet18_Custom()
        load_model(
            model=model,
            path=args.checkpoint,
            device=device,
        )

        origin_param = count_parameters(model)
        cifar10_c_dataset = Cifar10_C(encoder_name=args.encoder_name, args=args)
        model = model.to(device=device)

        test_loader = get_cifar10(train=False)
        train_loader = get_cifar10(train=True)

        for ratio in compression_ratios:

            if args.prune_data == 'train' or args.method in ['fold', 'mag-l2']:
                if ratio == 0.0:
                    compressed_model = copy.deepcopy(model).to(device)
                else:
                    # Apply pruning/folding
                    if args.method != "taylor" and args.method != "hessian":
                        pruner = pruner_map[args.method](copy.deepcopy(model), ratio)
                        if isinstance(pruner, (ResNet18_WandaPruning)):
                            pruner.run_calibration(train_loader, device, num_batches=50 if not args.debug else 2)
                        compressed_model = pruner.apply().to(device)
                    elif args.method == "taylor" or args.method == "hessian":
                        compressed_model = one_shot_structural_pruning_resnet18(model=copy.deepcopy(model),
                                                                                compression_method=args.method,
                                                                                train_loader=train_loader,
                                                                                compression_ratio=ratio,
                                                                                device=device,
                                                                                num_batches=50 if not args.debug else 2)
                    else:
                        raise NotImplementedError("Only mag-l1, mag-l2, wanda, folding, taylor, hessian are supported.")

                pruned_params = count_parameters(compressed_model)
                sparsity = 100.0 * (1.0 - (pruned_params / origin_param))
                if sparsity_per_ratio[ratio] is None:
                    sparsity_per_ratio[ratio] = sparsity

                if ratio != 0.0:
                    # REPAIR: re-estimate BatchNorm statistics on four training batches (Jordan et al., 2023)
                    for module in compressed_model.modules():
                        if isinstance(module, torch.nn.BatchNorm2d):
                            module.reset_running_stats()
                            module.momentum = None

                    compressed_model.train()
                    idx = 0
                    for x, _ in train_loader:
                        compressed_model(x.to(device=device))
                        idx += 1
                        if idx == 4:
                            break
            else:
                compressed_model = copy.deepcopy(model).to(device)

            compressed_model.eval()
            if not args.debug:
                acc, loss = eval_model(model=copy.deepcopy(compressed_model), dataloader=test_loader, device=device)
                compressed_val_set_acc_per_ratio[ratio].append(acc * 100)
            else:
                print("Sparsity at ratio {}: {:.2f}%".format(ratio, sparsity_per_ratio[ratio]))

            # Pre-adaptation metrics (no TTA)
            if args.prompt_entropy:
                ordered_names_train = get_resnet18_layernames(
                    compressed_model,
                    granularity=args.pe_granularity,
                    include_stem=args.pe_include_stem
                )
                pe_loader_train = get_cifar10(train=True, bs=args.m_sharpness, persistent_workers=False,
                                              pin_memory=True, num_workers=3)

                ent_pre_train, _ = compute_prompt_entropy(
                    model=compressed_model,
                    loader=pe_loader_train,
                    device=device,
                    layer_names=tuple(ordered_names_train),
                    max_batches=args.pe_max_batches,
                    eps=1e-6
                )

                # Replicate training entropy for all corruptions since it's computed on training data
                for corr_idx in range(len(corruptions_robustbench)):
                    activation_map_entropy_layers_pre_train[ratio][corr_idx].append(ent_pre_train)
                del pe_loader_train
                gc.collect()

            # TTA per corruption, with pre- and post-adaptation metrics
            for corruption_idx, corruption in enumerate(corruptions_robustbench):

                if args.prune_data == 'test' and args.method not in ['fold', 'mag-l2']:
                    if ratio == 0.0:
                        compressed_model = copy.deepcopy(model).to(device)
                    else:
                        corruption_loader = cifar10_c_dataset.get_corruption_data_loader(
                            corruption=corruption, batch_size=args.m_sharpness, shuffle=True,
                            persistent_workers=False, pin_memory=True
                        )

                        if args.method != "taylor" and args.method != "hessian":
                            pruner = pruner_map[args.method](copy.deepcopy(model), ratio)
                            if isinstance(pruner, (ResNet18_WandaPruning)):
                                pruner.run_calibration(corruption_loader, device,
                                                       num_batches=50 if not args.debug else 2)
                            compressed_model = pruner.apply().to(device)
                        elif args.method == "taylor" or args.method == "hessian":
                            compressed_model = one_shot_structural_pruning_resnet18(model=copy.deepcopy(model),
                                                                                    compression_method=args.method,
                                                                                    train_loader=corruption_loader,
                                                                                    compression_ratio=ratio,
                                                                                    device=device,
                                                                                    num_batches=50 if not args.debug else 2)
                        else:
                            raise NotImplementedError(
                                "Only mag-l1, mag-l2, wanda, folding, taylor, hessian are supported.")

                        if ratio != 0.0:
                            for module in compressed_model.modules():
                                if isinstance(module, torch.nn.BatchNorm2d):
                                    module.reset_running_stats()
                                    module.momentum = None

                            compressed_model.train()
                            idx = 0
                            for x, _ in corruption_loader:
                                compressed_model(x.to(device=device))
                                idx += 1
                                if idx == 4:
                                    break

                    pruned_params = count_parameters(compressed_model)
                    sparsity = 100.0 * (1.0 - (pruned_params / origin_param))
                    if sparsity_per_ratio[ratio] is None:
                        sparsity_per_ratio[ratio] = sparsity

                top1acc, top5_acc = cifar10_c_dataset.evaluate_model_all_corruption_data(
                    model=copy.deepcopy(compressed_model), device=device,
                    batch_size=64,
                    corruption=corruption, args=args, compute_acc=True)
                compressed_corruptions_acc_per_ratio[ratio][corruption_idx].append(top1acc)

                corruption_adapted_pruned_model = copy.deepcopy(compressed_model)

                if args.prompt_entropy:
                    prompt_entropy_model = copy.deepcopy(corruption_adapted_pruned_model)

                    ordered_names = get_resnet18_layernames(
                        prompt_entropy_model,
                        granularity=args.pe_granularity,
                        include_stem=args.pe_include_stem
                    )
                    pe_loader = cifar10_c_dataset.get_corruption_data_loader(
                        corruption=corruption, batch_size=256, shuffle=False,
                        pin_memory=True, num_workers=3
                    )
                    prompt_entropy_model.eval()
                    ent_pre, erank_pre = compute_prompt_entropy(
                        model=prompt_entropy_model,
                        loader=pe_loader,
                        device=device,
                        layer_names=tuple(ordered_names),
                        max_batches=args.pe_max_batches,
                        eps=1e-6
                    )
                    # store the per-layer dict (no mean)
                    activation_map_entropy_layers_pre[ratio][corruption_idx].append(ent_pre)
                    del pe_loader
                    gc.collect()

                if args.hessian_analysis:
                    try:
                        corruption_loader_pre_tta = cifar10_c_dataset.get_corruption_data_loader(
                            corruption=corruption, batch_size=args.m_sharpness, shuffle=True,
                            pin_memory=True, num_workers=3
                        )

                        ds_corruption = pack_to_gpu_from_loader(
                            loader=corruption_loader_pre_tta,
                            max_batches=args.eval_batches_num if not args.debug else 2,
                            device=device
                        )

                        model_to_analyze_corr = copy.deepcopy(compressed_model).to(device)
                        sharpness.configure_model_for_sar_sharpness(model_to_analyze_corr)
                        model_to_analyze_corr.eval()

                        # Enable grads only for normalization layers (affine) for Hessian
                        for n, p in model_to_analyze_corr.named_parameters():
                            if "bn" in n or "norm" in n:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False

                        evals_corr, _ = get_hessian(device=device, model=copy.deepcopy(model_to_analyze_corr),
                                                    dataset=ds_corruption,
                                                    loss_fn=sar_entropy_loss,
                                                    neigs=1,
                                                    physical_batch_size=args.m_sharpness, exclude_ln=False, args=args,
                                                    preprocess=corruption_loader_pre_tta.transform)
                        hess_val_corr = evals_corr[0].item()
                        hess_pre_corr_top2[ratio][corruption_idx].append(hess_val_corr)

                        train_loader_pre_tta = get_cifar10(train=True, bs=args.m_sharpness,
                                                           pin_memory=True, num_workers=3)

                        ds_train = pack_to_gpu_from_loader(
                            loader=train_loader_pre_tta,
                            max_batches=args.eval_batches_num if not args.debug else 2,
                            device=device
                        )

                        model_to_analyze_train = copy.deepcopy(compressed_model).to(device)
                        sharpness.configure_model_for_sar_sharpness(model_to_analyze_train)
                        model_to_analyze_train.eval()

                        # Enable grads only for normalization layers (affine) for Hessian
                        for n, p in model_to_analyze_train.named_parameters():
                            if "bn" in n or "norm" in n:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False

                        evals_train, _ = get_hessian(device, copy.deepcopy(model_to_analyze_train), ds_train,
                                                     loss_fn=F.cross_entropy,
                                                     neigs=1,
                                                     physical_batch_size=args.m_sharpness, exclude_ln=False, args=args,
                                                     preprocess=train_loader_pre_tta.transform)
                        hess_val_train = evals_train[0].item()
                        hess_pre_train_top2[ratio][corruption_idx].append(hess_val_train)

                        del corruption_loader_pre_tta
                        del ds_corruption
                        del train_loader_pre_tta
                        del ds_train
                        del model_to_analyze_corr
                        del model_to_analyze_train
                        gc.collect()

                    except Exception as e:
                        print(f"Error computing pre-TTA Hessian: {e}")

                if args.fisher_analysis:
                    # Create loader before try block to ensure .transform is available
                    corruption_loader_pre_tta = cifar10_c_dataset.get_corruption_data_loader(
                        corruption=corruption, batch_size=args.m_sharpness, shuffle=True,
                        pin_memory=True, num_workers=3
                    )
                    try:

                        model_to_analyze_corr = copy.deepcopy(compressed_model).to(device)
                        sharpness.configure_model_for_sar_sharpness(model_to_analyze_corr)
                        model_to_analyze_corr.eval()

                        # Enable grads only for normalization layers (affine) for Fisher
                        for n, p in model_to_analyze_corr.named_parameters():
                            if "bn" in n or "norm" in n:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False

                        fisher_result_corr = compute_fisher_trace_subspace(
                            device=device, model=copy.deepcopy(model_to_analyze_corr),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            loss_fn=sar_entropy_loss,
                            physical_batch_size=args.m_sharpness,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        fisher_pre_corr[ratio][corruption_idx].append(fisher_result_corr["raw"])
                        fisher_norm_pre_corr[ratio][corruption_idx].append(fisher_result_corr["normalized"])

                        # Layer-wise Fisher (Pre-TTA)
                        lw_fisher_pre = compute_layerwise_fisher(
                            device=device, model=copy.deepcopy(compressed_model).to(device),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=500 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        layerwise_fisher_pre_corr[ratio][corruption_idx].append(
                            get_sorted_layer_fisher_list(lw_fisher_pre))

                        # ECE (Pre-TTA)
                        ece_val = compute_ece(
                            device=device, model=copy.deepcopy(compressed_model).to(device),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            max_samples=500 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        ece_pre_corr[ratio][corruption_idx].append(ece_val)

                        # Gradient SNR (Pre-TTA)
                        snr_val = compute_gradient_snr(
                            device=device, model=copy.deepcopy(compressed_model).to(device),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        snr_pre_corr[ratio][corruption_idx].append(snr_val)

                        # Layer-wise Gradient SNR (Pre-TTA)
                        lw_snr_val = compute_layerwise_gradient_snr(
                            device=device, model=copy.deepcopy(compressed_model).to(device),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        layerwise_snr_pre_corr[ratio][corruption_idx].append(
                            get_sorted_layer_snr_list(lw_snr_val))

                        # Layer-wise Hessian Trace (Pre-TTA)
                        lw_hess_trace_val = compute_layerwise_hessian_trace(
                            device=device, model=copy.deepcopy(compressed_model).to(device),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            n_hutchinson_iters=10 if not args.debug else 2,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        layerwise_hessian_trace_pre_corr[ratio][corruption_idx].append(
                            get_sorted_layer_hessian_trace_list(lw_hess_trace_val))

                        train_loader_pre_tta = get_cifar10(train=True, bs=args.m_sharpness,
                                                           pin_memory=True, num_workers=3)

                        model_to_analyze_train = copy.deepcopy(compressed_model).to(device)
                        sharpness.configure_model_for_sar_sharpness(model_to_analyze_train)
                        model_to_analyze_train.eval()

                        # Enable grads only for normalization layers (affine) for Fisher
                        for n, p in model_to_analyze_train.named_parameters():
                            if "bn" in n or "norm" in n:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False

                        fisher_result_train = compute_fisher_trace_subspace(device,
                                                                         copy.deepcopy(model_to_analyze_train),
                                                                         train_loader.dataset, F.cross_entropy,
                                                                         physical_batch_size=args.m_sharpness,
                                                                         args=args,
                                                                         preprocess=train_loader_pre_tta.transform)
                        fisher_pre_train[ratio][corruption_idx].append(fisher_result_train["raw"])
                        fisher_norm_pre_train[ratio][corruption_idx].append(fisher_result_train["normalized"])

                        del corruption_loader_pre_tta
                        del train_loader_pre_tta
                        del model_to_analyze_corr
                        del model_to_analyze_train
                        gc.collect()

                    except Exception as e:
                        print(f"Error computing pre-TTA Fisher: {e}")

                # Pre-TTA gradient alignment (cosine similarity)
                if args.gradient_alignment:
                    corruption_loader_ga = cifar10_c_dataset.get_corruption_data_loader(
                        corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                        pin_memory=True, num_workers=3
                    )
                    try:
                        model_for_ga = copy.deepcopy(compressed_model).to(device)
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

                # Pre-TTA loss barrier (ID and OOD)
                if args.loss_barrier:
                    try:
                        # Get dense pretrained state dict (θ₀)
                        dense_pretrained_sd = copy.deepcopy(model.state_dict())
                        # Get pruned pre-TTA state dict (φ_pre)
                        pruned_pre_tta_sd = copy.deepcopy(compressed_model.state_dict())

                        # Compute dense TTA for this corruption (once per corruption, cached)
                        if corruption not in dense_tta_state_dicts:
                            dense_model_for_tta = copy.deepcopy(model).to(device)
                            optimizer_dense, _ = get_adaptation_optimizer_configure_model(
                                tta_method_name=args.tta_method,
                                model=dense_model_for_tta,
                                learning_rate=0.001 if args.tta_method == "tent" else 0.00025,
                                args=args, device=device,
                                cifar10=True
                            )
                            adapted_dense_model = learner.make(
                                name=args.tta_method,
                                model=dense_model_for_tta,
                                optimizer=optimizer_dense
                            )
                            # Adapt dense model on corruption
                            _ = cifar10_c_dataset.evaluate_model_all_corruption_data(
                                model=adapted_dense_model, device=device,
                                batch_size=128 if args.tta_method == "tent" else 64,
                                corruption=corruption, args=args
                            )
                            # Restore BN stats for dense TTA model using corruption data
                            # (same data distribution the model was adapted on)
                            bn_calib_loader = cifar10_c_dataset.get_corruption_data_loader(
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

                        # ID Barrier: B_ID(θ₀, θ̂_pre) on clean test data
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
                        corruption_loader_barrier = cifar10_c_dataset.get_corruption_data_loader(
                            corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                            pin_memory=True, num_workers=3
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

                if args.cka_analysis:
                    try:
                        # Dense vs Pruned (Pre-TTA) on Training Data
                        cka_calc_train = CKACalculator(model1=copy.deepcopy(model),
                                                       model2=copy.deepcopy(compressed_model),
                                                       dataloader=get_cifar10(train=True, bs=args.m_sharpness,
                                                                              persistent_workers=False,
                                                                              pin_memory=True),
                                                       device=device,
                                                       num_epochs=1,
                                                       debug=args.debug,
                                                       is_main_process=True,
                                                       hook_layer_types=CIFAR10_CKA_HOOK_LAYER_TYPES)
                        cka_matrix_train = cka_calc_train.calculate_cka_matrix()
                        cka_diagonal_train = cka_matrix_train.diagonal().tolist()
                        cka_dense_vs_pruned_pre_tta_train[ratio][corruption_idx].append(cka_diagonal_train)
                        cka_calc_train.reset()

                        # Dense vs Pruned (Pre-TTA) on Corruption Data
                        cka_calc_corr = CKACalculator(model1=copy.deepcopy(model),
                                                      model2=copy.deepcopy(compressed_model),
                                                      dataloader=cifar10_c_dataset.get_corruption_data_loader(
                                                          corruption=corruption,
                                                          batch_size=args.m_sharpness,
                                                          shuffle=False,
                                                          persistent_workers=False,
                                                          pin_memory=True),
                                                      device=device,
                                                      num_epochs=1,
                                                      debug=args.debug,
                                                      is_main_process=True,
                                                      hook_layer_types=CIFAR10_CKA_HOOK_LAYER_TYPES)
                        cka_matrix_corr = cka_calc_corr.calculate_cka_matrix()
                        cka_diagonal_corr = cka_matrix_corr.diagonal().tolist()
                        cka_dense_vs_pruned_pre_tta_corr[ratio][corruption_idx].append(cka_diagonal_corr)
                        cka_calc_corr.reset()

                    except Exception as e:
                        print(f"Error computing Pre-TTA CKA metrics: {e}")

                if args.sharpness:
                    try:
                        batches_sharpness = cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                         batch_size=args.m_sharpness,
                                                                                         shuffle=False,
                                                                                         persistent_workers=False,
                                                                                         pin_memory=True)

                        model_sharpness_corr = copy.deepcopy(corruption_adapted_pruned_model)
                        sharpness.configure_model_for_sar_sharpness(model_sharpness_corr)
                        model_sharpness_corr.eval()

                        sharpness_obj_pre_tta_corrupted_data, sharpness_err_pre_tta_corrupted_data, _, output_pre_tta = sharpness.eval_APGD_sharpness(
                            model=model_sharpness_corr, device=device, batches=batches_sharpness,
                            loss_f=sar_entropy_loss,
                            rho=0.002, n_iters=20, n_restarts=1,
                            step_size_mult=1.0,
                            debug=args.debug,
                            rand_init=False, no_grad_norm=False,
                            eval_batches_num=args.eval_batches_num,
                            verbose=False, return_output=True, adaptive=True, version='default',
                            norm='linf')

                        train_loader = get_cifar10_sharpness(train=True,
                                                             bs=args.m_sharpness,
                                                             persistent_workers=False,
                                                             pin_memory=True)

                        model_sharpness_train = copy.deepcopy(corruption_adapted_pruned_model)
                        sharpness.configure_model_for_sar_sharpness(model_sharpness_train)
                        model_sharpness_train.eval()

                        sharpness_obj_pre_tta_training_data, sharpness_err_pre_tta_training_data, _, output_pre_tta_training_data = sharpness.eval_APGD_sharpness(
                            model=model_sharpness_train, device=device, batches=train_loader,
                            loss_f=lambda logits, y: F.cross_entropy(logits, y, reduction='mean'),
                            rho=0.002, n_iters=20, n_restarts=1,
                            step_size_mult=1.0,
                            debug=args.debug,
                            rand_init=False, no_grad_norm=False,
                            eval_batches_num=args.eval_batches_num if not args.debug else 2,
                            verbose=False, return_output=True, adaptive=True, version='default',
                            norm='linf')

                        avg_loss_sharpness_pre_tta_per_ratio_corrupted_data[ratio][corruption_idx].append(
                            float(sharpness_obj_pre_tta_corrupted_data))
                        avg_loss_sharpness_pre_tta_per_ratio_training_data[ratio][corruption_idx].append(
                            float(sharpness_obj_pre_tta_training_data))
                    except Exception as e:
                        print(f"Error computing Pre-TTA Sharpness metrics: {e}")

                if not args.pre_tta_only:
                    optimizer_corruption, spa_model = get_adaptation_optimizer_configure_model(
                        tta_method_name=args.tta_method,
                        model=corruption_adapted_pruned_model,
                        learning_rate=0.001 if args.tta_method == "tent" else 0.00025,
                        args=args, device=device,
                        cifar10=True
                    )

                    # BP-free / passthrough methods: already wrapped by get_adaptation_optimizer_configure_model
                    if args.tta_method in ("norm", "no_adapt", "lame", "pea_resnet18", "pea_vit"):
                        adapted_model_corruption = spa_model if spa_model is not None else corruption_adapted_pruned_model
                    else:
                        adapted_model_corruption = learner.make(
                            name=args.tta_method,
                            model=corruption_adapted_pruned_model,
                            optimizer=optimizer_corruption
                        )
                    # PEA precomputes per-block source statistics (mean/var/cov^{1/2}) on the clean training set
                    if args.tta_method in ("pea_resnet18", "pea_vit"):
                        adapted_model_corruption.precompute_source_stats(
                            train_loader, device,
                        )
                    # Oracle: pass labels so forward_and_adapt uses supervised cross-entropy
                    top1acc, _ = cifar10_c_dataset.evaluate_model_all_corruption_data(
                        model=adapted_model_corruption, device=device,
                        batch_size=128 if args.tta_method == "tent" else 64, corruption=corruption, args=args,
                        pass_labels=(args.tta_method in ("oracle", "oracle_spa"))
                    )

                if args.pre_tta_only:
                    continue  # skip TTA adaptation and all post-TTA computation

                if args.hessian_analysis:
                    # Create loader before try block to ensure it's available
                    train_loader_pre_tta = cifar10_c_dataset.get_corruption_data_loader(
                        corruption=corruption, batch_size=256, shuffle=False
                    )
                    try:

                        ds_train = pack_to_gpu_from_loader(
                            loader=train_loader_pre_tta,
                            max_batches=args.eval_batches_num if not args.debug else 2,
                            device=device
                        )

                        model_to_analyze_train = copy.deepcopy(adapted_model_corruption.model).to(device)
                        sharpness.configure_model_for_sar_sharpness(model_to_analyze_train)
                        model_to_analyze_train.eval()

                        # Enable grads only for normalization layers (affine) for Hessian/Fisher
                        for n, p in model_to_analyze_train.named_parameters():
                            if "bn" in n or "norm" in n:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False

                        # Enable grads only for normalization layers (affine) for Hessian/Fisher
                        for n, p in model_to_analyze_train.named_parameters():
                            if "bn" in n or "norm" in n:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False

                        evals_train, _ = get_hessian(device=device, model=copy.deepcopy(model_to_analyze_train),
                                                     dataset=ds_train,
                                                     loss_fn=F.cross_entropy,
                                                     neigs=1,
                                                     physical_batch_size=args.m_sharpness, exclude_ln=False, args=args,
                                                     preprocess=train_loader_pre_tta.transform)
                        hess_val_train = evals_train[0].item()

                        fisher_result_train = compute_fisher_trace_subspace(device=device,
                                                                         model=copy.deepcopy(model_to_analyze_train),
                                                                         dataset=train_loader.dataset,
                                                                         loss_fn=F.cross_entropy,
                                                                         physical_batch_size=args.m_sharpness,
                                                                         args=args,
                                                                         preprocess=train_loader_pre_tta.transform)

                        hess_post_train_top2[ratio][corruption_idx].append(hess_val_train)
                        fisher_post_train[ratio][corruption_idx].append(fisher_result_train["raw"])
                        fisher_norm_post_train[ratio][corruption_idx].append(fisher_result_train["normalized"])

                        corruption_loader_pre_tta = cifar10_c_dataset.get_corruption_data_loader(
                            corruption=corruption,
                            batch_size=args.m_sharpness,
                            shuffle=True
                        )

                        ds_corruption = pack_to_gpu_from_loader(
                            loader=corruption_loader_pre_tta,
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

                        # Enable grads only for normalization layers (affine) for Hessian/Fisher
                        for n, p in model_to_analyze_corr.named_parameters():
                            if "bn" in n or "norm" in n:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False

                        evals_corr, _ = get_hessian(device=device, model=copy.deepcopy(model_to_analyze_corr),
                                                    dataset=ds_corruption,
                                                    loss_fn=sar_entropy_loss,
                                                    neigs=1,
                                                    physical_batch_size=args.m_sharpness, exclude_ln=False, args=args,
                                                    preprocess=corruption_loader_pre_tta.transform)
                        hess_val_corr = evals_corr[0].item()

                        fisher_result_corr = compute_fisher_trace_subspace(
                            device=device, model=copy.deepcopy(model_to_analyze_corr),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            loss_fn=sar_entropy_loss,
                            physical_batch_size=args.m_sharpness,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)

                        hess_post_corr_top2[ratio][corruption_idx].append(hess_val_corr)
                        fisher_post_corr[ratio][corruption_idx].append(fisher_result_corr["raw"])
                        fisher_norm_post_corr[ratio][corruption_idx].append(fisher_result_corr["normalized"])

                        # Layer-wise Fisher (Post-TTA)
                        lw_fisher_post = compute_layerwise_fisher(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=500 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        layerwise_fisher_post_corr[ratio][corruption_idx].append(
                            get_sorted_layer_fisher_list(lw_fisher_post))

                        # ECE (Post-TTA)
                        ece_val_post = compute_ece(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            max_samples=500 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        ece_post_corr[ratio][corruption_idx].append(ece_val_post)

                        # Gradient SNR (Post-TTA)
                        snr_val_post = compute_gradient_snr(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        snr_post_corr[ratio][corruption_idx].append(snr_val_post)

                        # Layer-wise Gradient SNR (Post-TTA)
                        lw_snr_val_post = compute_layerwise_gradient_snr(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        layerwise_snr_post_corr[ratio][corruption_idx].append(
                            get_sorted_layer_snr_list(lw_snr_val_post))

                        # Layer-wise Hessian Trace (Post-TTA)
                        lw_hess_trace_val_post = compute_layerwise_hessian_trace(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            n_hutchinson_iters=10 if not args.debug else 2,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        layerwise_hessian_trace_post_corr[ratio][corruption_idx].append(
                            get_sorted_layer_hessian_trace_list(lw_hess_trace_val_post))

                        del corruption_loader_pre_tta
                        del ds_corruption
                        del model_to_analyze_corr
                        gc.collect()

                    except Exception as e:
                        print(f"Error computing Post-TTA Hessian/Fisher metrics: {e}")

                # Post-TTA Fisher analysis
                if args.fisher_analysis:
                    # Create loader before try block to ensure .transform is available
                    corruption_loader_fisher = cifar10_c_dataset.get_corruption_data_loader(
                        corruption=corruption, batch_size=args.m_sharpness, shuffle=True,
                        pin_memory=True, num_workers=3
                    )
                    try:
                        model_to_analyze_fisher = copy.deepcopy(adapted_model_corruption.model).to(device)
                        sharpness.configure_model_for_sar_sharpness(model_to_analyze_fisher)
                        model_to_analyze_fisher.eval()

                        # Enable grads only for normalization layers (affine) for Fisher
                        for n, p in model_to_analyze_fisher.named_parameters():
                            if "bn" in n or "norm" in n:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False

                        # Fisher trace on corruption data (Post-TTA)
                        fisher_result_corr_post = compute_fisher_trace_subspace(
                            device=device, model=copy.deepcopy(model_to_analyze_fisher),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            loss_fn=sar_entropy_loss,
                            physical_batch_size=args.m_sharpness,
                            args=args,
                            preprocess=corruption_loader_fisher.transform)
                        fisher_post_corr[ratio][corruption_idx].append(fisher_result_corr_post["raw"])
                        fisher_norm_post_corr[ratio][corruption_idx].append(fisher_result_corr_post["normalized"])

                        # Fisher trace on training data (Post-TTA)
                        fisher_result_train_post = compute_fisher_trace_subspace(
                            device=device, model=copy.deepcopy(model_to_analyze_fisher),
                            dataset=train_loader.dataset,
                            loss_fn=F.cross_entropy,
                            physical_batch_size=args.m_sharpness,
                            args=args,
                            preprocess=corruption_loader_fisher.transform)
                        fisher_post_train[ratio][corruption_idx].append(fisher_result_train_post["raw"])
                        fisher_norm_post_train[ratio][corruption_idx].append(fisher_result_train_post["normalized"])

                        # Layer-wise Fisher (Post-TTA)
                        lw_fisher_post = compute_layerwise_fisher(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=500 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_fisher.transform)
                        layerwise_fisher_post_corr[ratio][corruption_idx].append(
                            get_sorted_layer_fisher_list(lw_fisher_post))

                        # ECE (Post-TTA)
                        ece_val_post = compute_ece(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            max_samples=500 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_fisher.transform)
                        ece_post_corr[ratio][corruption_idx].append(ece_val_post)

                        # Gradient SNR (Post-TTA)
                        snr_val_post = compute_gradient_snr(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_fisher.transform)
                        snr_post_corr[ratio][corruption_idx].append(snr_val_post)

                        # Layer-wise Gradient SNR (Post-TTA)
                        lw_snr_val_post = compute_layerwise_gradient_snr(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_fisher.transform)
                        layerwise_snr_post_corr[ratio][corruption_idx].append(
                            get_sorted_layer_snr_list(lw_snr_val_post))

                        # Layer-wise Hessian Trace (Post-TTA)
                        lw_hess_trace_val_post = compute_layerwise_hessian_trace(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                 batch_size=args.m_sharpness,
                                                                                 shuffle=False,
                                                                                 persistent_workers=False,
                                                                                 pin_memory=True).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            n_hutchinson_iters=10 if not args.debug else 2,
                            args=args,
                            preprocess=corruption_loader_fisher.transform)
                        layerwise_hessian_trace_post_corr[ratio][corruption_idx].append(
                            get_sorted_layer_hessian_trace_list(lw_hess_trace_val_post))

                        del corruption_loader_fisher
                        del model_to_analyze_fisher
                        gc.collect()

                    except Exception as e:
                        print(f"Error computing Post-TTA Fisher metrics: {e}")

                # Post-TTA gradient alignment (cosine similarity)
                if args.gradient_alignment:
                    corruption_loader_ga_post = cifar10_c_dataset.get_corruption_data_loader(
                        corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                        pin_memory=True, num_workers=3
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

                # Post-TTA loss barrier (ID and OOD)
                if args.loss_barrier:
                    try:
                        # Get dense pretrained state dict (θ₀)
                        dense_pretrained_sd = copy.deepcopy(model.state_dict())
                        
                        # SAR/TENT disable BN running stats; re-estimate them on corruption data so the barrier can be evaluated in eval() mode
                        model_for_bn_restore = copy.deepcopy(adapted_model_corruption.model).to(device)
                        bn_calib_loader = cifar10_c_dataset.get_corruption_data_loader(
                            corruption=corruption, batch_size=args.m_sharpness, shuffle=True
                        )
                        restore_bn_running_stats(
                            model=model_for_bn_restore,
                            calibration_loader=bn_calib_loader,
                            device=device,
                            num_batches=50
                        )
                        # Get pruned post-TTA state dict (φ_post) with restored BN stats
                        pruned_post_tta_sd = copy.deepcopy(model_for_bn_restore.state_dict())
                        del model_for_bn_restore, bn_calib_loader
                        gc.collect()

                        # Dense TTA should already be cached from pre-TTA computation
                        if corruption not in dense_tta_state_dicts:
                            dense_model_for_tta = copy.deepcopy(model).to(device)
                            optimizer_dense, _ = get_adaptation_optimizer_configure_model(
                                tta_method_name=args.tta_method,
                                model=dense_model_for_tta,
                                learning_rate=0.001 if args.tta_method == "tent" else 0.00025,
                                args=args, device=device,
                                cifar10=True
                            )
                            adapted_dense_model = learner.make(
                                name=args.tta_method,
                                model=dense_model_for_tta,
                                optimizer=optimizer_dense
                            )
                            _ = cifar10_c_dataset.evaluate_model_all_corruption_data(
                                model=adapted_dense_model, device=device,
                                batch_size=128 if args.tta_method == "tent" else 64,
                                corruption=corruption, args=args
                            )
                            # Restore BN stats for dense TTA model using corruption data
                            # (same data distribution the model was adapted on)
                            bn_calib_loader_dense = cifar10_c_dataset.get_corruption_data_loader(
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

                        # ID Barrier: B_ID(θ₀, θ̂_post) on clean test data
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
                        corruption_loader_barrier = cifar10_c_dataset.get_corruption_data_loader(
                            corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                            pin_memory=True, num_workers=3
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

                if args.cka_analysis:
                    try:

                        if ratio == 0.0:
                            dense_adapted_models_cka.append(
                                copy.deepcopy(adapted_model_corruption.model)
                            )

                        # Pruned Pre-TTA vs Post-TTA on Training Data
                        cka_calc_train = CKACalculator(model1=copy.deepcopy(dense_adapted_models_cka[corruption_idx]),
                                                       model2=copy.deepcopy(adapted_model_corruption.model),
                                                       dataloader=get_cifar10(train=True, bs=args.m_sharpness,
                                                                              persistent_workers=False,
                                                                              pin_memory=True, num_workers=3),
                                                       device=device,
                                                       num_epochs=1,
                                                       debug=args.debug,
                                                       is_main_process=True,
                                                       hook_layer_types=CIFAR10_CKA_HOOK_LAYER_TYPES)
                        cka_matrix_train = cka_calc_train.calculate_cka_matrix()
                        cka_diagonal_train = cka_matrix_train.diagonal().tolist()
                        cka_pruned_pre_vs_post_tta_train[ratio][corruption_idx].append(cka_diagonal_train)
                        cka_calc_train.reset()
                        del cka_calc_train
                        gc.collect()

                        # Pruned Pre-TTA vs Post-TTA on Corruption Data
                        cka_calc_corr = CKACalculator(model1=copy.deepcopy(dense_adapted_models_cka[corruption_idx]),
                                                      model2=copy.deepcopy(adapted_model_corruption.model),
                                                      dataloader=cifar10_c_dataset.get_corruption_data_loader(
                                                          corruption=corruption,
                                                          batch_size=args.m_sharpness,
                                                          shuffle=False, persistent_workers=False, pin_memory=True,
                                                          num_workers=3),
                                                      device=device,
                                                      num_epochs=1,
                                                      debug=args.debug,
                                                      is_main_process=True,
                                                      hook_layer_types=CIFAR10_CKA_HOOK_LAYER_TYPES)
                        cka_matrix_corr = cka_calc_corr.calculate_cka_matrix()
                        cka_diagonal_corr = cka_matrix_corr.diagonal().tolist()
                        cka_pruned_pre_vs_post_tta_corr[ratio][corruption_idx].append(cka_diagonal_corr)
                        cka_calc_corr.reset()
                        del cka_calc_corr
                        gc.collect()

                    except Exception as e:
                        print(f"Error computing Post-TTA CKA metrics: {e}")

                results_per_ratio[ratio][corruption_idx].append(top1acc)

                if args.sharpness:
                    try:
                        post_tta_compressed_model = copy.deepcopy(adapted_model_corruption.model)

                        sharpness.configure_model_for_sar_sharpness(post_tta_compressed_model)
                        post_tta_compressed_model.eval()

                        batches_coruption = cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                         batch_size=args.m_sharpness,
                                                                                         shuffle=False,
                                                                                         persistent_workers=False,
                                                                                         pin_memory=True, num_workers=3)

                        sharpness_corruption_post_tta, sharpness_corruption_err_post_tta, _, output_post_tta = sharpness.eval_APGD_sharpness(
                            model=post_tta_compressed_model, device=device,
                            batches=batches_coruption,
                            loss_f=sar_entropy_loss,
                            rho=0.002, n_iters=20, n_restarts=1, step_size_mult=1.0,
                            debug=args.debug, rand_init=False, no_grad_norm=False,
                            eval_batches_num=args.eval_batches_num if not args.debug else 2,
                            verbose=False, return_output=True, adaptive=True, version='default', norm='linf'
                        )
                        avg_loss_sharpness_post_tta_per_ratio_corrupted_data[ratio][corruption_idx].append(
                            sharpness_corruption_post_tta)

                        # Training Sharpness (Active Subspace Only)
                        post_tta_train_model = copy.deepcopy(adapted_model_corruption.model)
                        sharpness.configure_model_for_sar_sharpness(post_tta_train_model)
                        post_tta_train_model.eval()

                        loss_sharpness_train, err_sharpness_train, _, _ = sharpness.eval_APGD_sharpness(
                            model=post_tta_train_model, device=device,
                            batches=get_cifar10(train=True, bs=args.m_sharpness, persistent_workers=False,
                                                pin_memory=True, num_workers=3),
                            loss_f=lambda logits, y: F.cross_entropy(logits, y, reduction='mean'),
                            rho=0.002, n_iters=20, n_restarts=1, step_size_mult=1.0,
                            debug=args.debug, rand_init=False, no_grad_norm=False,
                            eval_batches_num=args.eval_batches_num if not args.debug else 2,
                            verbose=False, return_output=True, adaptive=True, version='default', norm='linf'
                        )
                        avg_loss_sharpness_post_tta_per_ratio_training_data[ratio][corruption_idx].append(
                            loss_sharpness_train)
                    except Exception as e:
                        print(f"Error computing Post-TTA Sharpness metrics: {e}")

                if args.prompt_entropy:
                    ordered_names = get_resnet18_layernames(adapted_model_corruption.model,
                                                            granularity=args.pe_granularity,
                                                            include_stem=args.pe_include_stem)
                    pe_post, _ = compute_prompt_entropy(
                        model=copy.deepcopy(adapted_model_corruption.model),
                        loader=cifar10_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                            batch_size=args.m_sharpness, shuffle=False,
                                                                            persistent_workers=False, pin_memory=True,
                                                                            num_workers=3),
                        device=device, layer_names=tuple(ordered_names), max_batches=args.pe_max_batches, eps=1e-6
                    )
                    activation_map_entropy_layers_post[ratio][corruption_idx].append(pe_post)

                    activation_map_entropy_layers_post[ratio][corruption_idx].append(pe_post)

                    pe_loader_train = get_cifar10(train=True, bs=args.m_sharpness, persistent_workers=False,
                                                  pin_memory=True, num_workers=3)
                    pe_post_train, _ = compute_prompt_entropy(
                        model=copy.deepcopy(adapted_model_corruption.model),
                        loader=pe_loader_train,
                        device=device, layer_names=tuple(ordered_names), max_batches=args.pe_max_batches, eps=1e-6
                    )
                    activation_map_entropy_layers_post_train[ratio][corruption_idx].append(pe_post_train)
                    del pe_loader_train
                    gc.collect()

    print("\n" + "=" * 50)
    print("CKA Analysis: ", args.cka_analysis)
    if True:
        print("Pre-TTA Metrics (Ascending Sparsity Ratios)")
        print("=" * 50)

        try:
            # Accuracy
            acc_pre_map = format_metric_map(compressed_corruptions_acc_per_ratio, compression_ratios,
                                            corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_map_accuracy_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(acc_pre_map)}")

        except Exception as e:
            print(f"Error printing accuracy maps: {e}")

        # Sharpness
        if args.sharpness:
            try:
                # Pre-TTA Loss
                sharp_loss_pre_map = format_metric_map(avg_loss_sharpness_pre_tta_per_ratio_corrupted_data,
                                                       compression_ratios, corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_map_sharpness_corruption_loss_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(sharp_loss_pre_map)}")

                # Pre-TTA Loss Training
                sharp_loss_training_pre_map = format_metric_map(avg_loss_sharpness_pre_tta_per_ratio_training_data,
                                                                compression_ratios, corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_map_sharpness_training_loss_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(sharp_loss_training_pre_map)}")

            except Exception as e:
                print(f"Error printing sharpness maps: {e}")

        # Prompt Entropy
        if args.prompt_entropy:
            try:
                # Pre-TTA (Corrupted)
                pe_pre_map = format_metric_map(activation_map_entropy_layers_pre, compression_ratios,
                                               corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_map_prompt_entropy_corruption_{args.tta_method}_{args.dataset_name} = {format_map_results(pe_pre_map)}")

                # Pre-TTA (Training)
                pe_pre_train_map = format_metric_map(activation_map_entropy_layers_pre_train, compression_ratios,
                                                     corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_map_prompt_entropy_training_{args.tta_method}_{args.dataset_name} = {format_map_results(pe_pre_train_map)}")

            except Exception as e:
                print(f"Error printing prompt entropy maps: {e}")

        # Hessian
        if args.hessian_analysis:
            try:
                # Pre-TTA Hessian
                hess_pre_corr_map = format_metric_map(hess_pre_corr_top2, compression_ratios, corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_hessian_eigenvalues_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(hess_pre_corr_map)}")

                hess_pre_train_map = format_metric_map(hess_pre_train_top2, compression_ratios, corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_hessian_eigenvalues_training_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(hess_pre_train_map)}")
            except Exception as e:
                print(f"Error printing Hessian maps: {e}")

        if args.fisher_analysis:
            try:
                # Pre-TTA Fisher
                fisher_pre_corr_map = format_metric_map(fisher_pre_corr, compression_ratios, corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_fisher_trace_in_U_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_pre_corr_map)}")

                # Pre-TTA Normalized Fisher (per-parameter)
                fisher_norm_pre_corr_map = format_metric_map(fisher_norm_pre_corr, compression_ratios, corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_fisher_trace_normalized_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_norm_pre_corr_map)}")

                # Pre-TTA Layer-wise Fisher
                lw_fisher_pre_corr_map = format_metric_map(layerwise_fisher_pre_corr, compression_ratios, corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_layerwise_fisher_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_fisher_pre_corr_map)}")

                # Pre-TTA ECE
                ece_pre_corr_map = format_metric_map(ece_pre_corr, compression_ratios, corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_ece_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ece_pre_corr_map)}")

                # Pre-TTA Gradient SNR
                snr_pre_corr_map = format_metric_map(snr_pre_corr, compression_ratios, corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_gradient_snr_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(snr_pre_corr_map)}")

                # Pre-TTA Layer-wise Gradient SNR
                lw_snr_pre_corr_map = format_metric_map(layerwise_snr_pre_corr, compression_ratios, corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_layerwise_gradient_snr_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_snr_pre_corr_map)}")

                # Pre-TTA Layer-wise Hessian Trace
                lw_hess_trace_pre_corr_map = format_metric_map(layerwise_hessian_trace_pre_corr, compression_ratios, corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_layerwise_hessian_trace_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_hess_trace_pre_corr_map)}")

                fisher_pre_train_map = format_metric_map(fisher_pre_train, compression_ratios, corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_fisher_trace_in_U_training_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_pre_train_map)}")

                # Pre-TTA Normalized Fisher on training data
                fisher_norm_pre_train_map = format_metric_map(fisher_norm_pre_train, compression_ratios, corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_fisher_trace_normalized_training_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_norm_pre_train_map)}")
            except Exception as e:
                print(f"Error printing Fisher maps: {e}")

        if args.gradient_alignment:
            try:
                ga_pre_corr_map = format_metric_map(grad_align_pre_corr, compression_ratios,
                                                    corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_gradient_alignment_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ga_pre_corr_map)}")
            except Exception as e:
                print(f"Error printing pre-TTA Gradient Alignment maps: {e}")

            try:
                gn_tta_pre_map = format_metric_map(grad_norm_tta_pre_corr, compression_ratios,
                                                   corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_grad_norm_tta_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(gn_tta_pre_map)}")
                gn_oracle_pre_map = format_metric_map(grad_norm_oracle_pre_corr, compression_ratios,
                                                      corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_grad_norm_oracle_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(gn_oracle_pre_map)}")
            except Exception as e:
                print(f"Error printing pre-TTA Gradient Norm maps: {e}")

        # CKA
        if args.cka_analysis:
            try:
                # Dense vs Pruned (Pre-TTA)
                cka_pre_train_map = format_metric_map(cka_dense_vs_pruned_pre_tta_train, compression_ratios,
                                                      corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_cka_dense_vs_pruned_training_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(cka_pre_train_map)}")

                cka_pre_corr_map = format_metric_map(cka_dense_vs_pruned_pre_tta_corr, compression_ratios,
                                                     corruptions_robustbench)
                print(
                    f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_cka_dense_vs_pruned_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(cka_pre_corr_map)}")

            except Exception as e:
                print(f"Error printing Pre-TTA CKA maps: {e}")

    print("=" * 50 + "\n")

    # Post-TTA metrics
    print("\n" + "=" * 50)
    print("Post-TTA Metrics (Ascending Sparsity Ratios)")
    print("=" * 50)

    try:
        # Accuracy
        acc_post_map = format_metric_map(results_per_ratio, compression_ratios, corruptions_robustbench)
        print(
            f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_map_accuracy_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(acc_post_map)}")
    except Exception as e:
        print(f"Error printing accuracy maps: {e}")

    # Sharpness
    if args.sharpness:
        try:
            # Post-TTA Loss (Corruption)
            sharp_loss_post_map = format_metric_map(avg_loss_sharpness_post_tta_per_ratio_corrupted_data,
                                                    compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_map_sharpness_corruption_loss_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(sharp_loss_post_map)}")

            # Post-TTA Loss (Training)
            sharp_loss_train_post_map = format_metric_map(avg_loss_sharpness_post_tta_per_ratio_training_data,
                                                          compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_map_sharpness_training_loss_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(sharp_loss_train_post_map)}")

        except Exception as e:
            print(f"Error printing post-TTA sharpness maps: {e}")

    # Prompt Entropy
    if args.prompt_entropy:
        try:
            # Post-TTA (Corruption)
            pe_post_map = format_metric_map(activation_map_entropy_layers_post, compression_ratios,
                                            corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_map_prompt_entropy_corruption_{args.tta_method}_{args.dataset_name} = {format_map_results(pe_post_map)}")

            # Post-TTA (Training)
            pe_post_train_map = format_metric_map(activation_map_entropy_layers_post_train, compression_ratios,
                                                  corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_map_prompt_entropy_training_{args.tta_method}_{args.dataset_name} = {format_map_results(pe_post_train_map)}")

        except Exception as e:
            print(f"Error printing post-TTA prompt entropy maps: {e}")

    # Post-TTA Hessian
    if args.hessian_analysis:
        try:
            # Post-TTA Hessian
            hess_post_corr_map = format_metric_map(hess_post_corr_top2, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_hessian_eigenvalues_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(hess_post_corr_map)}")

            hess_post_train_map = format_metric_map(hess_post_train_top2, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_hessian_eigenvalues_training_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(hess_post_train_map)}")
        except Exception as e:
            print(f"Error printing post-TTA Hessian maps: {e}")

    if args.fisher_analysis:
        try:
            # Post-TTA Fisher
            fisher_post_corr_map = format_metric_map(fisher_post_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_fisher_trace_in_U_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_post_corr_map)}")

            # Post-TTA Normalized Fisher (per-parameter)
            fisher_norm_post_corr_map = format_metric_map(fisher_norm_post_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_fisher_trace_normalized_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_norm_post_corr_map)}")

            # Post-TTA Layer-wise Fisher
            lw_fisher_post_corr_map = format_metric_map(layerwise_fisher_post_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_layerwise_fisher_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_fisher_post_corr_map)}")

            # Post-TTA ECE
            ece_post_corr_map = format_metric_map(ece_post_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_ece_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ece_post_corr_map)}")

            # Post-TTA Gradient SNR
            snr_post_corr_map = format_metric_map(snr_post_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_gradient_snr_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(snr_post_corr_map)}")

            # Post-TTA Layer-wise Gradient SNR
            lw_snr_post_corr_map = format_metric_map(layerwise_snr_post_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_layerwise_gradient_snr_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_snr_post_corr_map)}")

            # Post-TTA Layer-wise Hessian Trace
            lw_hess_trace_post_corr_map = format_metric_map(layerwise_hessian_trace_post_corr, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_layerwise_hessian_trace_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_hess_trace_post_corr_map)}")

            fisher_post_train_map = format_metric_map(fisher_post_train, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_fisher_trace_in_U_training_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_post_train_map)}")

            # Post-TTA Normalized Fisher on training data
            fisher_norm_post_train_map = format_metric_map(fisher_norm_post_train, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_fisher_trace_normalized_training_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_norm_post_train_map)}")
        except Exception as e:
            print(f"Error printing post-TTA Fisher maps: {e}")

    if args.gradient_alignment:
        try:
            ga_post_corr_map = format_metric_map(grad_align_post_corr, compression_ratios,
                                                 corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_gradient_alignment_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ga_post_corr_map)}")
        except Exception as e:
            print(f"Error printing post-TTA Gradient Alignment maps: {e}")

        try:
            gn_tta_post_map = format_metric_map(grad_norm_tta_post_corr, compression_ratios,
                                                corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_grad_norm_tta_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(gn_tta_post_map)}")
            gn_oracle_post_map = format_metric_map(grad_norm_oracle_post_corr, compression_ratios,
                                                   corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_grad_norm_oracle_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(gn_oracle_post_map)}")
        except Exception as e:
            print(f"Error printing post-TTA Gradient Norm maps: {e}")

    # Post-TTA CKA
    if args.cka_analysis:
        try:
            # Pruned Pre-TTA vs Post-TTA
            cka_post_train_map = format_metric_map(cka_pruned_pre_vs_post_tta_train, compression_ratios,
                                                   corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_cka_pruned_pre_vs_post_training_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(cka_post_train_map)}")

            cka_post_corr_map = format_metric_map(cka_pruned_pre_vs_post_tta_corr, compression_ratios,
                                                  corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_cka_pruned_pre_vs_post_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(cka_post_corr_map)}")

        except Exception as e:
            print(f"Error printing Post-TTA CKA maps: {e}")

    # Loss Barrier (ID and OOD)
    if args.loss_barrier:
        try:
            # Pre-TTA ID Barrier
            id_barrier_pre_map = format_metric_map(id_barrier_pre_corr, compression_ratios,
                                                   corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_id_barrier_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(id_barrier_pre_map)}")

            # Pre-TTA OOD Barrier
            ood_barrier_pre_map = format_metric_map(ood_barrier_pre_corr, compression_ratios,
                                                    corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_ood_barrier_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ood_barrier_pre_map)}")

            # Post-TTA ID Barrier
            id_barrier_post_map = format_metric_map(id_barrier_post_corr, compression_ratios,
                                                    corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_id_barrier_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(id_barrier_post_map)}")

            # Post-TTA OOD Barrier
            ood_barrier_post_map = format_metric_map(ood_barrier_post_corr, compression_ratios,
                                                     corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_ood_barrier_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ood_barrier_post_map)}")

            # Full Barrier Profiles (barriers at each alpha in [0, 0.1, ..., 1.0])
            # Pre-TTA ID Barrier Profile
            id_barrier_profile_pre_map = format_metric_map(id_barrier_profile_pre_corr, compression_ratios,
                                                           corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_id_barrier_profile_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(id_barrier_profile_pre_map)}")

            # Pre-TTA OOD Barrier Profile
            ood_barrier_profile_pre_map = format_metric_map(ood_barrier_profile_pre_corr, compression_ratios,
                                                            corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_pre_adaptations_ood_barrier_profile_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ood_barrier_profile_pre_map)}")

            # Post-TTA ID Barrier Profile
            id_barrier_profile_post_map = format_metric_map(id_barrier_profile_post_corr, compression_ratios,
                                                            corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_id_barrier_profile_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(id_barrier_profile_post_map)}")

            # Post-TTA OOD Barrier Profile
            ood_barrier_profile_post_map = format_metric_map(ood_barrier_profile_post_corr, compression_ratios,
                                                             corruptions_robustbench)
            print(
                f"resnet18_cifar10_{args.method}_calibration_{args.prune_data}_post_adaptations_ood_barrier_profile_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ood_barrier_profile_post_map)}")

        except Exception as e:
            print(f"Error printing Loss Barrier maps: {e}")

    if args.gradient_alignment:
        try:
            from utils.ga_json import dump_ga_json
            ga_out_dir = os.path.join('logs', 'ga_matched_oracle')
            oracle_method = 'oracle' if args.tta_method == 'sar' else args.tta_method
            # this script defines no --pre_tta_metrics_computation flag; getattr keeps the check compatible
            if getattr(args, 'pre_tta_metrics_computation', False) or args.pre_tta_only:
                p = dump_ga_json(
                    output_dir=ga_out_dir,
                    arch=args.encoder_name,
                    tta_method=args.tta_method,
                    oracle_method=oracle_method,
                    compression_method=args.method,
                    calibration=args.prune_data,
                    severity=args.severity,
                    phase='pre_tta',
                    checkpoint_path=args.checkpoint,
                    compression_ratios=compression_ratios,
                    corruptions=corruptions_robustbench,
                    grad_align=grad_align_pre_corr,
                    grad_norm_tta=grad_norm_tta_pre_corr,
                    grad_norm_oracle=grad_norm_oracle_pre_corr,
                )
                print(f"Wrote PRE-TTA matched-oracle GA JSON: {p}")
            if args.post_tta_metrics_computation and not args.pre_tta_only:
                p = dump_ga_json(
                    output_dir=ga_out_dir,
                    arch=args.encoder_name,
                    tta_method=args.tta_method,
                    oracle_method=oracle_method,
                    compression_method=args.method,
                    calibration=args.prune_data,
                    severity=args.severity,
                    phase='post_tta',
                    checkpoint_path=args.checkpoint,
                    compression_ratios=compression_ratios,
                    corruptions=corruptions_robustbench,
                    grad_align=grad_align_post_corr,
                    grad_norm_tta=grad_norm_tta_post_corr,
                    grad_norm_oracle=grad_norm_oracle_post_corr,
                )
                print(f"Wrote POST-TTA matched-oracle GA JSON: {p}")
        except Exception as e:
            print(f"Error writing matched-oracle GA JSON: {e}")

    return {
        'acc_pre': compressed_corruptions_acc_per_ratio,
        'acc_post': results_per_ratio,
        'sparsity': sparsity_per_ratio,
    }


def main(args):
    """Multi-checkpoint orchestrator. Runs experiment per checkpoint, then aggregates."""
    import copy as _copy
    from utils.multi_checkpoint import parse_checkpoint_paths, print_aggregated_results

    checkpoint_paths = parse_checkpoint_paths(args)
    all_results = []

    for ckpt_idx, ckpt_path in enumerate(checkpoint_paths):
        if len(checkpoint_paths) > 1:
            print(f"\n{'='*70}")
            print(f"CHECKPOINT {ckpt_idx+1}/{len(checkpoint_paths)}: {ckpt_path}")
            print(f"{'='*70}\n")

        args.checkpoint = ckpt_path
        results = _run_for_checkpoint(args)
        if results is not None:
            all_results.append(_copy.deepcopy(results))

    if len(checkpoint_paths) > 1 and len(all_results) > 1:
        corruptions_robustbench = [
            "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
            "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
            "elastic_transform", "pixelate", "jpeg_compression"
        ]
        if args.debug:
            corruptions_robustbench = ["shot_noise"]
        compression_ratios = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85,
                              0.95]
        if args.debug:
            compression_ratios = [0.0, 0.5, 0.95]
        print_aggregated_results(
            all_results, compression_ratios, corruptions_robustbench,
            args, script_prefix="resnet18_cifar10")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet18 compression across ratios/checkpoints")
    parser.add_argument("--method", type=str, default="wanda",
                        choices=["fold", "entropy_rank", "mag-l1", "taylor", "hessian", "mag-l2", "wanda", "rand-fold",
                                 "rand-prune",
                                 "singleton"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--tta_method', default='sar', type=str)
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug evaluation')
    parser.add_argument('--severity', default=5, type=int)
    parser.add_argument("--corruptions_root", default="{}/Few-Shot-Adaptation-Learning/datasets/CIFAR-10-C/{}/{}",
                        type=str)
    parser.add_argument("--dataset_root", default="{}/REDS-Resource-Efficient-Deep-Subnetworks/datasets/imagenet",
                        type=str)
    parser.add_argument('--cuda_device',
                        help='gpu device number',
                        type=int, default=-1)
    parser.add_argument("--repair", type=str, default="REPAIR", help="")
    parser.add_argument('--hessian_analysis', default=False, action='store_true')
    parser.add_argument('--fisher_analysis', default=False, action='store_true')
    parser.add_argument('--gradient_alignment', default=False, action='store_true',
                        help='Compute gradient cosine similarity between TTA and Oracle gradients')
    parser.add_argument('--pre_tta_only', default=False, action='store_true',
                        help='Skip TTA adaptation and all post-TTA computation. '
                             'Use to compute only pre-TTA metrics (e.g., pre-TTA gradient alignment) '
                             'without the cost of running the full adaptation loop.')
    parser.add_argument('--loss_barrier', default=False, action='store_true',
                        help='Compute ID and OOD loss barriers for basin connectivity analysis')
    parser.add_argument('--cka_analysis', default=False, action='store_true',
                        help='Compute CKA analysis')
    parser.add_argument('--sharpness', default=False, action='store_true',
                        help='Compute sharpness of the loss landscape')
    parser.add_argument('--eval_batches_num', default=32, type=int)
    parser.add_argument('--m_sharpness', default=64, type=int)
    parser.add_argument('--no_adapt', default=False, action='store_true',
                        help='Do not adapt the model at test time')
    parser.add_argument('--repair_test_data', default=False, action='store_true',
                        help='Apply repair using test data')
    parser.add_argument('--not_reset_statistics', default=False, action='store_true',
                        help='Do not reset batchnorm statistics after repair')
    parser.add_argument("--checkpoint", type=str, default="pretrained/resnet18_cifar10.pth")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated list of checkpoint paths for multi-checkpoint evaluation. "
                             "Overrides --checkpoint. Results are aggregated with mean ± std.")
    parser.add_argument("--wandb_key", type=str, help="wandb key for login",
                        default=os.environ.get("WANDB_API_KEY"))
    parser.add_argument("--proj_name", type=str, help="", default="{}_fold_resnet18_cifar10")
    parser.add_argument("--exp_name", type=str, help="", default="resnet18_cifar10_pruning")
    parser.add_argument('--encoder_name', default='resnet18', type=str)
    parser.add_argument('--dataset_name', default='cifar', type=str)

    parser.add_argument('--prompt_entropy', default=False, action='store_true',
                        help='Compute Gram-based prompt entropy (pre/post TTA) with batch_size=256')
    parser.add_argument('--pe_max_batches', type=int, default=20,
                        help='Accumulate prompt-entropy over the first N batches (e.g., 20).')
    parser.add_argument('--pe_granularity', default='block', choices=['block', 'conv'],
                        help="Hook BasicBlock outputs ('block') or internal convs ('conv').")
    parser.add_argument('--pe_include_stem', default=False, action='store_true',
                        help='Also include the stem output (relu_conv).')
    parser.add_argument('--prune_data', default='train', choices=['train', 'test'],
                        help='Data to use for pruning calibration: "train" (default) or "test" (corruption data).')

    args = parser.parse_args()
    main(args=args)
