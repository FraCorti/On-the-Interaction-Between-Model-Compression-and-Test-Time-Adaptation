import os
import argparse
import math
from collections import OrderedDict
import gc
import torch
import torch.nn.functional as F
import torch_pruning as tp
from compression.fold import ResNet18_ModelFolding
from utils.compute_hessian_vision import pack_to_gpu_from_loader, get_hessian, compute_fisher_trace_subspace, compute_ece, compute_gradient_snr, compute_layerwise_gradient_snr, get_sorted_layer_snr_list, compute_layerwise_hessian_trace, get_sorted_layer_hessian_trace_list
from utils.fisher_layerwise import compute_layerwise_fisher, get_sorted_layer_fisher_list
from utils.gradient_alignment import compute_gradient_alignment
from utils.loss_barrier import get_id_barrier, get_ood_barrier, inflate_state_dict, restore_bn_running_stats
from utils.custom_structural_pruning import one_shot_structural_pruning_resnet18
from utils.format_results import format_numpy_floats_map, format_metric_map, format_map_results
from cka import CKACalculator
from utils.utils import load_model, eval_model

from utils.seed import set_seed
from compression.mag_prune import ResNet18_MagnitudePruning
from utils.imagenet_c import set_individual_corruption, ImageNet_C
from compression.wanda import ResNet18_WandaPruning
from compression.rand_prune import ResNet18_RandomPruning
import wandb
from utils.datasets import get_imagenet, get_imagenet_sharpness_resnet18
import copy
from utils import learner, losses
from torchvision.models import resnet18
from utils.learner.set_optimizer import get_adaptation_optimizer_configure_model
from utils.imagenet_c import set_imagenet_corruptions_data_handlers
import sharpness


def get_resnet18_layernames(model, granularity="block",
                            include_stem=False, include_maxpool=False, include_avgpool=False):
    """
    Enumerate ResNet-18 layers to hook.
    granularity='block' -> hook the BasicBlock module outputs (post-residual add + final ReLU).
    granularity='conv'  -> hook conv1/conv2 inside each BasicBlock (pre-residual); mainly for ablations.
    The stem and pools are excluded by default.
    If the ResNet is wrapped (e.g. SPA's BYOLWrapper exposes it as `.model`), hook names are
    prefixed with "model." so they match the wrapper's `named_modules()` keys.
    """
    inner = model
    prefix = ""
    if (not hasattr(model, "layer1")) and hasattr(model, "model") \
            and hasattr(model.model, "layer1"):
        inner = model.model
        prefix = "model."

    names = []
    if include_stem:
        names.append(f"{prefix}relu")  # stem output after first ReLU
    if include_maxpool:
        names.append(f"{prefix}maxpool")

    for k in range(1, 5):
        layer = getattr(inner, f"layer{k}")
        for i in range(len(layer)):  # BasicBlock index
            if granularity == "block":
                names.append(f"{prefix}layer{k}.{i}")  # BasicBlock output (post-add, post-ReLU)
            elif granularity == "conv":
                names.append(f"{prefix}layer{k}.{i}.conv1")
                names.append(f"{prefix}layer{k}.{i}.conv2")
            else:
                raise ValueError("granularity must be 'block' or 'conv'")

    if include_avgpool:
        names.append(f"{prefix}avgpool")  # (N,C,1,1); valid for Gram/eigs
    return names


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


def _run_for_checkpoint(args):
    """Run full experiment for a single checkpoint. Returns result dicts for aggregation."""
    device = set_gpu(gpu=args.cuda_device)

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

    desc = {"experiment": args.exp_name.format(args.prune_data)}
    if 'cuda' in device.type:
        wandb.login(key=args.wandb_key)
        wandb.init(
            project="{}{}_resnet18_tta_imagenetc".format("DEBUG_" if args.debug else "",
                                                         os.environ.get("SLURM_JOB_ID")),
            config=desc,
            name="{}{}_{}".format("DEBUG_" if args.debug else "", args.method, args.prune_data),
            group=args.exp_name.format(args.prune_data)
        )

    set_imagenet_corruptions_data_handlers(severity=args.severity,
                                           corruptions=corruptions_robustbench, encoder_name=args.encoder_name,
                                           root_path=args.corruptions_root)

    compression_ratios = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

    results_per_ratio = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in compression_ratios}
    compressed_corruptions_acc_per_ratio = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                            compression_ratios}
    compressed_val_set_acc_per_ratio = {ratio: [] for ratio in compression_ratios}
    sparsity_per_ratio = {ratio: None for ratio in compression_ratios}

    avg_loss_sharpness_pre_tta_per_ratio_corrupted_data = {ratio: [[] for _ in range(len(corruptions_robustbench))] for
                                                           ratio in
                                                           compression_ratios}
    avg_loss_sharpness_pre_tta_per_ratio_training_data = {ratio: [[] for _ in range(len(corruptions_robustbench))] for
                                                          ratio in
                                                          compression_ratios}

    activation_map_entropy_pre_corruption = {ratio: [[] for _ in range(len(corruptions_robustbench))] for
                                             ratio in compression_ratios}
    activation_map_entropy_pre_train = {ratio: [[] for _ in range(len(corruptions_robustbench))] for
                                        ratio in compression_ratios}
    activation_map_entropy_post_corruption = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                              compression_ratios}
    activation_map_entropy_layers_post_train = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                                compression_ratios}

    avg_loss_sharpness_post_tta_per_ratio_corrupted_data = {ratio: [[] for _ in range(len(corruptions_robustbench))]
                                                            for ratio in compression_ratios}
    avg_loss_sharpness_post_tta_per_ratio_training_data = {ratio: [[] for _ in range(len(corruptions_robustbench))]
                                                           for ratio in compression_ratios}

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

    # Layer-wise Hessian Trace storage (on corruption data)
    layerwise_hessian_trace_pre_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                        compression_ratios}
    layerwise_hessian_trace_post_corr = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                         compression_ratios}
    # Layer-wise Hessian Trace on clean ID validation data
    layerwise_hessian_trace_pre_id = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
                                      compression_ratios}
    layerwise_hessian_trace_post_id = {ratio: [[] for _ in range(len(corruptions_robustbench))] for ratio in
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

    pruner_map = {
        "mag-l2": lambda m, r, pbr: ResNet18_MagnitudePruning(m, compression_ratio=r, p=2,
                                                               per_block_ratios=pbr),
        "wanda":  lambda m, r, pbr: ResNet18_WandaPruning(m, compression_ratio=r,
                                                           per_block_ratios=pbr),
        "fold":   lambda m, r, pbr: ResNet18_ModelFolding(m, compression_ratio=r,
                                                           per_block_ratios=pbr),
        "rand-prune": lambda m, r, pbr: ResNet18_RandomPruning(m, compression_ratio=r,
                                                                per_block_ratios=pbr),
    }

    def _build_ps_ratios(r):
        """Protected Stem + Uniform Tail per-block pruning schedule.

        conv1, layer1 -> protected (0% pruned)
        layer2        -> r/2 (half target)
        layer3, layer4 -> r  (full target)
        """
        return {'conv1': 0.0, 'layer1': 0.0, 'layer2': r / 2.0,
                'layer3': r, 'layer4': r}

    print("Pretrained model: ", args.encoder_name, "TTA method: ", args.tta_method, "Pruning method: ", args.method)
    print("Corruptions name and test-time evaluation order: ", corruptions_robustbench)
    print("Sharpness evaluation: ", args.sharpness)
    if args.sharpness:
        print("Number of batches for sharpness evaluation: ", args.eval_batches_num)
        print("Batch size for sharpness evaluation: ", args.m_sharpness)
        print("Prompt entropy evaluation: ", args.prompt_entropy)
    if args.prompt_entropy:
        print("PE granularity: ", args.pe_granularity, "Include stem: ", args.pe_include_stem)
        print("PE max batches: ", args.pe_max_batches)

    if args.prune_data == 'test':
        if args.method not in ['wanda', 'taylor', 'hessian', 'fold', 'mag-l2']:
            raise ValueError(
                "Pruning with test data is only supported for Wanda, Taylor, Hessian, Fold, and Mag-L2 methods.")
        print("---------------- Pruning using TEST data (Corruptions) ----------------")
    else:
        print("---------------- Pruning using TRAIN data ----------------")
    print("Debug mode: ", args.debug)
    print("Prune data: ", args.prune_data)
    print("Pre-TTA metrics computation: ", args.pre_tta_metrics_computation)
    print("Post-TTA metrics computation: ", args.post_tta_metrics_computation)
    print("CKA Analysis: ", args.cka_analysis)

    if args.debug:
        compression_ratios = [0.95]

    for seed in [2020]:
        set_seed(seed=seed)
        model = resnet18(num_classes=1000).to(device=device)
        load_model(
            model,
            args.checkpoint,
            device=device
        )
        origin_param = count_parameters(model)
        imagenet_c_dataset = ImageNet_C(encoder_name=args.encoder_name, args=args)

        test_loader = get_imagenet(datadir=args.dataset_root.format(os.path.dirname(os.getcwd())), train=False,
                                   bs=128)
        train_loader = get_imagenet(datadir=args.dataset_root.format(os.path.dirname(os.getcwd())), train=True,
                                    bs=64)

        for ratio in compression_ratios:

            if args.prune_data == 'train' or args.method in ['mag-l2', 'fold']:
                if ratio == 0.0:
                    compressed_model = copy.deepcopy(model)
                else:
                    # Build per-block schedule for Protected Stem + Uniform Tail if requested
                    ps_ratios = _build_ps_ratios(ratio) if args.non_uniform_pruning else None

                    # Apply pruning/folding
                    if args.method != "taylor" and args.method != "hessian":
                        pruner = pruner_map[args.method](copy.deepcopy(model), ratio, ps_ratios)
                        # Wanda needs calibration before apply()
                        if isinstance(pruner, ResNet18_WandaPruning):
                            pruner.run_calibration(train_loader, device, num_batches=50 if not args.debug else 2)
                        compressed_model = pruner.apply().to(device)
                    elif args.method == "taylor" or args.method == "hessian":
                        compressed_model = one_shot_structural_pruning_resnet18(model=copy.deepcopy(model),
                                                                                compression_method=args.method,
                                                                                train_loader=train_loader,
                                                                                compression_ratio=ratio,
                                                                                device=device,
                                                                                per_block_ratios=ps_ratios,
                                                                                num_batches=50 if not args.debug else 2)
                    else:
                        raise NotImplementedError("Only mag-l2, wanda, folding, taylor, hessian are supported.")

                pruned_macs, pruned_params = tp.utils.count_ops_and_params(model=compressed_model,
                                                                           example_inputs=torch.randn(1, 3, 224,
                                                                                                      224).to(device))
                sparsity = 100.0 * (1.0 - (pruned_params / origin_param))

                if sparsity_per_ratio[ratio] is None:
                    sparsity_per_ratio[ratio] = sparsity
            else:
                compressed_model = copy.deepcopy(model)

            acc, loss = eval_model(model=copy.deepcopy(compressed_model), dataloader=test_loader, device=device)
            compressed_val_set_acc_per_ratio[ratio].append(acc * 100)

            # Training loaders are created once per ratio to avoid exhausting file descriptors
            reused_train_loader_plain = None
            reused_train_loader_sharp = None

            if args.cka_analysis or args.prompt_entropy:
                reused_train_loader_plain = get_imagenet(
                    datadir=args.dataset_root.format(os.path.dirname(os.getcwd())),
                    train=True,
                    bs=args.m_sharpness,
                    persistent_workers=False,
                    pin_memory=True,
                    num_workers=7
                )

            if args.hessian_analysis or args.sharpness:
                reused_train_loader_sharp = get_imagenet_sharpness_resnet18(
                    datadir=args.dataset_root.format(os.path.dirname(os.getcwd())),
                    train=True,
                    bs=args.m_sharpness,
                    persistent_workers=False,
                    pin_memory=True,
                    num_workers=7
                )

            for corruption_idx, corruption in enumerate(corruptions_robustbench):

                if args.prune_data == 'test' and args.method not in ['mag-l2', 'fold']:
                    if ratio == 0.0:
                        compressed_model = copy.deepcopy(model)
                    else:
                        corruption_loader = imagenet_c_dataset.get_corruption_data_loader(
                            corruption=corruption, batch_size=args.m_sharpness, shuffle=True,
                            persistent_workers=False, pin_memory=True, num_workers=7 if args.method != "fold" else 1
                        )

                        # Apply pruning using corruption data
                        if args.method != "taylor" and args.method != "hessian":
                            ps_ratios = _build_ps_ratios(ratio) if args.non_uniform_pruning else None
                            pruner = pruner_map[args.method](copy.deepcopy(model), ratio, ps_ratios)
                            if isinstance(pruner, ResNet18_WandaPruning):
                                pruner.run_calibration(corruption_loader, device,
                                                       num_batches=50 if not args.debug else 2)
                            compressed_model = pruner.apply().to(device)
                        elif args.method == "taylor" or args.method == "hessian":
                            ps_ratios = _build_ps_ratios(ratio) if args.non_uniform_pruning else None
                            compressed_model = one_shot_structural_pruning_resnet18(model=copy.deepcopy(model),
                                                                                    compression_method=args.method,
                                                                                    train_loader=corruption_loader,
                                                                                    compression_ratio=ratio,
                                                                                    per_block_ratios=ps_ratios,
                                                                                    device=device,
                                                                                    num_batches=50 if not args.debug else 2)
                        else:
                            raise NotImplementedError(
                                "Only mag-l1, mag-l2, wanda, folding, taylor, hessian are supported.")

                    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model=compressed_model,
                                                                               example_inputs=torch.randn(1, 3, 224,
                                                                                                          224).to(
                                                                                   device))
                    sparsity = 100.0 * (1.0 - (pruned_params / origin_param))
                    if sparsity_per_ratio[ratio] is None:
                        sparsity_per_ratio[ratio] = sparsity

                    del corruption_loader
                    gc.collect()

                corruption_adapted_model = copy.deepcopy(compressed_model)

                top1acc, top5_acc = imagenet_c_dataset.evaluate_model_all_corruption_data(
                    model=copy.deepcopy(compressed_model), device=device,
                    batch_size=64,
                    corruption=corruption, args=args, compute_acc=True)

                compressed_corruptions_acc_per_ratio[ratio][corruption_idx].append(top1acc)

                if args.hessian_analysis and args.pre_tta_metrics_computation:
                    try:

                        corruption_loader_pre_tta = imagenet_c_dataset.get_corruption_data_loader(
                            corruption=corruption,
                            batch_size=args.m_sharpness,
                            shuffle=True,
                            persistent_workers=False, pin_memory=True, num_workers=7 if args.method != "fold" else 1
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

                        del corruption_loader_pre_tta
                        del ds_corruption
                        del model_to_analyze_corr
                        gc.collect()

                    except Exception as e:
                        print(f"Error computing pre-TTA Hessian: {e}")

                if args.fisher_analysis and args.pre_tta_metrics_computation:
                    # Create loader before try block to ensure .transform is available
                    corruption_loader_pre_tta = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption,
                        batch_size=args.m_sharpness,
                        shuffle=True,
                        persistent_workers=False, pin_memory=True, num_workers=7 if args.method != "fold" else 1
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
                            dataset=imagenet_c_dataset.get_corruption_data_loader(
                                corruption=corruption,
                                batch_size=args.m_sharpness,
                                shuffle=False,
                                persistent_workers=False,
                                pin_memory=True,
                                num_workers=7 if args.method != "fold" else 1
                            ).dataset,
                            loss_fn=sar_entropy_loss,
                            physical_batch_size=args.m_sharpness,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        fisher_pre_corr[ratio][corruption_idx].append(fisher_result_corr["raw"])
                        fisher_norm_pre_corr[ratio][corruption_idx].append(fisher_result_corr["normalized"])

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

                        # Layer-wise Hessian Trace (Pre-TTA) on clean ID validation data
                        lw_hess_trace_val_id = compute_layerwise_hessian_trace(
                            device=device, model=copy.deepcopy(compressed_model).to(device),
                            dataset=test_loader.dataset,
                            loss_fn=F.cross_entropy,
                            max_samples=128 if not args.debug else 10,
                            n_hutchinson_iters=10 if not args.debug else 2,
                            args=args,
                            preprocess=None)  # test_loader already has transforms
                        layerwise_hessian_trace_pre_id[ratio][corruption_idx].append(
                            get_sorted_layer_hessian_trace_list(lw_hess_trace_val_id))

                        del corruption_loader_pre_tta
                        del model_to_analyze_corr
                        gc.collect()

                    except Exception as e:
                        print(f"Error computing pre-TTA Fisher: {e}")

                # Pre-TTA gradient alignment (cosine similarity between TTA and oracle gradients)
                if args.gradient_alignment and args.pre_tta_metrics_computation:
                    corruption_loader_ga = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption,
                        batch_size=args.m_sharpness,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=6 if args.method != "fold" else 1
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

                # Pre-TTA loss barriers (ID and OOD)
                if args.loss_barrier and args.pre_tta_metrics_computation:
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
                                args=args, device=device
                            )
                            adapted_dense_model = learner.make(
                                name=args.tta_method,
                                model=dense_model_for_tta,
                                optimizer=optimizer_dense
                            )
                            # Adapt dense model on corruption
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
                            persistent_workers=False, pin_memory=True, num_workers=7 if args.method != "fold" else 1
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

                if args.cka_analysis and args.pre_tta_metrics_computation:
                    try:
                        # Dense vs Pruned (Pre-TTA) on Corruption Data
                        cka_calc_corr = CKACalculator(model1=copy.deepcopy(model),
                                                      model2=copy.deepcopy(compressed_model),
                                                      dataloader=imagenet_c_dataset.get_corruption_data_loader(
                                                          corruption=corruption,
                                                          batch_size=args.m_sharpness,
                                                          shuffle=False,
                                                          persistent_workers=False,
                                                          pin_memory=True,
                                                          num_workers=7 if args.method != "fold" else 1),
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
                        print(f"Error computing Pre-TTA CKA metrics: {e}")

                if args.prompt_entropy and args.pre_tta_metrics_computation:
                    ordered_names = get_resnet18_layernames(
                        compressed_model,
                        granularity=args.pe_granularity,
                        include_stem=args.pe_include_stem,
                        include_maxpool=args.pe_include_maxpool,
                        include_avgpool=args.pe_include_avgpool,
                    )
                    pe_loader = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                        persistent_workers=False, pin_memory=True, num_workers=7 if args.method != "fold" else 1
                    )
                    pe_pre_vec, _ = compute_prompt_entropy(
                        model=copy.deepcopy(compressed_model),
                        loader=pe_loader,
                        device=device,
                        layer_names=tuple(ordered_names),
                        max_batches=args.pe_max_batches,
                        eps=1e-6
                    )
                    activation_map_entropy_pre_corruption[ratio][corruption_idx].append(pe_pre_vec)
                    del pe_loader
                    gc.collect()

                if args.sharpness and args.pre_tta_metrics_computation:
                    try:
                        pre_tta_compressed_model = copy.deepcopy(corruption_adapted_model)
                        sharpness.configure_model_for_sar_sharpness(pre_tta_compressed_model)
                        pre_tta_compressed_model.eval()

                        batches_sharpness = imagenet_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                          batch_size=args.m_sharpness,
                                                                                          shuffle=False,
                                                                                          persistent_workers=False,
                                                                                          pin_memory=True,
                                                                                          num_workers=7 if args.method != "fold" else 1)

                        sharpness_obj_pre_tta_corrupted_data, sharpness_err_pre_tta_corrupted_data, _, output_pre_tta = sharpness.eval_APGD_sharpness(
                            model=copy.deepcopy(pre_tta_compressed_model), device=device, batches=batches_sharpness,
                            loss_f=sar_entropy_loss,
                            rho=0.002, n_iters=20, n_restarts=1,
                            step_size_mult=1.0,
                            debug=args.debug,
                            rand_init=False, no_grad_norm=False,
                            eval_batches_num=args.eval_batches_num if not args.debug else 2,
                            verbose=False, return_output=True, adaptive=True, version='default',
                            norm='linf')
                        train_loader_sharpness = get_imagenet_sharpness_resnet18(
                            datadir=args.dataset_root.format(os.path.dirname(os.getcwd())),
                            train=True,
                            bs=args.m_sharpness,
                            persistent_workers=False,
                            pin_memory=True)
                        sharpness_obj_pre_tta_training_data, sharpness_err_pre_tta_training_data, _, output_pre_tta_training_data = sharpness.eval_APGD_sharpness(
                            model=copy.deepcopy(pre_tta_compressed_model), device=device,
                            batches=train_loader_sharpness,
                            loss_f=lambda logits, y: F.cross_entropy(logits, y, reduction='mean'),
                            rho=0.002, n_iters=20, n_restarts=1,
                            step_size_mult=1.0,
                            debug=args.debug,
                            rand_init=False, no_grad_norm=False,
                            verbose=False, return_output=True, adaptive=True, version='default',
                            norm='linf')
                        avg_loss_sharpness_pre_tta_per_ratio_training_data[ratio][corruption_idx].append(
                            float(sharpness_obj_pre_tta_training_data))
                        del train_loader_sharpness
                        avg_loss_sharpness_pre_tta_per_ratio_corrupted_data[ratio][corruption_idx].append(
                            float(sharpness_obj_pre_tta_corrupted_data))

                        del batches_sharpness

                        gc.collect()
                    except Exception as e:
                        print(f"Error computing sharpness pre-TTA metrics: {e}")

                # Build the TTA learner and adapt
                if not args.pre_tta_only:
                    optimizer_corruption, spa_model = get_adaptation_optimizer_configure_model(
                        tta_method_name=args.tta_method,
                        model=corruption_adapted_model,
                        learning_rate=0.001,
                        args=args, device=device
                    )

                    if spa_model is not None:
                        corruption_adapted_model = spa_model

                    # BP-free / passthrough methods: already wrapped by get_adaptation_optimizer_configure_model
                    if args.tta_method in ("norm", "no_adapt", "lame", "pea_resnet18", "pea_vit"):
                        adapted_model_corruption = corruption_adapted_model
                    else:
                        adapted_model_corruption = learner.make(
                            name=args.tta_method,
                            model=corruption_adapted_model,
                            optimizer=optimizer_corruption
                        )

                    # PEA precomputes per-block source statistics (mean, variance, cov^{1/2}) on the ImageNet training split
                    if args.tta_method in ("pea_resnet18", "pea_vit"):
                        _pea_src = reused_train_loader_plain if reused_train_loader_plain is not None else train_loader
                        adapted_model_corruption.precompute_source_stats(
                            _pea_src, device,
                        )

                    # Oracle: pass labels so forward_and_adapt uses supervised cross-entropy
                    top1acc, _ = imagenet_c_dataset.evaluate_model_all_corruption_data(
                        model=adapted_model_corruption, device=device,
                        batch_size=64, corruption=corruption, args=args,
                        pass_labels=(args.tta_method in ("oracle", "oracle_spa"))
                    )

                    results_per_ratio[ratio][corruption_idx].append(top1acc)

                if args.pre_tta_only:
                    continue  # skip all post-TTA computation

                if args.hessian_analysis and args.post_tta_metrics_computation:
                    try:
                        corruption_loader_pre_tta = imagenet_c_dataset.get_corruption_data_loader(
                            corruption=corruption,
                            batch_size=args.m_sharpness,
                            shuffle=True,
                            persistent_workers=False,
                            pin_memory=True,
                            num_workers=7 if args.method != "fold" else 1
                        )

                        ds_corruption = pack_to_gpu_from_loader(
                            loader=corruption_loader_pre_tta,
                            max_batches=args.eval_batches_num if not args.debug else 2,
                            device=device
                        )

                        model_to_analyze_corr = copy.deepcopy(adapted_model_corruption.model).to(device)
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
                        hess_post_corr_top2[ratio][corruption_idx].append(hess_val_corr)

                        del corruption_loader_pre_tta
                        del ds_corruption
                        del model_to_analyze_corr
                        gc.collect()

                    except Exception as e:
                        print(f"Error computing post-TTA Hessian: {e}")

                if args.fisher_analysis and args.post_tta_metrics_computation:
                    # Create loader before try block to ensure .transform is available
                    corruption_loader_pre_tta = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption,
                        batch_size=args.m_sharpness,
                        shuffle=True,
                        persistent_workers=False,
                        pin_memory=True,
                        num_workers=7 if args.method != "fold" else 1
                    )
                    try:

                        model_to_analyze_corr = copy.deepcopy(adapted_model_corruption.model).to(device)
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
                            dataset=imagenet_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                  batch_size=args.m_sharpness,
                                                                                  shuffle=False,
                                                                                  persistent_workers=False,
                                                                                  pin_memory=True,
                                                                                  num_workers=7 if args.method != "fold" else 1).dataset,
                            loss_fn=sar_entropy_loss,
                            physical_batch_size=args.m_sharpness,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        fisher_post_corr[ratio][corruption_idx].append(fisher_result_corr["raw"])
                        fisher_norm_post_corr[ratio][corruption_idx].append(fisher_result_corr["normalized"])

                        # Layer-wise Fisher (Post-TTA)
                        lw_fisher_post = compute_layerwise_fisher(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                  batch_size=args.m_sharpness,
                                                                                  shuffle=False,
                                                                                  persistent_workers=False,
                                                                                  pin_memory=True,
                                                                                  num_workers=7 if args.method != "fold" else 1).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=500 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        layerwise_fisher_post_corr[ratio][corruption_idx].append(
                            get_sorted_layer_fisher_list(lw_fisher_post))

                        # ECE (Post-TTA)
                        ece_val_post = compute_ece(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                  batch_size=args.m_sharpness,
                                                                                  shuffle=False,
                                                                                  persistent_workers=False,
                                                                                  pin_memory=True,
                                                                                  num_workers=7 if args.method != "fold" else 1).dataset,
                            max_samples=500 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        ece_post_corr[ratio][corruption_idx].append(ece_val_post)

                        # Gradient SNR (Post-TTA)
                        snr_val_post = compute_gradient_snr(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                  batch_size=args.m_sharpness,
                                                                                  shuffle=False,
                                                                                  persistent_workers=False,
                                                                                  pin_memory=True,
                                                                                  num_workers=7 if args.method != "fold" else 1).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        snr_post_corr[ratio][corruption_idx].append(snr_val_post)

                        # Layer-wise Gradient SNR (Post-TTA)
                        lw_snr_val_post = compute_layerwise_gradient_snr(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                  batch_size=args.m_sharpness,
                                                                                  shuffle=False,
                                                                                  persistent_workers=False,
                                                                                  pin_memory=True,
                                                                                  num_workers=7 if args.method != "fold" else 1).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        layerwise_snr_post_corr[ratio][corruption_idx].append(
                            get_sorted_layer_snr_list(lw_snr_val_post))

                        # Layer-wise Hessian Trace (Post-TTA)
                        lw_hess_trace_val_post = compute_layerwise_hessian_trace(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                  batch_size=args.m_sharpness,
                                                                                  shuffle=False,
                                                                                  persistent_workers=False,
                                                                                  pin_memory=True,
                                                                                  num_workers=7 if args.method != "fold" else 1).dataset,
                            loss_fn=sar_entropy_loss,
                            max_samples=128 if not args.debug else 10,
                            n_hutchinson_iters=10 if not args.debug else 2,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        layerwise_hessian_trace_post_corr[ratio][corruption_idx].append(
                            get_sorted_layer_hessian_trace_list(lw_hess_trace_val_post))

                        # Layer-wise Hessian Trace (Post-TTA) on clean ID validation data
                        lw_hess_trace_val_post_id = compute_layerwise_hessian_trace(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=test_loader.dataset,
                            loss_fn=F.cross_entropy,
                            max_samples=128 if not args.debug else 10,
                            n_hutchinson_iters=10 if not args.debug else 2,
                            args=args,
                            preprocess=None)  # test_loader already has transforms
                        layerwise_hessian_trace_post_id[ratio][corruption_idx].append(
                            get_sorted_layer_hessian_trace_list(lw_hess_trace_val_post_id))

                        del corruption_loader_pre_tta
                        del model_to_analyze_corr
                        gc.collect()

                    except Exception as e:
                        print(f"Error computing post-TTA Fisher: {e}")

                # Post-TTA gradient alignment (cosine similarity between TTA and oracle gradients)
                if args.gradient_alignment and args.post_tta_metrics_computation:
                    corruption_loader_ga_post = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption,
                        batch_size=args.m_sharpness,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=6 if args.method != "fold" else 1
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
                        
                        # SAR/TENT discard BN running statistics; re-estimate them on corruption data so the barrier can be evaluated in eval() mode
                        model_for_bn_restore = copy.deepcopy(adapted_model_corruption.model).to(device)
                        bn_calib_loader = imagenet_c_dataset.get_corruption_data_loader(
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

                        # Dense TTA state is cached by the pre-TTA pass; compute it here if that pass was skipped
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
                            persistent_workers=False, pin_memory=True, num_workers=7 if args.method != "fold" else 1
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

                if args.cka_analysis and args.post_tta_metrics_computation:
                    try:
                        # Pruned Pre-TTA vs Post-TTA on Corruption Data

                        if ratio == 0.0:
                            dense_adapted_models_cka.append(copy.deepcopy(adapted_model_corruption.model))

                        cka_calc_corr = CKACalculator(model1=copy.deepcopy(dense_adapted_models_cka[corruption_idx]),
                                                      model2=copy.deepcopy(adapted_model_corruption.model),
                                                      dataloader=imagenet_c_dataset.get_corruption_data_loader(
                                                          corruption=corruption,
                                                          batch_size=args.m_sharpness,
                                                          shuffle=False,
                                                          persistent_workers=False,
                                                          pin_memory=True,
                                                          num_workers=7 if args.method != "fold" else 1),
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
                        print(f"Error computing Post-TTA CKA metrics: {e}")

                if args.sharpness and args.post_tta_metrics_computation:
                    try:
                        post_tta_compressed_model = copy.deepcopy(adapted_model_corruption.model)
                        sharpness.configure_model_for_sar_sharpness(post_tta_compressed_model)
                        post_tta_compressed_model.eval()

                        batches_coruption = imagenet_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                          batch_size=args.m_sharpness,
                                                                                          shuffle=False,
                                                                                          persistent_workers=False,
                                                                                          pin_memory=True,
                                                                                          num_workers=7 if args.method != "fold" else 1)

                        sharpness_corruption_post_tta, sharpness_corruption_err_post_tta, _, output_post_tta = sharpness.eval_APGD_sharpness(
                            model=copy.deepcopy(post_tta_compressed_model), batches=batches_coruption,
                            loss_f=sar_entropy_loss,
                            device=device,
                            rho=0.002, n_iters=20, n_restarts=1,
                            step_size_mult=1.0,
                            debug=args.debug,
                            eval_batches_num=args.eval_batches_num if not args.debug else 2,
                            rand_init=False, no_grad_norm=False,
                            verbose=False, return_output=True, adaptive=True, version='default',
                            norm='linf')

                        # Sharpness on training data without augmentation, following Andriushchenko et al., "A Modern Look at the Relationship between Sharpness and Generalization" (2023)
                        train_loader = reused_train_loader_sharp
                        sharpness_post_tta_training_data, sharpness_err_post_tta_training_data, _, output_post_tta_training_data = sharpness.eval_APGD_sharpness(
                            model=copy.deepcopy(post_tta_compressed_model), device=device, batches=train_loader,
                            loss_f=lambda logits, y: F.cross_entropy(logits, y, reduction='mean'),
                            rho=0.002, n_iters=20, n_restarts=1,
                            step_size_mult=1.0,
                            debug=args.debug,
                            eval_batches_num=args.eval_batches_num if not args.debug else 2,
                            rand_init=False, no_grad_norm=False,
                            verbose=False, return_output=True, adaptive=True, version='default',
                            norm='linf')
                        avg_loss_sharpness_post_tta_per_ratio_training_data[ratio][corruption_idx].append(
                            sharpness_post_tta_training_data)
                        avg_loss_sharpness_post_tta_per_ratio_corrupted_data[ratio][corruption_idx].append(
                            sharpness_corruption_post_tta)

                        del batches_coruption
                        gc.collect()
                    except Exception as e:
                        print(f"Error computing sharpness post-TTA metrics: {e}")

                if args.prompt_entropy and args.post_tta_metrics_computation:
                    pe_loader = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                        persistent_workers=False, pin_memory=True, num_workers=7 if args.method != "fold" else 1
                    )

                    ordered_names = get_resnet18_layernames(
                        adapted_model_corruption.model,
                        granularity=args.pe_granularity,
                        include_stem=args.pe_include_stem,
                        include_maxpool=False,
                        include_avgpool=args.pe_include_avgpool
                    )

                    pe_post_corruption, _ = compute_prompt_entropy(
                        model=adapted_model_corruption.model,
                        loader=pe_loader,
                        device=device,
                        layer_names=tuple(ordered_names),
                        max_batches=args.pe_max_batches,
                        eps=1e-6
                    )
                    activation_map_entropy_post_corruption[ratio][corruption_idx].append(pe_post_corruption)
                    del pe_loader
                    gc.collect()


            print(f"---- Completed evaluations for compression ratio: {ratio}. ----")

        print("\n" + "=" * 50)
        print("FINAL RESULTS MAPS (Ascending Sparsity Ratios)")
        print("=" * 50)

        diagnostic_metrics = {}

        try:
            acc_pre_map = format_metric_map(compressed_corruptions_acc_per_ratio, compression_ratios,
                                            corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_accuracy_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(acc_pre_map)}")

            acc_post_map = format_metric_map(results_per_ratio, compression_ratios, corruptions_robustbench)
            print(
                f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_accuracy_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(acc_post_map)}")
        except Exception as e:
            print(f"Error printing accuracy maps: {e}")

        if args.pre_tta_metrics_computation:
            print("Pre-TTA Metrics (Ascending Sparsity Ratios)")
            # Sharpness
            if args.sharpness:
                try:
                    # Pre-TTA Loss
                    sharp_loss_pre_map = format_metric_map(avg_loss_sharpness_pre_tta_per_ratio_corrupted_data,
                                                           compression_ratios, corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_map_sharpness_corruption_loss_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(sharp_loss_pre_map)}")

                    # Pre-TTA Loss Training
                    sharp_loss_training_pre_map = format_metric_map(avg_loss_sharpness_pre_tta_per_ratio_training_data,
                                                                    compression_ratios, corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_map_sharpness_training_loss_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(sharp_loss_training_pre_map)}")

                except Exception as e:
                    print(f"Error printing sharpness maps: {e}")

            if args.prompt_entropy:
                try:

                    pe_pre_map = format_metric_map(activation_map_entropy_pre_corruption, compression_ratios,
                                                   corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_map_prompt_entropy_corruption_{args.tta_method}_{args.dataset_name} = {format_map_results(pe_pre_map)}")
                    diagnostic_metrics['ame_pre'] = format_map_results(pe_pre_map)

                except Exception as e:
                    print(f"Error printing prompt entropy maps: {e}")

            if args.hessian_analysis:
                try:
                    hess_pre_corr_map = format_metric_map(hess_pre_corr_top2, compression_ratios,
                                                          corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_hessian_eigenvalues_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(hess_pre_corr_map)}")
                except Exception as e:
                    print(f"Error printing Hessian maps: {e}")

            if args.fisher_analysis:
                try:
                    # Pre-TTA Fisher Trace (Corruption)
                    fisher_pre_corr_map = format_metric_map(fisher_pre_corr, compression_ratios,
                                                            corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_fisher_trace_in_U_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_pre_corr_map)}")

                    # Pre-TTA Normalized Fisher (per-parameter)
                    fisher_norm_pre_corr_map = format_metric_map(fisher_norm_pre_corr, compression_ratios,
                                                                 corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_fisher_trace_normalized_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_norm_pre_corr_map)}")

                    # Pre-TTA Layer-wise Fisher (Corruption)
                    lw_fisher_pre_corr_map = format_metric_map(layerwise_fisher_pre_corr, compression_ratios,
                                                               corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_layerwise_fisher_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_fisher_pre_corr_map)}")

                    # Pre-TTA ECE (Corruption)
                    ece_pre_corr_map = format_metric_map(ece_pre_corr, compression_ratios,
                                                         corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_ece_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ece_pre_corr_map)}")

                    # Pre-TTA Gradient SNR (Corruption)
                    snr_pre_corr_map = format_metric_map(snr_pre_corr, compression_ratios,
                                                         corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_gradient_snr_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(snr_pre_corr_map)}")

                    # Pre-TTA Layer-wise Gradient SNR (Corruption)
                    lw_snr_pre_corr_map = format_metric_map(layerwise_snr_pre_corr, compression_ratios,
                                                            corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_layerwise_gradient_snr_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_snr_pre_corr_map)}")

                    # Pre-TTA Layer-wise Hessian Trace (Corruption)
                    lw_hess_trace_pre_corr_map = format_metric_map(layerwise_hessian_trace_pre_corr, compression_ratios,
                                                                   corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_layerwise_hessian_trace_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_hess_trace_pre_corr_map)}")

                    # Pre-TTA Layer-wise Hessian Trace on clean ID validation data
                    lw_hess_trace_pre_id_map = format_metric_map(layerwise_hessian_trace_pre_id, compression_ratios,
                                                                  corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_layerwise_hessian_trace_clean_id_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_hess_trace_pre_id_map)}")

                except Exception as e:
                    print(f"Error printing Fisher maps: {e}")

            if args.gradient_alignment:
                try:
                    ga_pre_corr_map = format_metric_map(grad_align_pre_corr, compression_ratios,
                                                        corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_gradient_alignment_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ga_pre_corr_map)}")
                    diagnostic_metrics['ga_pre'] = format_numpy_floats_map(ga_pre_corr_map)
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
                    diagnostic_metrics['gn_tta_pre'] = format_numpy_floats_map(gn_tta_pre_map)
                    diagnostic_metrics['gn_oracle_pre'] = format_numpy_floats_map(gn_oracle_pre_map)
                except Exception as e:
                    print(f"Error printing pre-TTA Gradient Norm maps: {e}")

            # CKA
            if args.cka_analysis:
                try:
                    # Dense vs Pruned (Pre-TTA)
                    cka_pre_corr_map = format_metric_map(cka_dense_vs_pruned_pre_tta_corr, compression_ratios,
                                                         corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_cka_dense_vs_pruned_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(cka_pre_corr_map)}")
                    diagnostic_metrics['cka_pre'] = format_numpy_floats_map(cka_pre_corr_map)

                except Exception as e:
                    print(f"Error printing Pre-TTA CKA maps: {e}")

        print("=" * 50 + "\n")

        if args.post_tta_metrics_computation:

            print("=" * 50 + "\n")
            print("Post-TTA Metrics (Ascending Sparsity Ratios)")
            if args.sharpness:
                try:
                    sharp_loss_post_map = format_metric_map(avg_loss_sharpness_post_tta_per_ratio_corrupted_data,
                                                            compression_ratios, corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_map_sharpness_corruption_loss_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(sharp_loss_post_map)}")

                    sharp_loss_train_post_map = format_metric_map(avg_loss_sharpness_post_tta_per_ratio_training_data,
                                                                 compression_ratios, corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_map_sharpness_training_loss_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(sharp_loss_train_post_map)}")

                except Exception as e:
                    print(f"Error printing post-TTA sharpness maps: {e}")

            if args.prompt_entropy:
                try:
                    pe_post_map = format_metric_map(activation_map_entropy_post_corruption, compression_ratios,
                                                    corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_map_prompt_entropy_corruption_{args.tta_method}_{args.dataset_name} = {format_map_results(pe_post_map)}")
                    diagnostic_metrics['ame_post'] = format_map_results(pe_post_map)

                except Exception as e:
                    print(f"Error printing post-TTA prompt entropy maps: {e}")

            if args.hessian_analysis:
                try:
                    hess_post_corr_map = format_metric_map(hess_post_corr_top2, compression_ratios,
                                                           corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_hessian_eigenvalues_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(hess_post_corr_map)}")
                except Exception as e:
                    print(f"Error printing post-TTA Hessian maps: {e}")

            if args.fisher_analysis:
                try:
                    fisher_post_corr_map = format_metric_map(fisher_post_corr, compression_ratios,
                                                             corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_fisher_trace_in_U_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_post_corr_map)}")

                    # Post-TTA Normalized Fisher (per-parameter)
                    fisher_norm_post_corr_map = format_metric_map(fisher_norm_post_corr, compression_ratios,
                                                                  corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_fisher_trace_normalized_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_norm_post_corr_map)}")

                    # Post-TTA Layer-wise Fisher (Corruption)
                    lw_fisher_post_corr_map = format_metric_map(layerwise_fisher_post_corr, compression_ratios,
                                                                corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_layerwise_fisher_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_fisher_post_corr_map)}")

                    # Post-TTA ECE (Corruption)
                    ece_post_corr_map = format_metric_map(ece_post_corr, compression_ratios,
                                                          corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_ece_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ece_post_corr_map)}")

                    # Post-TTA Gradient SNR (Corruption)
                    snr_post_corr_map = format_metric_map(snr_post_corr, compression_ratios,
                                                          corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_gradient_snr_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(snr_post_corr_map)}")

                    # Post-TTA Layer-wise Gradient SNR (Corruption)
                    lw_snr_post_corr_map = format_metric_map(layerwise_snr_post_corr, compression_ratios,
                                                             corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_layerwise_gradient_snr_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_snr_post_corr_map)}")

                    # Post-TTA Layer-wise Hessian Trace (Corruption)
                    lw_hess_trace_post_corr_map = format_metric_map(layerwise_hessian_trace_post_corr, compression_ratios,
                                                                    corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_layerwise_hessian_trace_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_hess_trace_post_corr_map)}")
                except Exception as e:
                    print(f"Error printing post-TTA Fisher maps: {e}")

            if args.gradient_alignment:
                try:
                    ga_post_corr_map = format_metric_map(grad_align_post_corr, compression_ratios,
                                                         corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_gradient_alignment_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ga_post_corr_map)}")
                    diagnostic_metrics['ga_post'] = format_numpy_floats_map(ga_post_corr_map)
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
                    diagnostic_metrics['gn_tta_post'] = format_numpy_floats_map(gn_tta_post_map)
                    diagnostic_metrics['gn_oracle_post'] = format_numpy_floats_map(gn_oracle_post_map)
                except Exception as e:
                    print(f"Error printing post-TTA Gradient Norm maps: {e}")

            # Post-TTA CKA
            if args.cka_analysis:
                try:
                    # Pruned Pre-TTA vs Post-TTA
                    cka_post_corr_map = format_metric_map(cka_pruned_pre_vs_post_tta_corr, compression_ratios,
                                                          corruptions_robustbench)
                    print(
                        f"resnet18_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_cka_pruned_pre_vs_post_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(cka_post_corr_map)}")
                    diagnostic_metrics['cka_post'] = format_numpy_floats_map(cka_post_corr_map)

                except Exception as e:
                    print(f"Error printing Post-TTA CKA maps: {e}")

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

    if args.gradient_alignment:
        try:
            from utils.ga_json import dump_ga_json
            ga_out_dir = os.path.join('logs', 'ga_matched_oracle')
            oracle_method = 'oracle' if args.tta_method == 'sar' else args.tta_method
            if args.pre_tta_metrics_computation or args.pre_tta_only:
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
        'diagnostic_metrics': diagnostic_metrics,
    }


def main(args):
    """Multi-checkpoint orchestrator."""
    import copy as _copy
    from utils.multi_checkpoint import (parse_checkpoint_paths,
                                        print_aggregated_results,
                                        print_aggregated_diagnostic_metrics)

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
            args, script_prefix="resnet18_imagenet")
        print_aggregated_diagnostic_metrics(
            all_results, corruptions_robustbench, args,
            script_prefix="resnet18_imagenet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet-18 compression + TTA on ImageNet-C")
    parser.add_argument("--method", type=str, default="wanda",
                        choices=["fold", "mag-l1", "taylor", "hessian", "mag-l2", "wanda", "rand-fold", "rand-prune",
                                 "singleton"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--tta_method', default='spa', type=str)
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug evaluation')
    parser.add_argument('--severity', default=5, type=int)
    parser.add_argument("--corruptions_root", default="{}/Few-Shot-Adaptation-Learning/datasets/imagenet-c/{}/{}",
                        type=str)
    parser.add_argument("--dataset_root", default="{}/REDS-Resource-Efficient-Deep-Subnetworks/datasets/imagenet",
                        type=str)
    parser.add_argument('--cuda_device',
                        help='gpu device number',
                        type=int, default=-1)
    parser.add_argument("--repair", type=str, default="REPAIR", help="")
    parser.add_argument('--no_adapt', default=False, action='store_true',
                        help='Do not adapt the model at test time')
    parser.add_argument('--pre_tta_metrics_computation', default=False, action='store_true',
                        help='Compute sharpness, activation map entropies, Hessian, Fisher before TTA')
    parser.add_argument('--post_tta_metrics_computation', default=False, action='store_true',
                        help='Compute sharpness, activation map entropies, Hessian, Fisher after TTA')
    parser.add_argument('--prune_data', default='train', choices=['train', 'test'],
                        help='Data to use for pruning calibration: "train" (default) or "test" (corruption data).')

    parser.add_argument('--not_reset_statistics', default=False, action='store_true',
                        help='Do not reset batchnorm statistics after repair')
    parser.add_argument("--checkpoint", type=str, default="pretrained/resnet18_imagenet.pth")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated checkpoint paths for multi-checkpoint evaluation.")
    parser.add_argument("--wandb_key", type=str, help='wandb key for login',
                        default=os.environ.get("WANDB_API_KEY"))
    parser.add_argument("--proj_name", type=str, help="", default="{}_fold_resnet18_imagenet")
    parser.add_argument("--exp_name", type=str, help="", default="vit_tta_compression_imagenet_c_calibration{}")
    parser.add_argument('--encoder_name', default='vit_base_patch16_224', type=str)
    parser.add_argument('--repair_test_data', default=False, action='store_true',
                        help='Apply repair using test data')
    parser.add_argument('--pe_max_batches', type=int, default=20,
                        help='Accumulate prompt-entropy over the first N batches (e.g., 20). '
                             'If None, use the full corruption split.')
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
    parser.add_argument('--dataset_name', default='imagenet', type=str)
    parser.add_argument('--prompt_entropy', default=False, action='store_true',
                        help='Compute Gram-based prompt entropy (pre/post TTA) using batch_size=args.m_sharpness')
    # Layer selection for prompt entropy (default: BasicBlock outputs only)
    parser.add_argument('--pe_granularity', default='block', choices=['block', 'conv'],
                        help="Hook BasicBlock outputs ('block') or internal convs ('conv').")
    parser.add_argument('--pe_include_stem', default=False, action='store_true',
                        help='Also include the stem output after first ReLU.')
    parser.add_argument('--pe_include_avgpool', default=False, action='store_true',
                        help='Also include avgpool output.')
    parser.add_argument('--pe_include_maxpool', default=False, action='store_true',
                        help='Also include maxpool output.')
    # Prompt-entropy options
    parser.add_argument('--pe_alpha', type=float, default=1.0,
                        help='Alpha for matrix alpha-entropy.')
    parser.add_argument('--pe_norm', type=str, default='maxEntropy',
                        choices=['maxEntropy', 'logN', 'logD', 'logNlogD', 'raw', 'length'],
                        help='Normalization for dataset entropy.')
    parser.add_argument('--pe_token_pool', type=str, default='cls', choices=['cls', 'mean'],
                        help='Which token rep to use per block: CLS or mean over patch tokens (only for dataset entropy).')
    parser.add_argument('--pe_entropy_mode', type=str, default='dataset', choices=['dataset', 'token'],
                        help='Compute entropy across batch (dataset) or across tokens (token).')

    parser.add_argument('--non-uniform-pruning', dest='non_uniform_pruning',
                        default=False, action='store_true',
                        help='Enable the Protected Stem + Uniform Tail pruning schedule '
                             '(conv1+layer1: 0%%, layer2: r/2, layer3+layer4: r). '
                             'Corresponds to the PS-{method} experiments in the paper.')

    args = parser.parse_args()
    main(args=args)
