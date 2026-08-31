import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from timm.data import resolve_data_config, create_transform
from timm.loss import LabelSmoothingCrossEntropy
from utils.compute_hessian_vision import pack_to_gpu_from_loader, get_hessian, compute_fisher_trace_subspace, compute_ece, compute_gradient_snr, compute_layerwise_gradient_snr, get_sorted_layer_snr_list, compute_layerwise_hessian_trace, get_sorted_layer_hessian_trace_list
from utils.fisher_layerwise import compute_layerwise_fisher, get_sorted_layer_fisher_list
from utils.gradient_alignment import compute_gradient_alignment
from utils.loss_barrier import get_id_barrier, get_ood_barrier, inflate_state_dict
from utils.custom_structural_pruning import one_shot_structural_pruning_vit
from utils.format_results import format_numpy_floats_map, format_metric_map
from utils.learner.spa import SPA
from utils.losses import SPAConsistencyLoss, OracleSPALoss, SAR_EntropyLoss
from utils.utils import eval_model, spa_configure_model
from cka import CKACalculator, VIT_HOOK_LAYER_TYPES
import torch_pruning as tp
from utils.seed import set_seed
from compression.fold import ViT_ModelFolding
from compression.mag_prune import ViT_MagnitudePruning
from utils.imagenet_c import set_individual_corruption, ImageNet_C
from compression.wanda import ViT_WandaPruning
from compression.rand_prune import ViT_RandomPruning
import wandb
from utils.datasets import get_imagenet_vit
import copy
from utils import learner
from utils.learner.set_optimizer import get_adaptation_optimizer_configure_model
from utils.imagenet_c import set_imagenet_corruptions_data_handlers
import sharpness
import timm
import gc
# Dataset (prompt) entropy utilities.
import math
import repitl.matrix_itl as itl


def entropy_normalization(entropy, normalization, N, D):
    """
    Normalize a matrix entropy value by the chosen scheme.
    """
    assert normalization in ['maxEntropy', 'logN', 'logD', 'logNlogD', 'raw', 'length']
    if normalization == 'maxEntropy':
        entropy /= min(math.log(N), math.log(D))
    elif normalization == 'logN':
        entropy /= math.log(N)
    elif normalization == 'logD':
        entropy /= math.log(D)
    elif normalization == 'logNlogD':
        entropy /= (math.log(N) * math.log(D))
    elif normalization == 'raw':
        pass
    elif normalization == 'length':
        entropy = N
    return entropy


def compute_entropy(hidden_states, alpha=1, normalizations=['maxEntropy']):
    """
    Compute dataset (prompt) entropy across ViT blocks.

    Args:
        hidden_states: torch.Tensor of shape [L, N, D]
                       L = #ViT blocks (12 for ViT-B/16), N = samples, D = embed dim (768)

    Returns:
        dict: {normalization_name: [length-L list of entropies]}
    """
    L, N, D = hidden_states.shape

    if N > D:
        cov = torch.matmul(hidden_states.transpose(1, 2), hidden_states)  # [L, D, D]
    else:
        cov = torch.matmul(hidden_states, hidden_states.transpose(1, 2))  # [L, N, N]

    # The Gram matrix is PSD with possibly negative off-diagonals; eigenvalues are clamped inside matrixAlphaEntropy.
    entropies = []
    for layer_cov in cov:
        try:
            layer_cov = layer_cov.double() / torch.trace(layer_cov.double())
            entropies.append(itl.matrixAlphaEntropy(layer_cov, alpha=alpha).item())
        except Exception:
            entropies.append(float('nan'))

    return {norm: [entropy_normalization(x, norm, N, D) for x in entropies] for norm in normalizations}


def compute_token_entropy(hidden_states, alpha=1, normalizations=['maxEntropy']):
    """
    Compute token entropy (per-image entropy of tokens, averaged over the batch).

    Args:
        hidden_states: torch.Tensor of shape [L, N, T, D]
                       L = #ViT blocks, N = samples, T = tokens, D = embed dim

    Returns:
        dict: {normalization_name: [length-L list of entropies]}
    """
    L, N, T, D = hidden_states.shape

    entropies_per_layer = []  # List of L values (each is average over N images)

    for l in range(L):
        layer_entropies = []
        for n in range(N):
            # [T, D] matrix for one image
            Z = hidden_states[l, n, :, :]

            if T > D:
                cov = torch.matmul(Z.transpose(0, 1), Z)  # [D, D]
            else:
                cov = torch.matmul(Z, Z.transpose(0, 1))  # [T, T]

            # The Gram matrix is PSD; small negative eigenvalues are filtered below.
            tr = torch.trace(cov)
            if tr > 0:
                cov = cov / tr
                try:
                    eigs = torch.linalg.eigvalsh(cov)
                    # Filter small negatives (numerical artifacts from PSD matrix)
                    eigs = eigs[eigs > 1e-12]
                    if alpha == 1.0:
                        # Use base 2 for bits
                        e = -torch.sum(eigs * torch.log2(eigs))
                    else:
                        e = (1.0 / (1.0 - alpha)) * torch.log2(torch.sum(eigs ** alpha))
                    layer_entropies.append(e.item())
                except Exception:
                    layer_entropies.append(float('nan'))
            else:
                layer_entropies.append(0.0)

        if len(layer_entropies) > 0:
            entropies_per_layer.append(np.nanmean(layer_entropies))
        else:
            entropies_per_layer.append(float('nan'))

    # For token entropy the N of the normalization is the sequence length T.
    return {norm: [entropy_normalization(x, norm, T, D) for x in entropies_per_layer] for norm in normalizations}


def unwrap_model(m):
    """
    Unwrap TTA/prompt/DDP wrappers (`.vit`, `.model`, `.module`) until a timm ViT exposing `.blocks` is reached.
    """
    seen = set()
    while True:
        if id(m) in seen:
            break
        seen.add(id(m))
        if hasattr(m, 'blocks'):
            break
        if hasattr(m, 'vit'):
            m = m.vit
            continue
        if hasattr(m, 'model'):
            m = m.model
            continue
        if hasattr(m, 'module'):
            m = m.module
            continue
        break
    return m


@torch.no_grad()
def _collect_vit_block_outputs(model_or_wrapper, loader, device, max_batches=20, token_pool='cls'):
    """
    Capture outputs of each ViT block and pool tokens to [B, D], then stack to [L, N, D].

    token_pool:
        'cls'  -> take x[:, 0, :]
        'mean' -> take mean over patch tokens x[:, 1:, :].mean(dim=1)
        'none' -> take all tokens x[:, :, :] (returns [L, N, T, D])
    """
    model = unwrap_model(model_or_wrapper)
    assert hasattr(model, 'blocks'), "Expected a timm ViT model with `.blocks`."
    L = len(model.blocks)

    per_layer_chunks = [[] for _ in range(L)]
    hooks = []

    def make_hook(layer_idx):
        def _hook(_module, _inp, out):
            x = out[0] if isinstance(out, tuple) else out  # [B, T, D]
            if token_pool == 'cls':
                z = x[:, 0, :]  # [B, D]
            elif token_pool == 'mean':
                z = x[:, 1:, :].mean(dim=1)  # [B, D]
            elif token_pool == 'none':
                z = x  # [B, T, D]
            else:
                raise ValueError(f"Unsupported token_pool: {token_pool}")
            per_layer_chunks[layer_idx].append(z.detach().cpu())

        return _hook

    for i, blk in enumerate(model.blocks):
        hooks.append(blk.register_forward_hook(make_hook(i)))

    model_or_wrapper.eval()
    n_batches = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        _ = model_or_wrapper(images)  # forward through wrapper so hooks fire
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    for h in hooks:
        h.remove()

    # Infer embed dim from first non-empty chunk
    D = None
    for i in range(L):
        if len(per_layer_chunks[i]) > 0:
            D = per_layer_chunks[i][0].shape[1]
            break
    if D is None:
        return torch.empty(0, 0, 0)

    # Stack into [L, N, D] or [L, N, T, D], trimming to min N across layers
    per_layer = []
    for i in range(L):
        if len(per_layer_chunks[i]) == 0:
            if D is None: return torch.empty(0)
            if token_pool == 'none':
                per_layer.append(torch.empty(0, 0, D))
            else:
                per_layer.append(torch.empty(0, D))
        else:
            per_layer.append(torch.cat(per_layer_chunks[i], dim=0))  # [N_i, D] or [N_i, T, D]
    Ns = [p.shape[0] for p in per_layer if p.shape[0] > 0]
    N = min(Ns) if len(Ns) else 0
    per_layer = [p[:N] for p in per_layer]
    hidden_states = torch.stack(per_layer, dim=0)  # [L, N, D] or [L, N, T, D]
    return hidden_states


def compute_vit_dataset_entropy(model_or_wrapper, data_loader, device, max_batches, token_pool, alpha, normalization,
                                entropy_mode='dataset'):
    """
    Convenience wrapper: collect representations then compute entropy.
    entropy_mode: 'dataset' (across batch) or 'token' (across tokens per image).
    """
    # Token entropy needs all tokens.
    pool_arg = 'none' if entropy_mode == 'token' else token_pool

    hs = _collect_vit_block_outputs(
        model_or_wrapper=model_or_wrapper,
        loader=data_loader,
        device=device,
        max_batches=max_batches,
        token_pool=pool_arg,
    )  # [L, N, D] or [L, N, T, D]

    if entropy_mode == 'token':
        out = compute_token_entropy(hs, alpha=alpha, normalizations=[normalization])
    else:
        out = compute_entropy(hs, alpha=alpha, normalizations=[normalization])
    return out[normalization]


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


def _run_for_checkpoint(args):
    """Run full experiment for a single checkpoint. Returns result dicts for aggregation."""
    device = set_gpu(gpu=args.cuda_device)
    train_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)
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
            project="{}{}_vit_base_patch16_224_tta_imagenetc".format("DEBUG_" if args.debug else "",
                                                                     os.environ.get("SLURM_JOB_ID")),
            config=desc,
            name="{}{}_{}".format("DEBUG_" if args.debug else "", args.method, args.prune_data),
            group=args.exp_name.format(args.prune_data)
        )

    set_imagenet_corruptions_data_handlers(severity=args.severity,
                                           corruptions=corruptions_robustbench, encoder_name=args.encoder_name,
                                           root_path=args.corruptions_root)

    compression_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

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

    # Layer-wise Fisher storage (Dict[str, float] per ratio/corruption)
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
        "fold": lambda m, r: ViT_ModelFolding(m, compression_ratio=r),
        "mag-l2": lambda m, r: ViT_MagnitudePruning(m, compression_ratio=r, p=2),
        "wanda": lambda m, r: ViT_WandaPruning(m, compression_ratio=r),
    }

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
    print("Pre-TTA metrics computation: ", args.pre_tta_metrics_computation)
    print("Post-TTA metrics computation: ", args.post_tta_metrics_computation)
    print("CKA Analysis: ", args.cka_analysis)

    for seed in [2020]:
        set_seed(seed=seed)
        if hasattr(args, 'checkpoint') and args.checkpoint and os.path.isfile(args.checkpoint):
            model = timm.create_model(model_name=args.encoder_name, pretrained=False)
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"Loaded checkpoint: {args.checkpoint}")
        else:
            model = timm.create_model(model_name=args.encoder_name, pretrained=True)
        model = model.to(device=device)
        config = resolve_data_config({}, model=model)
        imagenet_preprocessing_validation_set = create_transform(**config, is_training=False)
        imagenet_preprocessing_training_set = create_transform(**config, is_training=True)

        origin_macs, origin_param = tp.utils.count_ops_and_params(model=model,
                                                                  example_inputs=torch.randn(1, 3, 224, 224).to(device))
        imagenet_c_dataset = ImageNet_C(encoder_name=args.encoder_name, args=args)

        test_loader = get_imagenet_vit(datadir=args.dataset_root.format(os.path.dirname(os.getcwd())), train=False,
                                       bs=128, transform=imagenet_preprocessing_validation_set)
        train_loader = get_imagenet_vit(datadir=args.dataset_root.format(os.path.dirname(os.getcwd())), train=True,
                                        bs=64, transform=imagenet_preprocessing_training_set)

        if args.debug:
            compression_ratios = [0.7]

        for ratio in compression_ratios:

            if args.prune_data == 'train' or args.method in ['mag-l2', 'fold']:
                if ratio == 0.0:
                    compressed_model = copy.deepcopy(model)
                else:
                    # Apply pruning/folding
                    if args.method != "taylor" and args.method != "hessian":
                        pruner = pruner_map[args.method](copy.deepcopy(model), ratio)
                        # Wanda needs calibration before apply()
                        if isinstance(pruner, ViT_WandaPruning):
                            pruner.run_calibration(train_loader, device, num_batches=50 if not args.debug else 2)
                        compressed_model = pruner.apply().to(device)
                    elif args.method == "taylor" or args.method == "hessian":
                        compressed_model = one_shot_structural_pruning_vit(model=copy.deepcopy(model),
                                                                           compression_method=args.method,
                                                                           train_loader=train_loader,
                                                                           compression_ratio=ratio,
                                                                           device=device,
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

            for corruption_idx, corruption in enumerate(corruptions_robustbench):

                if args.prune_data == 'test' and args.method not in ['mag-l2', 'fold']:
                    if ratio == 0.0:
                        compressed_model = copy.deepcopy(model)
                    else:
                        corruption_loader = imagenet_c_dataset.get_corruption_data_loader(
                            corruption=corruption, batch_size=args.m_sharpness, shuffle=True,
                            persistent_workers=False, pin_memory=True, num_workers=7
                        )

                        # Apply pruning using corruption data
                        if args.method != "taylor" and args.method != "hessian":
                            pruner = pruner_map[args.method](copy.deepcopy(model), ratio)
                            if isinstance(pruner, ViT_WandaPruning):
                                pruner.run_calibration(corruption_loader, device,
                                                       num_batches=50 if not args.debug else 2)
                            compressed_model = pruner.apply().to(device)
                        elif args.method == "taylor" or args.method == "hessian":
                            compressed_model = one_shot_structural_pruning_vit(model=copy.deepcopy(model),
                                                                               compression_method=args.method,
                                                                               train_loader=corruption_loader,
                                                                               compression_ratio=ratio,
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
                            persistent_workers=False,
                            pin_memory=True,
                            num_workers=7
                        )

                        ds_corruption = pack_to_gpu_from_loader(
                            loader=corruption_loader_pre_tta,
                            max_batches=args.eval_batches_num if not args.debug else 2,
                            device=device
                        )

                        temp_model_for_wrapper = copy.deepcopy(compressed_model)
                        temp_model_for_wrapper.eval()

                        _, temp_spa_wrapper = get_adaptation_optimizer_configure_model(
                            tta_method_name='spa',
                            model=temp_model_for_wrapper,
                            learning_rate=args.lr,
                            args=args,
                            device=device
                        )
                        spa_loss_pre = SPAConsistencyLoss(SPA(model=temp_spa_wrapper, optimizer=None),
                                                          bypass_predictor=True)

                        model_to_analyze_corr = copy.deepcopy(compressed_model).to(device)
                        spa_configure_model(model_to_analyze_corr)
                        model_to_analyze_corr.eval()

                        evals_corr, _ = get_hessian(device=device, model=copy.deepcopy(model_to_analyze_corr),
                                                    dataset=ds_corruption,
                                                    loss_fn=spa_loss_pre,
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
                        print(f"Error computing pre-TTA Hessian metrics: {e}")

                if args.fisher_analysis and args.pre_tta_metrics_computation:
                    # Create loader before try block to ensure .transform is available
                    corruption_loader_pre_tta = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption,
                        batch_size=args.m_sharpness,
                        shuffle=True,
                        persistent_workers=False,
                        pin_memory=True,
                        num_workers=7
                    )
                    try:

                        temp_model_for_wrapper = copy.deepcopy(compressed_model)
                        temp_model_for_wrapper.eval()

                        _, temp_spa_wrapper = get_adaptation_optimizer_configure_model(
                            tta_method_name='spa',
                            model=temp_model_for_wrapper,
                            learning_rate=args.lr,
                            args=args,
                            device=device
                        )

                        # bypass_predictor=True: the SPA predictor head is untrained before adaptation.
                        spa_loss_pre = SPAConsistencyLoss(
                            SPA(model=temp_spa_wrapper, optimizer=None),
                            bypass_predictor=True)

                        model_to_analyze = copy.deepcopy(temp_spa_wrapper.model).to(device)
                        spa_configure_model(model_to_analyze)
                        model_to_analyze.eval()

                        # Scalar Fisher (Pre-TTA)
                        fisher_result_corr = compute_fisher_trace_subspace(
                            device=device, model=copy.deepcopy(model_to_analyze),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(
                                corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                                persistent_workers=False, pin_memory=True, num_workers=7
                            ).dataset,
                            loss_fn=spa_loss_pre,
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
                                persistent_workers=False, pin_memory=True, num_workers=7
                            ).dataset,
                            loss_fn=spa_loss_pre,
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
                                persistent_workers=False, pin_memory=True, num_workers=7
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
                                persistent_workers=False, pin_memory=True, num_workers=7
                            ).dataset,
                            loss_fn=spa_loss_pre,
                            max_samples=128 if not args.debug else 10,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        snr_pre_corr[ratio][corruption_idx].append(snr_val)

                        # Layer-wise Gradient SNR (Pre-TTA)
                        lw_snr_val = compute_layerwise_gradient_snr(
                            device=device, model=copy.deepcopy(compressed_model).to(device),
                            dataset=imagenet_c_dataset.get_corruption_data_loader(
                                corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                                persistent_workers=False, pin_memory=True, num_workers=7
                            ).dataset,
                            loss_fn=spa_loss_pre,
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
                                persistent_workers=False, pin_memory=True, num_workers=7
                            ).dataset,
                            loss_fn=spa_loss_pre,
                            max_samples=128 if not args.debug else 10,
                            n_hutchinson_iters=10 if not args.debug else 2,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)
                        layerwise_hessian_trace_pre_corr[ratio][corruption_idx].append(
                            get_sorted_layer_hessian_trace_list(lw_hess_trace_val))

                        del corruption_loader_pre_tta
                        del model_to_analyze
                        gc.collect()

                    except Exception as e:
                        print(f"Error computing pre-TTA Fisher metrics: {e}")

                # Pre-TTA gradient alignment (cosine similarity)
                if args.gradient_alignment and args.pre_tta_metrics_computation:
                    corruption_loader_ga = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption,
                        batch_size=args.m_sharpness,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=6
                    )
                    try:
                        temp_model_for_wrapper = copy.deepcopy(compressed_model)
                        temp_model_for_wrapper.eval()

                        # Matched oracle per TTA method: SPAConsistencyLoss vs OracleSPALoss, SAR_EntropyLoss vs cross-entropy.
                        if args.tta_method == 'spa':
                            _, temp_spa_wrapper = get_adaptation_optimizer_configure_model(
                                tta_method_name='spa',
                                model=temp_model_for_wrapper,
                                learning_rate=args.lr,
                                args=args,
                                device=device
                            )
                            _spa_for_ga = SPA(model=temp_spa_wrapper, optimizer=None)
                            tta_loss_for_ga = SPAConsistencyLoss(
                                _spa_for_ga, bypass_predictor=True)
                            oracle_loss_for_ga = OracleSPALoss(
                                _spa_for_ga, bypass_predictor=True)
                            model_for_ga = copy.deepcopy(temp_spa_wrapper.model).to(device)
                            spa_configure_model(model_for_ga)
                        elif args.tta_method == 'sar':
                            tta_loss_for_ga = SAR_EntropyLoss(
                                margin_e0=0.4 * math.log(1000),
                                fallback_on_empty=False)
                            oracle_loss_for_ga = F.cross_entropy
                            model_for_ga = copy.deepcopy(temp_model_for_wrapper).to(device)
                            sharpness.configure_model_for_sar_sharpness(model_for_ga)
                        else:
                            raise ValueError(
                                f"PRE-TTA gradient alignment supports tta_method "
                                f"in {{spa, sar}}; got {args.tta_method}.")
                        model_for_ga.eval()

                        ga_result = compute_gradient_alignment(
                            device=device,
                            model=model_for_ga,
                            dataset=corruption_loader_ga.dataset,
                            tta_loss_fn=tta_loss_for_ga,
                            oracle_loss_fn=oracle_loss_for_ga,
                            physical_batch_size=args.m_sharpness,
                            max_samples=None,
                            args=args,
                            preprocess=corruption_loader_ga.transform)
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
                if args.loss_barrier and args.pre_tta_metrics_computation:
                    try:
                        # Get dense pretrained state dict (θ₀)
                        dense_pretrained_sd = copy.deepcopy(model.state_dict())
                        # Get pruned pre-TTA state dict (φ_pre)
                        pruned_pre_tta_sd = copy.deepcopy(compressed_model.state_dict())

                        # Dense TTA state dict for this corruption, computed once and cached.
                        # FOA/NORM do not update weights, so weight-interpolation barriers are skipped.
                        if args.tta_method in ["foa", "norm"]:
                            pass
                        elif corruption not in dense_tta_state_dicts:
                            dense_model_for_tta = copy.deepcopy(model).to(device)
                            optimizer_dense, spa_model_dense = get_adaptation_optimizer_configure_model(
                                tta_method_name=args.tta_method,
                                model=dense_model_for_tta,
                                learning_rate=0.001 if args.tta_method == "tent" else 0.00025,
                                args=args, device=device
                            )
                            if spa_model_dense is not None:
                                dense_model_for_tta = spa_model_dense
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
                            dense_tta_state_dicts[corruption] = copy.deepcopy(adapted_dense_model.model.state_dict())
                            del dense_model_for_tta, optimizer_dense, adapted_dense_model
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
                                                          pin_memory=True, num_workers=7),
                                                      device=device,
                                                      num_epochs=1,
                                                      is_main_process=True,
                                                      debug=args.debug,
                                                      hook_layer_types=VIT_HOOK_LAYER_TYPES)
                        cka_matrix_corr = cka_calc_corr.calculate_cka_matrix()
                        cka_diagonal_corr = cka_matrix_corr.diagonal().tolist()
                        cka_dense_vs_pruned_pre_tta_corr[ratio][corruption_idx].append(cka_diagonal_corr)
                        cka_calc_corr.reset()
                        del cka_calc_corr
                        gc.collect()

                    except Exception as e:
                        print(f"Error computing Pre-TTA CKA metrics: {e}")

                if args.prompt_entropy and args.pre_tta_metrics_computation:
                    gc.collect()

                    pe_loader = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                        persistent_workers=False, pin_memory=True, num_workers=7
                    )
                    pe_pre_vec = compute_vit_dataset_entropy(
                        model_or_wrapper=copy.deepcopy(compressed_model),
                        data_loader=pe_loader,
                        device=device,
                        max_batches=args.pe_max_batches,  # default 20
                        token_pool=getattr(args, 'pe_token_pool', 'cls'),
                        alpha=getattr(args, 'pe_alpha', 1.0),
                        normalization=getattr(args, 'pe_norm', 'maxEntropy'),
                        entropy_mode=getattr(args, 'pe_entropy_mode', 'dataset'),
                    )
                    activation_map_entropy_pre_corruption[ratio][corruption_idx].append(pe_pre_vec)
                    del pe_loader
                    gc.collect()

                if args.sharpness and args.pre_tta_metrics_computation:
                    try:
                        pre_tta_compressed_model = copy.deepcopy(corruption_adapted_model)
                        spa_configure_model(pre_tta_compressed_model)
                        pre_tta_compressed_model.eval()

                        _, temp_spa_wrapper = get_adaptation_optimizer_configure_model(
                            tta_method_name='spa', model=copy.deepcopy(pre_tta_compressed_model),
                            args=args, device=device
                        )

                        loss_fn_pre = SPAConsistencyLoss(spa_wrapper=SPA(model=temp_spa_wrapper, optimizer=None),
                                                         bypass_predictor=True)

                        batches_sharpness = imagenet_c_dataset.get_corruption_data_loader(
                            corruption=corruption, batch_size=args.m_sharpness,
                            shuffle=False, persistent_workers=False, pin_memory=True, num_workers=7
                        )

                        sharpness_obj_pre_tta_corrupted_data, _, _, _ = sharpness.eval_APGD_sharpness(
                            model=copy.deepcopy(pre_tta_compressed_model),
                            device=device, batches=batches_sharpness,
                            loss_f=loss_fn_pre,
                            rho=0.002, n_iters=20, n_restarts=1, step_size_mult=1.0,
                            debug=args.debug, rand_init=False, no_grad_norm=False,
                            eval_batches_num=args.eval_batches_num if not args.debug else 2,
                            verbose=False, return_output=True, adaptive=True, version='default', norm='linf'
                        )
                        avg_loss_sharpness_pre_tta_per_ratio_corrupted_data[ratio][corruption_idx].append(
                            float(sharpness_obj_pre_tta_corrupted_data))
                        del batches_sharpness
                        train_loader = reused_train_loader
                        sharpness_obj_pre_tta_training_data, sharpness_err_pre_tta_training_data, _, output_pre_tta_training_data = sharpness.eval_APGD_sharpness(
                            model=copy.deepcopy(pre_tta_compressed_model), device=device, batches=train_loader,
                            loss_f=train_loss_fn,
                            rho=0.002, n_iters=20, n_restarts=1,
                            step_size_mult=1.0,
                            debug=args.debug,
                            rand_init=False, no_grad_norm=False,
                            verbose=False, return_output=True, adaptive=True, version='default',
                            norm='linf')

                        avg_loss_sharpness_pre_tta_per_ratio_training_data[ratio][corruption_idx].append(
                            float(sharpness_obj_pre_tta_training_data))
                        gc.collect()
                    except Exception as e:
                        print(f"Error computing sharpness pre TTA: {e}")

                # Build the TTA learner and adapt.
                if args.pre_tta_only:
                    continue  # skip TTA adaptation and all post-TTA computation

                optimizer_corruption, spa_model = get_adaptation_optimizer_configure_model(
                    tta_method_name=args.tta_method,
                    model=corruption_adapted_model,
                    learning_rate=0.001,
                    args=args, device=device
                )

                if spa_model is not None:
                    corruption_adapted_model = spa_model

                # FOA, NORM, NoAdapt, LAME and PEA are returned already wrapped and skip learner.make().
                if args.tta_method == "foa":
                    adapted_model_corruption = corruption_adapted_model
                    # FOA estimates its source statistics on the ImageNet validation set (Niu et al., 2024).
                    adapted_model_corruption.obtain_origin_stat(test_loader, device=device)
                elif args.tta_method in ("norm", "no_adapt", "lame", "pea_resnet18", "pea_vit"):
                    adapted_model_corruption = corruption_adapted_model
                    # PEA precomputes per-block source statistics on four training batches.
                    if args.tta_method in ("pea_resnet18", "pea_vit"):
                        adapted_model_corruption.precompute_source_stats(
                            train_loader, device, max_batches=4
                        )
                else:
                    adapted_model_corruption = learner.make(
                        name=args.tta_method,
                        model=corruption_adapted_model,
                        optimizer=optimizer_corruption
                    )

                # Oracle: pass labels so forward_and_adapt uses supervised cross-entropy
                top1acc, _ = imagenet_c_dataset.evaluate_model_all_corruption_data(
                    model=adapted_model_corruption, device=device,
                    batch_size=64, corruption=corruption, args=args,
                    pass_labels=(args.tta_method in ("oracle", "oracle_spa"))
                )

                results_per_ratio[ratio][corruption_idx].append(top1acc)


                if args.hessian_analysis and args.post_tta_metrics_computation:
                    try:

                        current_spa_wrapper = copy.deepcopy(adapted_model_corruption.model)
                        spa_loss_post = SPAConsistencyLoss(spa_wrapper=SPA(model=current_spa_wrapper, optimizer=None),
                                                           bypass_predictor=False)

                        corruption_loader_pre_tta = imagenet_c_dataset.get_corruption_data_loader(
                            corruption=corruption,
                            batch_size=args.m_sharpness,
                            shuffle=True,
                            persistent_workers=False,
                            pin_memory=True,
                            num_workers=7
                        )

                        ds_corruption = pack_to_gpu_from_loader(
                            loader=corruption_loader_pre_tta,
                            max_batches=args.eval_batches_num if not args.debug else 2,
                            device=device
                        )

                        model_to_analyze_corr = copy.deepcopy(current_spa_wrapper.model).to(device)
                        model_to_analyze_corr.eval()
                        spa_configure_model(model_to_analyze_corr)

                        evals_corr, _ = get_hessian(device=device, model=copy.deepcopy(model_to_analyze_corr),
                                                    dataset=ds_corruption,
                                                    loss_fn=spa_loss_post,
                                                    neigs=1,
                                                    physical_batch_size=args.m_sharpness, exclude_ln=False, args=args,
                                                    preprocess=corruption_loader_pre_tta.transform)
                        hess_val_corr = evals_corr[0].item()
                        hess_post_corr_top2[ratio][corruption_idx].append(hess_val_corr)

                    except Exception as e:
                        print(f"Error computing corruption metrics: {e}")

                if args.fisher_analysis and args.post_tta_metrics_computation:
                    # Create loader before try block to ensure .transform is available
                    corruption_loader_pre_tta = imagenet_c_dataset.get_corruption_data_loader(
                            corruption=corruption,
                            batch_size=args.m_sharpness,
                            shuffle=True,
                            persistent_workers=False,
                            pin_memory=True,
                            num_workers=7
                    )
                    try:
                        current_spa_wrapper = copy.deepcopy(adapted_model_corruption.model)
                        spa_loss_post = SPAConsistencyLoss(spa_wrapper=SPA(model=current_spa_wrapper, optimizer=None),
                                                               bypass_predictor=False)
                        # Layer-wise Fisher (Post-TTA)
                        lw_fisher_post = compute_layerwise_fisher(
                                device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                                dataset=imagenet_c_dataset.corruptions_data_handlers[corruption],
                                loss_fn=spa_loss_post,
                                max_samples=500 if not args.debug else 10,
                                args=args,
                                preprocess=corruption_loader_pre_tta.transform)
                        layerwise_fisher_post_corr[ratio][corruption_idx].append(
                                get_sorted_layer_fisher_list(lw_fisher_post))

                        model_to_analyze_corr = copy.deepcopy(current_spa_wrapper.model).to(device)
                        model_to_analyze_corr.eval()
                        spa_configure_model(model_to_analyze_corr)
                        fisher_result_corr = compute_fisher_trace_subspace(
                            device=device, model=copy.deepcopy(model_to_analyze_corr),
                            dataset=imagenet_c_dataset.corruptions_data_handlers[corruption],
                            loss_fn=spa_loss_post,
                            physical_batch_size=args.m_sharpness,
                            args=args,
                            preprocess=corruption_loader_pre_tta.transform)

                        del corruption_loader_pre_tta
                        del model_to_analyze_corr
                        gc.collect()
                        fisher_post_corr[ratio][corruption_idx].append(fisher_result_corr["raw"])
                        fisher_norm_post_corr[ratio][corruption_idx].append(fisher_result_corr["normalized"])

                        # ECE (Post-TTA)
                        ece_val_post = compute_ece(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=imagenet_c_dataset.corruptions_data_handlers[corruption],
                            max_samples=500 if not args.debug else 10,
                            args=args,
                            preprocess=imagenet_c_dataset._get_gpu_transform(device='cuda'))
                        ece_post_corr[ratio][corruption_idx].append(ece_val_post)

                        # Gradient SNR (Post-TTA)
                        snr_val_post = compute_gradient_snr(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=imagenet_c_dataset.corruptions_data_handlers[corruption],
                            loss_fn=spa_loss_post,
                            max_samples=128 if not args.debug else 10,
                            args=args,
                            preprocess=imagenet_c_dataset._get_gpu_transform(device='cuda'))
                        snr_post_corr[ratio][corruption_idx].append(snr_val_post)

                        # Layer-wise Gradient SNR (Post-TTA)
                        lw_snr_val_post = compute_layerwise_gradient_snr(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=imagenet_c_dataset.corruptions_data_handlers[corruption],
                            loss_fn=spa_loss_post,
                            max_samples=128 if not args.debug else 10,
                            args=args,
                            preprocess=imagenet_c_dataset._get_gpu_transform(device='cuda'))
                        layerwise_snr_post_corr[ratio][corruption_idx].append(
                            get_sorted_layer_snr_list(lw_snr_val_post))

                        # Layer-wise Hessian Trace (Post-TTA)
                        lw_hess_trace_val_post = compute_layerwise_hessian_trace(
                            device=device, model=copy.deepcopy(adapted_model_corruption.model).to(device),
                            dataset=imagenet_c_dataset.corruptions_data_handlers[corruption],
                            loss_fn=spa_loss_post,
                            max_samples=128 if not args.debug else 10,
                            n_hutchinson_iters=10 if not args.debug else 2,
                            args=args,
                            preprocess=imagenet_c_dataset._get_gpu_transform(device='cuda'))
                        layerwise_hessian_trace_post_corr[ratio][corruption_idx].append(
                            get_sorted_layer_hessian_trace_list(lw_hess_trace_val_post))
                    except Exception as e:
                        print(f"Error computing corruption metrics: {e}")

                # Post-TTA gradient alignment (cosine similarity)
                if args.gradient_alignment and args.post_tta_metrics_computation:
                    corruption_loader_ga_post = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption,
                        batch_size=args.m_sharpness,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=6
                    )
                    try:
                        current_spa_wrapper_ga = copy.deepcopy(adapted_model_corruption.model)
                        spa_loss_ga_post = SPAConsistencyLoss(
                            spa_wrapper=SPA(model=current_spa_wrapper_ga, optimizer=None),
                            bypass_predictor=False)

                        model_for_ga_post = copy.deepcopy(current_spa_wrapper_ga.model).to(device)
                        spa_configure_model(model_for_ga_post)
                        model_for_ga_post.eval()

                        ga_result_post = compute_gradient_alignment(
                            device=device,
                            model=model_for_ga_post,
                            dataset=corruption_loader_ga_post.dataset,
                            tta_loss_fn=spa_loss_ga_post,
                            oracle_loss_fn=F.cross_entropy,
                            physical_batch_size=args.m_sharpness,
                            max_samples=None,
                            args=args,
                            preprocess=corruption_loader_ga_post.transform)
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
                if args.loss_barrier and args.post_tta_metrics_computation:
                    # FOA/NORM do not update weights, so weight-interpolation barriers are skipped.
                    if args.tta_method in ["foa", "norm"]:
                        pass
                    else:
                        try:
                            # Get dense TTA-adapted state dict (θ^post_dense) for this corruption
                            dense_tta_sd = dense_tta_state_dicts[corruption]
                            # Get pruned post-TTA state dict (φ_post)
                            pruned_post_tta_sd = copy.deepcopy(adapted_model_corruption.model.state_dict())

                            # Dense TTA should already be cached from pre-TTA computation
                            if corruption not in dense_tta_state_dicts:
                                dense_model_for_tta = copy.deepcopy(model).to(device)
                                optimizer_dense, spa_model_dense = get_adaptation_optimizer_configure_model(
                                    tta_method_name=args.tta_method,
                                    model=dense_model_for_tta,
                                    learning_rate=0.001 if args.tta_method == "tent" else 0.00025,
                                    args=args, device=device
                                )
                                if spa_model_dense is not None:
                                    dense_model_for_tta = spa_model_dense
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
                                dense_tta_state_dicts[corruption] = copy.deepcopy(adapted_dense_model.model.state_dict())
                                dense_tta_sd = dense_tta_state_dicts[corruption]
                                del dense_model_for_tta, optimizer_dense, adapted_dense_model
                                gc.collect()

                            # ID Barrier: B_ID(θ^post_dense, φ_post) on clean validation data
                            # Measures connectivity between dense TTA-adapted and pruned TTA-adapted basins
                            id_barrier_val, id_barrier_details = get_id_barrier(
                                model_dense=copy.deepcopy(model),
                                sd_dense_pretrained=dense_tta_sd,
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

                        if ratio == 0.0:
                            dense_adapted_models_cka.append(copy.deepcopy(adapted_model_corruption.model))

                        # Pruned Pre-TTA vs Post-TTA on Corruption Data
                        cka_calc_corr = CKACalculator(model1=copy.deepcopy(dense_adapted_models_cka[corruption_idx]),
                                                      model2=copy.deepcopy(adapted_model_corruption.model),
                                                      dataloader=imagenet_c_dataset.get_corruption_data_loader(
                                                          corruption=corruption,
                                                          batch_size=args.m_sharpness,
                                                          shuffle=False,
                                                          persistent_workers=False,
                                                          pin_memory=True, num_workers=7),
                                                      device=device,
                                                      debug=args.debug,
                                                      num_epochs=1,
                                                      is_main_process=True,
                                                      hook_layer_types=VIT_HOOK_LAYER_TYPES)
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
                        current_spa_wrapper = adapted_model_corruption.model
                        post_tta_compressed_model = copy.deepcopy(current_spa_wrapper.model)
                        post_tta_compressed_model.eval()
                        spa_configure_model(post_tta_compressed_model)

                        spa_loss_post = SPAConsistencyLoss(spa_wrapper=SPA(model=current_spa_wrapper, optimizer=None),
                                                           bypass_predictor=False)

                        batches_corruption = imagenet_c_dataset.get_corruption_data_loader(corruption=corruption,
                                                                                           batch_size=args.m_sharpness,
                                                                                           shuffle=False,
                                                                                           persistent_workers=False,
                                                                                           pin_memory=True,
                                                                                           num_workers=7)

                        sharpness_corruption_post_tta, sharpness_corruption_err_post_tta, _, output_post_tta = sharpness.eval_APGD_sharpness(
                            model=copy.deepcopy(post_tta_compressed_model), batches=batches_corruption,
                            loss_f=spa_loss_post,
                            device=device,
                            rho=0.002, n_iters=20, n_restarts=1,
                            step_size_mult=1.0,
                            debug=args.debug,
                            eval_batches_num=args.eval_batches_num if not args.debug else 2,
                            rand_init=False, no_grad_norm=False,
                            verbose=False, return_output=True, adaptive=True, version='default',
                            norm='linf')
                        avg_loss_sharpness_post_tta_per_ratio_corrupted_data[ratio][corruption_idx].append(
                            sharpness_corruption_post_tta)
                        del batches_corruption
                        gc.collect()
                        # Sharpness on training data without augmentation, following Andriushchenko et al. (2023).
                        train_loader = reused_train_loader
                        sharpness_post_tta_training_data, sharpness_err_post_tta_training_data, _, output_post_tta_training_data = sharpness.eval_APGD_sharpness(
                            model=copy.deepcopy(post_tta_compressed_model), device=device, batches=train_loader,
                            loss_f=train_loss_fn,
                            rho=0.002, n_iters=20, n_restarts=1,
                            step_size_mult=1.0,
                            debug=args.debug,
                            eval_batches_num=args.eval_batches_num if not args.debug else 2,
                            rand_init=False, no_grad_norm=False,
                            verbose=False, return_output=True, adaptive=True, version='default',
                            norm='linf')

                        avg_loss_sharpness_post_tta_per_ratio_training_data[ratio][corruption_idx].append(
                            sharpness_post_tta_training_data)
                    except Exception as e:
                        print(f"Error computing sharpness post TTA: {e}")

                if args.prompt_entropy and args.post_tta_metrics_computation:
                    pe_loader = imagenet_c_dataset.get_corruption_data_loader(
                        corruption=corruption, batch_size=args.m_sharpness, shuffle=False,
                        persistent_workers=False, pin_memory=True, num_workers=7
                    )
                    pe_post_corruption = compute_vit_dataset_entropy(
                        model_or_wrapper=copy.deepcopy(adapted_model_corruption.model),
                        data_loader=pe_loader,
                        device=device,
                        max_batches=args.pe_max_batches,
                        token_pool=getattr(args, 'pe_token_pool', 'cls'),
                        alpha=getattr(args, 'pe_alpha', 1.0),
                        normalization=getattr(args, 'pe_norm', 'maxEntropy'),
                        entropy_mode=getattr(args, 'pe_entropy_mode', 'dataset'),
                    )
                    activation_map_entropy_post_corruption[ratio][corruption_idx].append(pe_post_corruption)

            print(f"---- Completed evaluations for compression ratio: {ratio}. ----")

        print("\n" + "=" * 50)
        print("FINAL RESULTS MAPS (Ascending Sparsity Ratios)")
        print("=" * 50)

        diagnostic_metrics = {}

        try:
            acc_pre_map = format_metric_map(compressed_corruptions_acc_per_ratio, compression_ratios,
                                            corruptions_robustbench)
            print(
                f"vit_base_{args.method}_calibration_{args.prune_data}_pre_adaptations_accuracy_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(acc_pre_map)}")

            acc_post_map = format_metric_map(results_per_ratio, compression_ratios, corruptions_robustbench)
            print(
                f"vit_base_{args.method}_calibration_{args.prune_data}_post_adaptations_accuracy_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(acc_post_map)}")
        except Exception as e:
            print(f"Error printing accuracy maps: {e}")

        if args.pre_tta_metrics_computation:
            print("Pre-TTA Metrics (Ascending Sparsity Ratios)")
            if args.sharpness:
                try:
                    # Pre-TTA Loss
                    sharp_loss_pre_map = format_metric_map(avg_loss_sharpness_pre_tta_per_ratio_corrupted_data,
                                                           compression_ratios, corruptions_robustbench)
                    print(
                        f"vit_base_{args.method}_calibration_{args.prune_data}_pre_adaptations_map_sharpness_corruption_loss_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(sharp_loss_pre_map)}")

                    # Pre-TTA Loss Training
                    sharp_loss_training_pre_map = format_metric_map(avg_loss_sharpness_pre_tta_per_ratio_training_data,
                                                                    compression_ratios, corruptions_robustbench)
                    print(
                        f"vit_base_{args.method}_calibration_{args.prune_data}_pre_adaptations_map_sharpness_training_loss_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(sharp_loss_training_pre_map)}")

                except Exception as e:
                    print(f"Error printing sharpness maps: {e}")

            if args.prompt_entropy:
                try:

                    pe_pre_map = format_metric_map(activation_map_entropy_pre_corruption, compression_ratios,
                                                   corruptions_robustbench)
                    print(
                        f"vit_base_{args.method}_calibration_{args.prune_data}_pre_adaptations_map_prompt_entropy_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(pe_pre_map)}")
                    diagnostic_metrics['ame_pre'] = format_numpy_floats_map(pe_pre_map)

                except Exception as e:
                    print(f"Error printing prompt entropy maps: {e}")

            if args.hessian_analysis:
                try:
                    hess_pre_corr_map = format_metric_map(hess_pre_corr_top2, compression_ratios,
                                                          corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_hessian_eigenvalues_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(hess_pre_corr_map)}")
                except Exception as e:
                    print(f"Error printing Hessian maps: {e}")

            if args.fisher_analysis:
                try:
                    # Pre-TTA Fisher Trace (Corruption)
                    fisher_pre_corr_map = format_metric_map(fisher_pre_corr, compression_ratios,
                                                            corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_fisher_trace_in_U_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_pre_corr_map)}")

                    # Pre-TTA Normalized Fisher (per-parameter)
                    fisher_norm_pre_corr_map = format_metric_map(fisher_norm_pre_corr, compression_ratios,
                                                                 corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_fisher_trace_normalized_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_norm_pre_corr_map)}")

                    # Pre-TTA Layer-wise Fisher (Corruption)
                    lw_fisher_pre_corr_map = format_metric_map(layerwise_fisher_pre_corr, compression_ratios,
                                                               corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_layerwise_fisher_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_fisher_pre_corr_map)}")

                    # Pre-TTA ECE (Corruption)
                    ece_pre_corr_map = format_metric_map(ece_pre_corr, compression_ratios,
                                                         corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_ece_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ece_pre_corr_map)}")

                    # Pre-TTA Gradient SNR (Corruption)
                    snr_pre_corr_map = format_metric_map(snr_pre_corr, compression_ratios,
                                                         corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_gradient_snr_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(snr_pre_corr_map)}")

                    # Pre-TTA Layer-wise Gradient SNR (Corruption)
                    lw_snr_pre_corr_map = format_metric_map(layerwise_snr_pre_corr, compression_ratios,
                                                            corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_layerwise_gradient_snr_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_snr_pre_corr_map)}")

                    # Pre-TTA Layer-wise Hessian Trace (Corruption)
                    lw_hess_trace_pre_corr_map = format_metric_map(layerwise_hessian_trace_pre_corr, compression_ratios,
                                                                   corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_layerwise_hessian_trace_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_hess_trace_pre_corr_map)}")
                except Exception as e:
                    print(f"Error printing Fisher maps: {e}")

            if args.gradient_alignment:
                try:
                    ga_pre_corr_map = format_metric_map(grad_align_pre_corr, compression_ratios,
                                                        corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_gradient_alignment_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ga_pre_corr_map)}")
                    diagnostic_metrics['ga_pre'] = format_numpy_floats_map(ga_pre_corr_map)
                except Exception as e:
                    print(f"Error printing pre-TTA Gradient Alignment maps: {e}")

                try:
                    gn_tta_pre_map = format_metric_map(grad_norm_tta_pre_corr, compression_ratios,
                                                       corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_grad_norm_tta_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(gn_tta_pre_map)}")
                    gn_oracle_pre_map = format_metric_map(grad_norm_oracle_pre_corr, compression_ratios,
                                                          corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_pre_adaptations_grad_norm_oracle_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(gn_oracle_pre_map)}")
                    diagnostic_metrics['gn_tta_pre'] = format_numpy_floats_map(gn_tta_pre_map)
                    diagnostic_metrics['gn_oracle_pre'] = format_numpy_floats_map(gn_oracle_pre_map)
                except Exception as e:
                    print(f"Error printing pre-TTA Gradient Norm maps: {e}")

            if args.cka_analysis:
                try:
                    # Dense vs Pruned (Pre-TTA)
                    cka_pre_corr_map = format_metric_map(cka_dense_vs_pruned_pre_tta_corr, compression_ratios,
                                                         corruptions_robustbench)
                    print(
                        f"vit_{args.method}_calibration_{args.prune_data}_pre_adaptations_cka_dense_vs_pruned_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(cka_pre_corr_map)}")
                    diagnostic_metrics['cka_pre'] = format_numpy_floats_map(cka_pre_corr_map)

                except Exception as e:
                    print(f"Error printing Pre-TTA CKA maps: {e}")

        if args.post_tta_metrics_computation:

            print("=" * 50 + "\n")
            print("Post-TTA Metrics (Ascending Sparsity Ratios)")
            if args.sharpness:
                try:
                    sharp_loss_post_map = format_metric_map(avg_loss_sharpness_post_tta_per_ratio_corrupted_data,
                                                            compression_ratios, corruptions_robustbench)
                    print(
                        f"vit_{args.method}_calibration_{args.prune_data}_post_adaptations_map_sharpness_corruption_loss_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(sharp_loss_post_map)}")

                    sharp_loss_train_post_map = format_metric_map(avg_loss_sharpness_post_tta_per_ratio_training_data,
                                                                 compression_ratios, corruptions_robustbench)
                    print(
                        f"vit_{args.method}_calibration_{args.prune_data}_post_adaptations_map_sharpness_training_loss_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(sharp_loss_train_post_map)}")

                except Exception as e:
                    print(f"Error printing post-TTA sharpness maps: {e}")

            if args.prompt_entropy:
                try:
                    pe_post_map = format_metric_map(activation_map_entropy_post_corruption, compression_ratios,
                                                    corruptions_robustbench)
                    print(
                        f"vit_{args.method}_post_adaptations_map_prompt_entropy_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(pe_post_map)}")
                    diagnostic_metrics['ame_post'] = format_numpy_floats_map(pe_post_map)

                except Exception as e:
                    print(f"Error printing post-TTA prompt entropy maps: {e}")

            if args.hessian_analysis:
                try:
                    hess_post_corr_map = format_metric_map(hess_post_corr_top2, compression_ratios,
                                                           corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_hessian_eigenvalues_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(hess_post_corr_map)}")
                except Exception as e:
                    print(f"Error printing post-TTA Hessian maps: {e}")

            if args.fisher_analysis:
                try:
                    fisher_post_corr_map = format_metric_map(fisher_post_corr, compression_ratios,
                                                             corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_fisher_trace_in_U_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_post_corr_map)}")

                    # Post-TTA Normalized Fisher (per-parameter)
                    fisher_norm_post_corr_map = format_metric_map(fisher_norm_post_corr, compression_ratios,
                                                                  corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_fisher_trace_normalized_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(fisher_norm_post_corr_map)}")

                    # Post-TTA Layer-wise Fisher (Corruption)
                    lw_fisher_post_corr_map = format_metric_map(layerwise_fisher_post_corr, compression_ratios,
                                                                corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_layerwise_fisher_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_fisher_post_corr_map)}")

                    # Post-TTA ECE (Corruption)
                    ece_post_corr_map = format_metric_map(ece_post_corr, compression_ratios,
                                                          corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_ece_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ece_post_corr_map)}")

                    # Post-TTA Gradient SNR (Corruption)
                    snr_post_corr_map = format_metric_map(snr_post_corr, compression_ratios,
                                                          corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_gradient_snr_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(snr_post_corr_map)}")

                    # Post-TTA Layer-wise Gradient SNR (Corruption)
                    lw_snr_post_corr_map = format_metric_map(layerwise_snr_post_corr, compression_ratios,
                                                             corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_layerwise_gradient_snr_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_snr_post_corr_map)}")

                    # Post-TTA Layer-wise Hessian Trace (Corruption)
                    lw_hess_trace_post_corr_map = format_metric_map(layerwise_hessian_trace_post_corr, compression_ratios,
                                                                    corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_layerwise_hessian_trace_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(lw_hess_trace_post_corr_map)}")
                except Exception as e:
                    print(f"Error printing post-TTA Fisher maps: {e}")

            if args.gradient_alignment:
                try:
                    ga_post_corr_map = format_metric_map(grad_align_post_corr, compression_ratios,
                                                         corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_gradient_alignment_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ga_post_corr_map)}")
                    diagnostic_metrics['ga_post'] = format_numpy_floats_map(ga_post_corr_map)
                except Exception as e:
                    print(f"Error printing post-TTA Gradient Alignment maps: {e}")

                try:
                    gn_tta_post_map = format_metric_map(grad_norm_tta_post_corr, compression_ratios,
                                                        corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_grad_norm_tta_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(gn_tta_post_map)}")
                    gn_oracle_post_map = format_metric_map(grad_norm_oracle_post_corr, compression_ratios,
                                                           corruptions_robustbench)
                    print(
                        f"vit_imagenet_{args.method}_calibration_{args.prune_data}_post_adaptations_grad_norm_oracle_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(gn_oracle_post_map)}")
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
                        f"vit_{args.method}_calibration_{args.prune_data}_post_adaptations_cka_pruned_pre_vs_post_corruption_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(cka_post_corr_map)}")
                    diagnostic_metrics['cka_post'] = format_numpy_floats_map(cka_post_corr_map)

                except Exception as e:
                    print(f"Error printing Post-TTA CKA maps: {e}")

            # Loss barrier (ID and OOD)
            if args.loss_barrier:
                try:
                    # Pre-TTA ID Barrier
                    id_barrier_pre_map = format_metric_map(id_barrier_pre_corr, compression_ratios,
                                                           corruptions_robustbench)
                    print(
                        f"vit_{args.method}_calibration_{args.prune_data}_pre_adaptations_id_barrier_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(id_barrier_pre_map)}")

                    # Pre-TTA OOD Barrier
                    ood_barrier_pre_map = format_metric_map(ood_barrier_pre_corr, compression_ratios,
                                                            corruptions_robustbench)
                    print(
                        f"vit_{args.method}_calibration_{args.prune_data}_pre_adaptations_ood_barrier_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ood_barrier_pre_map)}")

                    # Post-TTA ID Barrier
                    id_barrier_post_map = format_metric_map(id_barrier_post_corr, compression_ratios,
                                                            corruptions_robustbench)
                    print(
                        f"vit_{args.method}_calibration_{args.prune_data}_post_adaptations_id_barrier_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(id_barrier_post_map)}")

                    # Post-TTA OOD Barrier
                    ood_barrier_post_map = format_metric_map(ood_barrier_post_corr, compression_ratios,
                                                             corruptions_robustbench)
                    print(
                        f"vit_{args.method}_calibration_{args.prune_data}_post_adaptations_ood_barrier_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ood_barrier_post_map)}")

                    # Full Barrier Profiles (barriers at each alpha in [0, 0.1, ..., 1.0])
                    # Pre-TTA ID Barrier Profile
                    id_barrier_profile_pre_map = format_metric_map(id_barrier_profile_pre_corr, compression_ratios,
                                                                   corruptions_robustbench)
                    print(
                        f"vit_{args.method}_calibration_{args.prune_data}_pre_adaptations_id_barrier_profile_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(id_barrier_profile_pre_map)}")

                    # Pre-TTA OOD Barrier Profile
                    ood_barrier_profile_pre_map = format_metric_map(ood_barrier_profile_pre_corr, compression_ratios,
                                                                    corruptions_robustbench)
                    print(
                        f"vit_{args.method}_calibration_{args.prune_data}_pre_adaptations_ood_barrier_profile_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ood_barrier_profile_pre_map)}")

                    # Post-TTA ID Barrier Profile
                    id_barrier_profile_post_map = format_metric_map(id_barrier_profile_post_corr, compression_ratios,
                                                                    corruptions_robustbench)
                    print(
                        f"vit_{args.method}_calibration_{args.prune_data}_post_adaptations_id_barrier_profile_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(id_barrier_profile_post_map)}")

                    # Post-TTA OOD Barrier Profile
                    ood_barrier_profile_post_map = format_metric_map(ood_barrier_profile_post_corr, compression_ratios,
                                                                     corruptions_robustbench)
                    print(
                        f"vit_{args.method}_calibration_{args.prune_data}_post_adaptations_ood_barrier_profile_{args.tta_method}_{args.dataset_name} = {format_numpy_floats_map(ood_barrier_profile_post_map)}")

                except Exception as e:
                    print(f"Error printing Loss Barrier maps: {e}")

    if args.gradient_alignment:
        try:
            from utils.ga_json import dump_ga_json
            ga_out_dir = os.path.join(
                'logs', 'ga_matched_oracle'
            )
            oracle_method = 'oracle_spa' if args.tta_method == 'spa' else 'oracle'
            if args.pre_tta_metrics_computation:
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
        compression_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        if args.debug:
            compression_ratios = [0.0, 0.5, 0.7]
        print_aggregated_results(
            all_results, compression_ratios, corruptions_robustbench,
            args, script_prefix="vit_imagenet")
        print_aggregated_diagnostic_metrics(
            all_results, corruptions_robustbench, args,
            script_prefix="vit_imagenet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ViT-Base compression + TTA on ImageNet-C")
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
    parser.add_argument("--checkpoint", type=str, default="pretrained/vit_base_patch16_224.pth")
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
    parser.add_argument('--sharpness', default=False, action='store_true',
                        help='Compute sharpness of the loss landscape')
    parser.add_argument('--eval_batches_num', default=32, type=int)
    parser.add_argument('--m_sharpness', default=64, type=int)
    parser.add_argument('--dataset_name', default='imagenet', type=str)
    parser.add_argument('--prompt_entropy', default=False, action='store_true',
                        help='Compute Gram-based prompt entropy (pre/post TTA) using batch_size=args.m_sharpness')
    # Layer selection for the entropy hooks.
    parser.add_argument('--pe_granularity', default='block', choices=['block', 'conv'],
                        help="Hook BasicBlock outputs ('block') or internal convs ('conv').")
    parser.add_argument('--pe_include_stem', default=False, action='store_true',
                        help='Also include the stem output after first ReLU.')
    parser.add_argument('--pe_include_avgpool', default=False, action='store_true',
                        help='Also include avgpool output.')
    parser.add_argument('--cka_analysis', default=False, action='store_true')
    parser.add_argument('--fisher_analysis', default=False, action='store_true')
    parser.add_argument('--loss_barrier', default=False, action='store_true',
                        help='Compute ID and OOD loss barriers for basin connectivity analysis')
    parser.add_argument('--hessian_analysis', default=False, action='store_true')
    parser.add_argument('--gradient_alignment', default=False, action='store_true',
                        help='Compute gradient cosine similarity between TTA and Oracle gradients')
    parser.add_argument('--pre_tta_only', default=False, action='store_true',
                        help='Skip TTA adaptation and all post-TTA computation. '
                             'Use to compute only pre-TTA metrics (e.g., pre-TTA gradient alignment) '
                             'without the cost of running the full adaptation loop.')
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

    args = parser.parse_args()
    main(args=args)
