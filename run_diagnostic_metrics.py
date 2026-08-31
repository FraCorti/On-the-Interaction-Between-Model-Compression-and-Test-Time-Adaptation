import argparse
import copy
import gc
import json
import math
import os
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import timm
from torchvision.models import resnet18 as tv_resnet18
from model_cifar10.resnet import ResNet18_Custom, BasicBlock_Custom
from train_smaller_rn18_cifar10 import ResNet_Custom_Narrow
from train_smaller_rn18_imagenet import ResNetTV_Narrow

from compression.fold import ResNet18_ModelFolding, ViT_ModelFolding
from compression.mag_prune import ResNet18_MagnitudePruning, ViT_MagnitudePruning
from compression.wanda import ResNet18_WandaPruning, ViT_WandaPruning
from utils.custom_structural_pruning import (
    one_shot_structural_pruning_resnet18,
    one_shot_structural_pruning_vit,
)

# Importing each module registers it via the @register decorator.
from utils.learner import sar, pea, spa, oracle, oracle_spa, no_adapt  # noqa: F401
from utils.learner.learner import make as make_learner
from utils.learner.set_optimizer import get_adaptation_optimizer_configure_model

from utils.cifar10_c import Cifar10_C, set_cifar10_corruptions_data_handlers
from utils.imagenet_c import (
    ImageNet_C, set_imagenet_corruptions_data_handlers, set_individual_corruption,
)
from utils.seed import set_seed

from utils.effective_rank import (
    compute_layerwise_effective_ranks,
    compute_layerwise_ame_paper_recipe,
)
from utils.metrics.msa import msa_distance

# Optional dependency; CKA is reported as NaN when not importable.
try:
    from cka import CKACalculator
except ImportError:
    CKACalculator = None


# Layer-wise diagnostics (AME, CKA) hook only the post-residual block outputs:
# BasicBlock for ResNet-18, Block for the timm ViT-Base (paper Sec. 3.2).
def post_residual_hook_targets(arch):
    if arch == 'rn18':
        from model_cifar10.resnet import BasicBlock_Custom
        from torchvision.models.resnet import BasicBlock as TVBasicBlock
        # BasicBlockTV (narrow ImageNet RN-18) is a distinct class from
        # torchvision's BasicBlock and must be registered explicitly.
        from train_smaller_rn18_imagenet import BasicBlockTV
        return (BasicBlock_Custom, TVBasicBlock, BasicBlockTV)
    from timm.models.vision_transformer import Block
    return (Block,)

SUPPORTED_METHODS = ('fold', 'taylor', 'hessian', 'wanda', 'mag-l2')
SCHEMES = ('dense_prune_tta', 'dense_prune_finetune_in_tta', 'smaller_dense_tta')
ADAPT_RN18 = ('NONE', 'SAR', 'SARCORR', 'PEA', 'oracle')
ADAPT_VIT = ('NONE', 'SAR', 'SARCORR', 'PEA', 'SPA', 'FOA', 'oracle', 'oracle_spa')

CORRUPTIONS_15 = (
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression',
)

# Map the --adapt value to the learner registry name.
def _learner_name(adapt: str, arch: str) -> str:
    if adapt == 'NONE':
        return 'no_adapt'
    if adapt == 'SAR':
        return 'sar'
    if adapt == 'SARCORR':
        # SAR whose reliability filter also drops confidently wrong samples
        # using ground-truth labels (label-dependent diagnostic).
        return 'sar_correct'
    if adapt == 'SPA':
        return 'spa'
    if adapt == 'FOA':
        # FOA (Niu et al. ICML 2024) is a BP-free ViT-only adapter that wraps
        # the model in PromptViT + uses CMA-ES on prompt embeddings.
        return 'foa'
    if adapt == 'PEA':
        return 'pea_resnet18' if arch == 'rn18' else 'pea_vit'
    if adapt == 'oracle':
        return 'oracle'
    if adapt == 'oracle_spa':
        return 'oracle_spa'
    raise ValueError(f"unknown adapt: {adapt}")


def build_dense_model(arch: str, dataset: str,
                      vit_timm_tag: str = None) -> nn.Module:
    if arch == 'rn18':
        if dataset == 'cifar10':
            return ResNet18_Custom(num_classes=10)
        return tv_resnet18(weights=None, num_classes=1000)
    # ViT: without `vit_timm_tag`, build the architecture only (the caller
    # loads the weights); with a tag, instantiate it with timm's pretrained weights.
    tag = vit_timm_tag if vit_timm_tag else 'vit_base_patch16_224'
    return timm.create_model(tag, pretrained=bool(vit_timm_tag),
                             num_classes=1000)


def build_narrow_model(arch: str, dataset: str, r: float) -> nn.Module:
    if arch != 'rn18':
        raise ValueError("smaller_dense_tta is RN-18 only")
    if dataset == 'cifar10':
        return ResNet_Custom_Narrow(BasicBlock_Custom, [2, 2, 2, 2],
                                    r=r, num_classes=10)
    return ResNetTV_Narrow(r=r, num_classes=1000)


def load_state_dict_from_payload(model: nn.Module, ckpt_path: str, device):
    payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = payload.get('model_state_dict',
                        payload.get('state_dict', payload))
    model.load_state_dict(state, strict=True)
    return model.to(device)


def apply_repair_bn_stats(model: nn.Module, loader, device, num_batches: int = 4):
    """REPAIR (Jordan et al. 2022, Wang et al. 2025): re-estimate the BatchNorm
    running statistics of a structurally compressed model from a few training
    batches. `momentum=None` makes the running average an arithmetic mean over
    the seen batches. Needed after on-the-fly compression because `_adjust_bn`
    only subsets the dense statistics by the retained channels.
    """
    n_seen = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True)
            model(x)
            n_seen += 1
            if n_seen >= num_batches:
                break
    model.eval()
    return model


def compress_on_the_fly(model: nn.Module, arch: str, method: str, ratio: float,
                        calib_loader, device, num_calib_batches: int = 50):
    """Apply structural compression to the dense model (scheme dense_prune_tta)."""
    if arch == 'rn18':
        if method == 'fold':
            return ResNet18_ModelFolding(model, compression_ratio=ratio).apply().to(device)
        if method == 'mag-l2':
            return ResNet18_MagnitudePruning(model, compression_ratio=ratio, p=2).apply().to(device)
        if method == 'wanda':
            pr = ResNet18_WandaPruning(model, compression_ratio=ratio)
            pr.run_calibration(calib_loader, device, num_batches=num_calib_batches)
            return pr.apply().to(device)
        return one_shot_structural_pruning_resnet18(
            model=model, compression_method=method, train_loader=calib_loader,
            compression_ratio=ratio, device=device, num_batches=num_calib_batches)
    # ViT-Base
    if method == 'fold':
        return ViT_ModelFolding(model, compression_ratio=ratio).apply().to(device)
    if method == 'mag-l2':
        return ViT_MagnitudePruning(model, compression_ratio=ratio, p=2).apply().to(device)
    if method == 'wanda':
        pr = ViT_WandaPruning(model, compression_ratio=ratio)
        pr.run_calibration(calib_loader, device, num_batches=num_calib_batches)
        return pr.apply().to(device)
    out = one_shot_structural_pruning_vit(
        model=model, compression_method=method, train_loader=calib_loader,
        compression_ratio=ratio, device=device, num_batches=num_calib_batches)
    return out[0] if isinstance(out, tuple) else out


def resolve_dense_checkpoint_path(args) -> str:
    # --dense_checkpoint overrides the seed-tagged default path.
    if getattr(args, 'dense_checkpoint', None):
        return args.dense_checkpoint
    base = os.path.join(args.checkpoints_root, 'pretrained')
    if args.arch == 'rn18':
        if args.dataset == 'cifar10':
            return os.path.join(base, f'resnet18_cifar10_seed{args.seed}.pth')
        return os.path.join(base, f'resnet18_imagenet_seed{args.seed}.pth')
    # ViT-Base: AugReg2 multi-seed checkpoints (seeds 1, 2, 3)
    return os.path.join(base, f'vit_base_patch16_224_augreg2_seed{args.seed}.pth')


def resolve_scheme_checkpoint_path(args):
    """Returns (path, action) where action in {'load', 'load_and_compress'}.

    For 'load_and_compress' (scheme A) the path is the dense checkpoint to
    load; the caller applies compression after loading.
    For 'load' (schemes B + C) the path is the already-compressed checkpoint.
    """
    base = args.checkpoints_root
    if args.scheme == 'dense_prune_tta':
        return resolve_dense_checkpoint_path(args), 'load_and_compress'
    if args.scheme == 'dense_prune_finetune_in_tta':
        if args.arch == 'rn18':
            return (os.path.join(base, 'pretrained_finetune',
                f'rn18_{args.dataset}_{args.method}_'
                f'r{args.compression_ratio}_seed{args.seed}_ft.pth'), 'load')
        return (os.path.join(base, 'pretrained_finetune',
            f'vit_base_imagenet_{args.method}_'
            f'r{args.compression_ratio}_seed{args.seed}_ft.pth'), 'load')
    if args.scheme == 'smaller_dense_tta':
        if args.arch != 'rn18':
            raise ValueError("smaller_dense_tta is RN-18 only")
        return (os.path.join(base, 'pretrained',
            f'smaller_rn18_{args.dataset}_r{args.compression_ratio}_seed{args.seed}.pth'),
            'load')
    raise ValueError(f"unknown scheme: {args.scheme}")


def build_clean_train_loader(args, device):
    """Clean training loader used for compression calibration (wanda/taylor/
    hessian), REPAIR and PEA/FOA source statistics."""
    from utils.datasets import get_cifar10, get_imagenet
    if args.dataset == 'cifar10':
        return get_cifar10(train=True, bs=args.batch_size,
                           num_workers=args.num_workers,
                           persistent_workers=False, pin_memory=True)
    return get_imagenet(args.data_root_imagenet, train=True,
                        bs=args.batch_size, num_workers=args.num_workers,
                        persistent_workers=False, pin_memory=True)


def build_corruption_loader(args, corruption: str, batch_size=None,
                            shuffle=False):
    """Returns a DataLoader yielding (x, y) over one corruption at a fixed
    severity. `batch_size`/`shuffle` default to args.batch_size and False; the
    online-accuracy pass uses bs=64, shuffle=True.

    `Cifar10_C` and `ImageNet_C` are containers of per-corruption Datasets;
    the DataLoader comes from their `get_corruption_data_loader`.
    """
    bs = args.batch_size if batch_size is None else batch_size
    enc = 'vit_base_patch16_224' if args.arch == 'vit' else 'resnet18'
    if args.dataset == 'cifar10':
        ds_handle = Cifar10_C(encoder_name=enc, args=args)
        return ds_handle.get_corruption_data_loader(
            corruption=corruption, batch_size=bs,
            shuffle=shuffle, persistent_workers=False, pin_memory=True)
    set_individual_corruption(corruption_name=corruption)
    ds_handle = ImageNet_C(encoder_name=enc, args=args)
    return ds_handle.get_corruption_data_loader(
        corruption=corruption, batch_size=bs,
        shuffle=shuffle, persistent_workers=False, pin_memory=True,
        num_workers=args.num_workers)


@torch.no_grad()
def metric_prediction_entropy(model: nn.Module, loader, device,
                              max_batches=None) -> dict:
    model.eval()
    all_h = []
    n = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)
        logits = model(x)
        per_sample = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        all_h.append(per_sample.cpu())
        n += x.size(0)
    if not all_h:
        return {'pred_entropy_mean': float('nan'), 'pred_entropy_std': float('nan'),
                'pred_entropy_n': 0}
    cat = torch.cat(all_h)
    return {'pred_entropy_mean': float(cat.mean()),
            'pred_entropy_std': float(cat.std()),
            'pred_entropy_n': int(n)}


def metric_ame(model: nn.Module, loader, device, arch: str,
               num_batches: int = 1, recipe: str = 'pooled') -> dict:
    """AME = Shannon entropy of the normalized eigenvalues of the (C x C)
    activation Gram matrix, per hooked layer. `recipe` selects the reduction:
    'pooled' pools the Grams of all batches into one centered covariance and
    takes a single eigendecomposition; 'paper' computes the entropy per batch
    on the uncentered Gram and averages across batches.
    """
    targets = post_residual_hook_targets(arch)
    if recipe == 'paper':
        ranks = compute_layerwise_ame_paper_recipe(
            copy.deepcopy(model).to(device), loader, device=device,
            num_batches=num_batches, target_layer_types=targets)
    else:
        ranks = compute_layerwise_effective_ranks(
            copy.deepcopy(model).to(device), loader, device=device,
            num_batches=num_batches, target_layer_types=targets)
    if not ranks:
        return {'ame_mean': float('nan'), 'ame_min': float('nan'),
                'ame_per_layer': {}}
    vals = list(ranks.values())
    return {'ame_mean': float(sum(vals) / len(vals)),
            'ame_min': float(min(vals)),
            'ame_per_layer': {k: float(v) for k, v in ranks.items()}}


def metric_msa_dsr(dense_model: nn.Module, compressed_model: nn.Module,
                   loader, device, n_samples: int = 16,
                   n_components: int = 8) -> dict:
    # The fused SDPA kernels have no differentiable backward, which breaks the
    # jvp inside msa_distance(); use the math backend for this call only.
    prev_flash = torch.backends.cuda.flash_sdp_enabled()
    prev_memeff = torch.backends.cuda.mem_efficient_sdp_enabled()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    try:
        mean_d, std_d = msa_distance(
            net1=copy.deepcopy(dense_model).to(device),
            net2=copy.deepcopy(compressed_model).to(device),
            dataloader=loader, n_samples=n_samples, n_components=n_components,
            device=device)
    finally:
        torch.backends.cuda.enable_flash_sdp(prev_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(prev_memeff)
    return {'msa_dsr_mean': float(mean_d), 'msa_dsr_std': float(std_d)}


def metric_cka_to_dense(dense_model: nn.Module, compressed_model: nn.Module,
                        loader, device, arch: str, num_epochs: int = 1) -> dict:
    if CKACalculator is None:
        return {'cka_full_mean': float('nan'), 'cka_diag_mean': float('nan'),
                'cka_diag_min': float('nan'), 'cka_note': 'cka package not importable'}
    # Restrict CKA hooks to the post-residual block classes only.
    targets = post_residual_hook_targets(arch)
    m1 = copy.deepcopy(dense_model).to(device)
    m2 = copy.deepcopy(compressed_model).to(device)
    calc = CKACalculator(model1=m1, model2=m2, dataloader=loader,
                         hook_fn='flatten', hook_layer_types=targets,
                         device=device, num_epochs=num_epochs, group_size=512,
                         epsilon=1e-4, is_main_process=True)
    mat = calc.calculate_cka_matrix().cpu()
    L = min(mat.shape)
    diag = mat[torch.arange(L), torch.arange(L)]
    return {'cka_full_mean': float(mat.mean()),
            'cka_diag_mean': float(diag.mean()),
            'cka_diag_min': float(diag.min())}


def adapt_one_epoch(adapted, loader, device, requires_labels: bool):
    """Run a single pass of adaptation over the corruption loader.

    `requires_labels` is True for oracle / oracle_spa (supervised CE)."""
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        if requires_labels:
            y = batch[1].to(device, non_blocking=True)
            _ = adapted(x, y)
        else:
            _ = adapted(x)


def prepare_adapted_model(model: nn.Module, arch: str, dataset: str,
                          adapt: str, args, device,
                          source_loader=None) -> nn.Module:
    """Returns the TTA-wrapped adapted model. For adapt=='NONE' returns the
    raw model unchanged (PRE phase)."""
    if adapt == 'NONE':
        return model
    name = _learner_name(adapt, arch)
    print(f"[tta] method={name} tta_lr={args.tta_lr}", flush=True)
    # sar_correct uses SAR's model configuration and optimizer; only the
    # update mask differs.
    factory_name = 'sar' if name == 'sar_correct' else name
    opt, net = get_adaptation_optimizer_configure_model(
        tta_method_name=factory_name, model=model, learning_rate=args.tta_lr,
        args=args, device=device, cifar10=(dataset == 'cifar10'))
    if net is None:
        # Model configured in place; wrap it with the registered learner.
        extra = {}
        if name in ('sar', 'sar_correct'):
            # SAR reliability threshold E0 = 0.4 ln(C) (SAR Eq. 2). The factory
            # default assumes C = 1000, which would admit every CIFAR-10 sample.
            num_classes = 10 if dataset == 'cifar10' else 1000
            extra['margin_e0'] = 0.4 * math.log(num_classes)
        adapted = make_learner(name, model=model, optimizer=opt, **extra)
    elif name in ('spa', 'oracle_spa'):
        # SPA / Oracle-SPA: the adaptation step lives in the SPA adapter
        # class, not in the BYOLWrapper returned by set_optimizer.
        adapted = make_learner(name, model=net, optimizer=opt)
    elif name == 'foa':
        # FOA.forward reads self.train_info, which obtain_origin_stat sets
        # from the source loader.
        adapted = net
        adapted.obtain_origin_stat(source_loader, device=device,
                                   max_batches=args.pea_source_batches)
    else:
        adapted = net  # PEA: wrapper already instantiated
    # PEA needs a one-shot source-stat estimation before adapting.
    if name in ('pea_resnet18', 'pea_vit'):
        adapted.precompute_source_stats(source_loader, device,
                                        max_batches=args.pea_source_batches)
    return adapted


def output_filename(args) -> str:
    if args.phase == 'PRE':
        adapt_tag = 'adaptNONE'
    else:
        adapt_tag = f'adapt{args.adapt}'
    # AME-only runs carry only {ame_mean, ame_min, ame_per_layer} and get
    # their own suffix.
    suffix = '_ameonly' if getattr(args, 'ame_only', False) else ''
    # The `paper` AME recipe writes to a sibling file.
    if getattr(args, 'ame_recipe', 'pooled') == 'paper':
        suffix += '_paper'
    # REPAIR runs get their own suffix so both variants can coexist.
    if getattr(args, 'apply_repair', False) and args.arch == 'rn18':
        suffix += '_repair'
    # Accuracy-only runs store different fields (acc_top1/5/n) and get their
    # own suffix so --skip_if_exists stays correct for both kinds of file.
    if getattr(args, 'acc_only', False):
        suffix += '_acc'
    return (f'{args.arch}_{args.dataset}_{args.scheme}_{adapt_tag}_'
            f'method{args.method}_r{args.compression_ratio}_'
            f'seed{args.seed}_sev{args.severity}_phase{args.phase}{suffix}.json')


@torch.inference_mode()
def metric_accuracy(model: nn.Module, loader, device, max_batches=None) -> dict:
    """Top-1/top-5 accuracy of `model` on a labelled loader; NaN if the loader
    yields no labels."""
    model.eval()
    c1 = c5 = n = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            return {'acc_top1': float('nan'), 'acc_top5': float('nan'), 'acc_n': 0}
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)
        logits = model(x)
        c1 += (logits.argmax(1) == y).sum().item()
        k = min(5, logits.size(1))
        _, p5 = logits.topk(k, dim=1)
        c5 += (p5 == y.unsqueeze(1)).any(1).sum().item()
        n += x.size(0)
    if n == 0:
        return {'acc_top1': float('nan'), 'acc_top5': float('nan'), 'acc_n': 0}
    return {'acc_top1': 100.0 * c1 / n, 'acc_top5': 100.0 * c5 / n, 'acc_n': int(n)}


def metric_accuracy_online(adapted, loader, device, requires_labels: bool) -> dict:
    """Online TTA accuracy: each batch is scored with the prediction made
    while adapting on it, and the running average over the stream is
    reported. `adapted` is the TTA wrapper whose forward adapts and returns
    logits (forward(x, y) for oracle). Not wrapped in no_grad: SAR/SPA take
    their gradient step inside the forward.
    """
    c1 = c5 = n = 0
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)
        out = adapted(x, y) if requires_labels else adapted(x)
        logits = out.detach()
        c1 += (logits.argmax(1) == y).sum().item()
        k = min(5, logits.size(1))
        _, p5 = logits.topk(k, dim=1)
        c5 += (p5 == y.unsqueeze(1)).any(1).sum().item()
        n += x.size(0)
    if n == 0:
        return {'acc_top1': float('nan'), 'acc_top5': float('nan'), 'acc_n': 0}
    return {'acc_top1': 100.0 * c1 / n, 'acc_top5': 100.0 * c5 / n, 'acc_n': int(n)}


def save_payload(args, per_corruption, ckpt_path, n_params, dense_n_params,
                 wall_seconds):
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    fname = output_filename(args)
    fpath = os.path.join(out_dir, fname)
    payload = {
        'context': {
            'arch': args.arch, 'dataset': args.dataset, 'scheme': args.scheme,
            'adapt': args.adapt, 'method': args.method,
            'compression_ratio': args.compression_ratio, 'seed': args.seed,
            'severity': args.severity, 'phase': args.phase,
            'checkpoint_path': ckpt_path, 'n_params': n_params,
            'dense_n_params': dense_n_params, 'wall_seconds': wall_seconds,
            'corruptions': list(CORRUPTIONS_15),
        },
        'per_corruption': per_corruption,
    }
    with open(fpath, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"[output] {fpath}")
    return fpath


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=('rn18', 'vit'), required=True)
    p.add_argument('--dataset', choices=('cifar10', 'imagenet'), required=True)
    p.add_argument('--scheme', choices=SCHEMES, required=True)
    p.add_argument('--adapt', required=True,
                   help='NONE for PRE phase; SAR/PEA/SPA/oracle/oracle_spa for POST')
    p.add_argument('--method', choices=SUPPORTED_METHODS, required=True)
    p.add_argument('--compression_ratio', type=float, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--severity', type=int, choices=(3, 5), required=True)
    p.add_argument('--phase', choices=('PRE', 'POST'), required=True)
    p.add_argument('--checkpoints_root', type=str, required=True)
    p.add_argument('--data_root_cifar', type=str, default='')
    p.add_argument('--data_root_imagenet', type=str, default='')
    p.add_argument('--corruptions_root_cifar', type=str, default='')
    p.add_argument('--corruptions_root_imagenet', type=str, default='')
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--num_calib_batches', type=int, default=50)
    p.add_argument('--tta_lr', type=float, default=2.5e-4,
                   help='TTA optimizer learning rate; 2.5e-4 follows the SAR '
                        'paper (Niu et al., ICLR 2023).')
    p.add_argument('--pea_source_batches', type=int, default=4)
    p.add_argument('--cka_num_epochs', type=int, default=1)
    p.add_argument('--ame_num_batches', type=int, default=20,
                   help='Mini-batches aggregated into the channel-wise Gram '
                        'matrix for AME.')
    p.add_argument('--ame_recipe', choices=('pooled', 'paper'), default='pooled',
                   help='AME aggregator: `pooled` (pooled covariance, one '
                        'eigendecomposition) or `paper` (per-batch entropy, '
                        'then mean across batches).')
    p.add_argument('--apply_repair', action='store_true',
                   help='Apply REPAIR (Jordan et al. 2022, Wang et al. 2025) '
                        'after on-the-fly structural compression: reset the BN '
                        'running stats and re-estimate them from training '
                        'batches in train() mode. RN-18 only; a no-op for ViT '
                        '(LayerNorm).')
    p.add_argument('--repair_num_batches', type=int, default=4,
                   help='Number of training batches used for REPAIR BN '
                        're-estimation.')
    p.add_argument('--include_dense_ame', action='store_true',
                   help='Also compute the dense-model AME on the same loader, '
                        'so that `||AME_compressed - AME_dense||_2` can be '
                        'formed downstream.')
    p.add_argument('--ame_only', action='store_true',
                   help='Compute only AME (skip PE/CKA/MSA). The output '
                        'filename gets an "_ameonly" suffix.')
    p.add_argument('--msa_n_samples', type=int, default=16)
    p.add_argument('--msa_n_components', type=int, default=8)
    p.add_argument('--max_batches_metrics', type=int, default=None,
                   help='Cap batches per corruption (debug / smoke).')
    p.add_argument('--cuda_device', type=int, default=0)
    p.add_argument('--acc_only', action='store_true',
                   help='Compute only accuracy (no-adapt at PRE, TTA-adapted '
                        'at POST); skip PE/AME/CKA/MSA.')
    p.add_argument('--skip_if_exists', action='store_true',
                   help='Skip if the output JSON already exists.')
    p.add_argument('--dense_checkpoint', type=str, default=None,
                   help='Explicit path to the dense reference checkpoint, '
                        'overriding the seed-tagged default. For the '
                        'dense_prune_tta scheme this is also the compression '
                        'source.')
    p.add_argument('--vit_timm_tag', type=str, default=None,
                   help='ViT only: build the dense/source model via '
                        'timm.create_model(tag, pretrained=True) instead of '
                        'loading a state dict (e.g. '
                        'vit_base_patch16_224.augreg2_in21k_ft_in1k).')
    p.add_argument('--encoder_name', type=str, default='resnet18')
    p.add_argument('--corruptions', type=str, default=None,
                   help='Optional CSV of corruption names; defaults to all 15.')
    args = p.parse_args()

    # Argument consistency checks.
    if args.phase == 'PRE' and args.adapt != 'NONE':
        sys.exit(f"PRE phase requires --adapt NONE, got {args.adapt}")
    if args.phase == 'POST' and args.adapt == 'NONE':
        sys.exit("POST phase requires a non-NONE --adapt")
    if args.arch == 'vit' and args.scheme == 'smaller_dense_tta':
        sys.exit("smaller_dense_tta scheme is RN-18 only")
    if args.arch == 'rn18' and args.adapt not in ADAPT_RN18:
        sys.exit(f"RN-18 adapt must be in {ADAPT_RN18}, got {args.adapt}")
    if args.arch == 'vit' and args.adapt not in ADAPT_VIT:
        sys.exit(f"ViT adapt must be in {ADAPT_VIT}, got {args.adapt}")

    out_path = os.path.join(args.output_dir, output_filename(args))
    if args.skip_if_exists and os.path.isfile(out_path):
        print(f"[skip] output already exists: {out_path}")
        return

    set_seed(args.seed)
    device = torch.device(f'cuda:{args.cuda_device}'
                          if torch.cuda.is_available() else 'cpu')
    print(f"[ctx] arch={args.arch} dataset={args.dataset} scheme={args.scheme} "
          f"adapt={args.adapt} method={args.method} r={args.compression_ratio} "
          f"seed={args.seed} sev={args.severity} phase={args.phase} device={device}",
          flush=True)

    t0 = time.time()

    # With --vit_timm_tag the dense ViT comes from timm with pretrained
    # weights and no state dict is loaded.
    vit_tag = getattr(args, 'vit_timm_tag', None)
    use_timm = bool(vit_tag) and args.arch == 'vit'
    ckpt_path, action = resolve_scheme_checkpoint_path(args)
    if not use_timm and not os.path.isfile(ckpt_path):
        sys.exit(f"[fatal] required checkpoint missing: {ckpt_path}")

    # Dense model (needed by CKA + MSA + scheme A compression source)
    dense = build_dense_model(args.arch, args.dataset, vit_timm_tag=vit_tag)
    if use_timm:
        print(f"[ckpt] ViT dense/source from timm pretrained tag "
              f"'{vit_tag}' (no state-dict load)", flush=True)
    else:
        dense_ckpt = args.dense_checkpoint or resolve_dense_checkpoint_path(args)
        if not os.path.isfile(dense_ckpt):
            sys.exit(f"[fatal] dense reference checkpoint missing: {dense_ckpt}")
        dense = load_state_dict_from_payload(dense, dense_ckpt, device)
    dense.eval()
    dense_n_params = sum(p.numel() for p in dense.parameters())

    # Clean training loader: compression calibration, REPAIR, PEA/FOA source
    # statistics.
    train_loader = build_clean_train_loader(args, device)

    # Build the compressed model according to the scheme.
    if action == 'load_and_compress':
        # Scheme A: load a fresh dense copy, apply compression
        model = build_dense_model(args.arch, args.dataset, vit_timm_tag=vit_tag)
        if not use_timm:
            load_state_dict_from_payload(model, ckpt_path, device)
        else:
            # torch_pruning tracing (Taylor/OBD) needs the model on the same
            # device as the calibration batches.
            model = model.to(device)
        model = compress_on_the_fly(
            model, args.arch, args.method, args.compression_ratio,
            train_loader, device, num_calib_batches=args.num_calib_batches)
        # REPAIR applies only to on-the-fly compressed RN-18: schemes B and C
        # load post-REPAIR BN statistics from their checkpoints, ViT has no BN.
        if getattr(args, 'apply_repair', False) and args.arch == 'rn18':
            print(f"[REPAIR] re-estimating BN stats with "
                  f"{args.repair_num_batches} batches of training data ...",
                  flush=True)
            apply_repair_bn_stats(
                model, train_loader, device,
                num_batches=args.repair_num_batches)
    else:
        if args.scheme == 'smaller_dense_tta':
            # Scheme C: matched-dense narrow, load directly
            model = build_narrow_model(args.arch, args.dataset,
                                       args.compression_ratio)
            load_state_dict_from_payload(model, ckpt_path, device)
        else:
            # Scheme B: compress a fresh dense model to obtain the shape, then
            # load the fine-tuned weights. The model must be on `device` before
            # compress_on_the_fly: torch_pruning tracing (Taylor/OBD) fails on
            # mixed devices.
            model = build_dense_model(args.arch, args.dataset).to(device)
            model = compress_on_the_fly(
                model, args.arch, args.method, args.compression_ratio,
                train_loader, device,
                num_calib_batches=args.num_calib_batches)
            load_state_dict_from_payload(model, ckpt_path, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] dense {dense_n_params:,} -> compressed {n_params:,} "
          f"({100*(1-n_params/dense_n_params):.1f}% sparsity)", flush=True)

    # Corruption-data handlers.
    if args.dataset == 'cifar10':
        set_cifar10_corruptions_data_handlers(
            severity=args.severity, root_path=args.corruptions_root_cifar,
            corruptions=list(CORRUPTIONS_15), encoder_name=args.encoder_name)
    else:
        set_imagenet_corruptions_data_handlers(
            corruptions=list(CORRUPTIONS_15), severity=args.severity,
            root_path=args.corruptions_root_imagenet)

    corruptions = (args.corruptions.split(',') if args.corruptions
                   else list(CORRUPTIONS_15))

    per_corruption = {}
    for corr_idx, corr in enumerate(corruptions):
        print(f"\n[corruption {corr_idx+1}/{len(corruptions)}] {corr}", flush=True)
        try:
            corr_loader = build_corruption_loader(args, corr)
        except Exception as e:
            traceback.print_exc()
            per_corruption[corr] = {'error': f'loader_failed: {type(e).__name__}: {e}'}
            continue

        # For POST: copy + adapt on this corruption
        if args.phase == 'POST':
            try:
                model_for_corr = copy.deepcopy(model).to(device)
                adapted = prepare_adapted_model(
                    model_for_corr, args.arch, args.dataset, args.adapt,
                    args, device, source_loader=train_loader)
                # SARCORR uses the labels only to build its update mask.
                requires_labels = args.adapt in ('oracle', 'oracle_spa',
                                                 'SARCORR')
                # acc_only: online TTA accuracy (adapt and score each batch;
                # bs=64, shuffled).
                if args.acc_only:
                    online_loader = build_corruption_loader(
                        args, corr, batch_size=64, shuffle=True)
                    m = metric_accuracy_online(
                        adapted, online_loader, device, requires_labels)
                    # SARCORR: record mask occupancy (fraction reliable,
                    # reliable-and-correct, batches skipped by the empty-mask guard).
                    if hasattr(adapted, 'mask_stats'):
                        m.update({f'mask_{k}': v
                                  for k, v in adapted.mask_stats().items()})
                        print(f"  mask_stats: {adapted.mask_stats()}",
                              flush=True)
                    per_corruption[corr] = m
                    print(f"  acc_only[online]: top1={m.get('acc_top1')} "
                          f"top5={m.get('acc_top5')}", flush=True)
                    del adapted, model_for_corr, online_loader, corr_loader
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                # Diagnostics: adapt over the whole stream, then measure
                # CKA/AME/MSA on the final adapted state.
                adapt_one_epoch(adapted, corr_loader, device, requires_labels)
                # Re-build a fresh corruption loader for metric eval (some
                # are non-resettable).
                corr_loader = build_corruption_loader(args, corr)
                # Use the underlying inner model for CKA/AME/MSA. For SPA
                # adapter wrapping a BYOLWrapper, the inner model is .model.
                if hasattr(adapted, 'model') and isinstance(adapted.model, nn.Module):
                    inner = adapted.model
                    if hasattr(inner, 'model'):  # BYOLWrapper case
                        inner = inner.model
                else:
                    inner = adapted
                eval_model = inner
            except Exception as e:
                traceback.print_exc()
                per_corruption[corr] = {'error': f'adapt_failed: {type(e).__name__}: {e}'}
                continue
        else:
            eval_model = model

        m = {}
        # Accuracy (no-adapt at PRE, TTA-adapted at POST).
        try:
            corr_loader_acc = build_corruption_loader(args, corr)
            m.update(metric_accuracy(eval_model, corr_loader_acc, device,
                                     max_batches=args.max_batches_metrics))
        except Exception as e:
            traceback.print_exc()
            m['acc_error'] = f'{type(e).__name__}: {e}'
        if args.acc_only:
            per_corruption[corr] = m
            print(f"  acc_only: top1={m.get('acc_top1')} top5={m.get('acc_top5')}", flush=True)
            del corr_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        if not args.ame_only:
            try:
                m.update(metric_prediction_entropy(
                    eval_model, corr_loader, device,
                    max_batches=args.max_batches_metrics))
            except Exception as e:
                traceback.print_exc()
                m['pred_entropy_error'] = f'{type(e).__name__}: {e}'

        # CKA, AME, MSA use a fresh loader pass each
        try:
            corr_loader_ame = build_corruption_loader(args, corr)
            m.update(metric_ame(eval_model, corr_loader_ame, device,
                                arch=args.arch,
                                num_batches=args.ame_num_batches,
                                recipe=args.ame_recipe))
        except Exception as e:
            traceback.print_exc()
            m['ame_error'] = f'{type(e).__name__}: {e}'

        # Dense-model AME on the same corruption loader, for
        # ||AME_compressed - AME_dense||_2.
        if args.include_dense_ame:
            try:
                corr_loader_dense_ame = build_corruption_loader(args, corr)
                dense_ame = metric_ame(dense, corr_loader_dense_ame, device,
                                       arch=args.arch,
                                       num_batches=args.ame_num_batches,
                                       recipe=args.ame_recipe)
                # Re-key so dense AME fields don't clash with compressed.
                for k, v in dense_ame.items():
                    m[f'dense_{k}'] = v
            except Exception as e:
                traceback.print_exc()
                m['dense_ame_error'] = f'{type(e).__name__}: {e}'

        if not args.ame_only and args.cka_num_epochs > 0:
            try:
                corr_loader_cka = build_corruption_loader(args, corr)
                m.update(metric_cka_to_dense(dense, eval_model, corr_loader_cka,
                                             device, arch=args.arch,
                                             num_epochs=args.cka_num_epochs))
            except Exception as e:
                traceback.print_exc()
                m['cka_error'] = f'{type(e).__name__}: {e}'
        elif not args.ame_only:
            m['cka_skipped'] = 'cka_num_epochs=0 (smoke / fast mode)'

        if not args.ame_only:
            try:
                corr_loader_msa = build_corruption_loader(args, corr)
                m.update(metric_msa_dsr(dense, eval_model, corr_loader_msa, device,
                                        n_samples=args.msa_n_samples,
                                        n_components=args.msa_n_components))
            except Exception as e:
                traceback.print_exc()
                m['msa_error'] = f'{type(e).__name__}: {e}'

        per_corruption[corr] = m
        def _fmt(v, p):
            try: return f"{float(v):.{p}f}"
            except (ValueError, TypeError): return "n/a"
        print(f"  metrics: pe={_fmt(m.get('pred_entropy_mean'), 3)} "
              f"ame={_fmt(m.get('ame_mean'), 2)} "
              f"cka_min={_fmt(m.get('cka_diag_min'), 3)} "
              f"msa={_fmt(m.get('msa_dsr_mean'), 3)}", flush=True)
        # Free per-corruption memory
        del corr_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wall = time.time() - t0
    save_payload(args, per_corruption, ckpt_path, n_params, dense_n_params, wall)
    print(f"\n[done] {output_filename(args)} in {wall:.0f}s")


if __name__ == '__main__':
    main()
