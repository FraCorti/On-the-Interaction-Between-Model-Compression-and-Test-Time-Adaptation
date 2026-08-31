import argparse
import os
import time

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from timm.data import create_transform, Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchvision import datasets, transforms

from utils.seed import set_seed


NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


def get_pretrain_loaders(data_root, batch_size=512, num_workers=8):
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    # timm's 'rand-m9-mstd0.5-inc1' is the PyTorch equivalent of AugReg's aug_strong2;
    # timm clips the magnitude at 10, so 'rand-m20-n2' is not equivalent.
    train_transform = create_transform(
        input_size=224,
        is_training=True,
        auto_augment='rand-m9-mstd0.5-inc1',
        re_prob=0.0,
        interpolation='bicubic',
        mean=NORM_MEAN,
        std=NORM_STD,
    )
    val_transform = transforms.Compose([
        transforms.Resize(248,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    val_set = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def get_finetune_loaders(data_root, batch_size=512, num_workers=8):
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(248,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    val_set = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def disable_regularization(model):
    model.pos_drop.p = 0.0
    if hasattr(model, 'head_drop'):
        model.head_drop.p = 0.0
    for block in model.blocks:
        block.drop_path1.drop_prob = 0.0
        block.drop_path2.drop_prob = 0.0


def build_param_groups(model, weight_decay):
    """Split parameters into decayed / no-decay groups.

    Weight decay applies to 2D+ tensors and is excluded from 1D tensors
    (LayerNorm, biases) and from cls_token / pos_embed, as in the ViT reference.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith('.bias') \
                or name in ('cls_token', 'pos_embed'):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def pretrain_one_epoch(model, loader, criterion, optimizer, device, epoch,
                       mixup_fn, grad_accum, max_norm=1.0,
                       scheduler=None, num_updates=0):
    model.train()
    total_loss, total = 0.0, 0
    optimizer.zero_grad()
    t0 = time.time()

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if mixup_fn is not None:
            x, y = mixup_fn(x, y)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits, y) / grad_accum

        loss.backward()

        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()
            num_updates += 1
            if scheduler is not None:
                scheduler.step_update(num_updates)

        total_loss += loss.item() * grad_accum * x.size(0)
        total += x.size(0)

        if i % 500 == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch} [{i}/{len(loader)}] "
                  f"Loss {loss.item()*grad_accum:.4f} | LR {lr:.6f} | "
                  f"{elapsed:.0f}s", flush=True)

    # Partial accumulations at the epoch boundary are dropped so that num_updates
    # stays aligned with the scheduler's t_initial.
    return total_loss / total, num_updates


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, correct5, total = 0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
        _, top5 = logits.topk(5, dim=1)
        correct += (logits.argmax(1) == y).sum().item()
        correct5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
        total += x.size(0)
    return 100.0 * correct / total, 100.0 * correct5 / total


def main():
    parser = argparse.ArgumentParser(
        description='Train ViT-Base/16 on ImageNet-1K (AugReg recipe)')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--pretrain_epochs', type=int, default=300)
    parser.add_argument('--pretrain_batch', type=int, default=512,
                        help='Per-GPU batch size for pre-training')
    parser.add_argument('--effective_batch', type=int, default=4096,
                        help='Effective batch via gradient accumulation')
    parser.add_argument('--pretrain_lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--mixup_alpha', type=float, default=0.8)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--finetune_steps', type=int, default=20000)
    parser.add_argument('--finetune_batch', type=int, default=512)
    parser.add_argument('--finetune_lr', type=float, default=0.01)
    parser.add_argument('--finetune_warmup', type=int, default=500)
    parser.add_argument('--save_dir', type=str, default='pretrained')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to ImageNet root (train/ and val/)')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--dry_run', action='store_true',
                        help='Run 2 epochs + 100 ft steps for testing')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f'cuda:{args.cuda_device}'
                          if torch.cuda.is_available() else 'cpu')

    grad_accum = args.effective_batch // args.pretrain_batch
    assert args.effective_batch % args.pretrain_batch == 0, \
        f"effective_batch ({args.effective_batch}) must be divisible " \
        f"by pretrain_batch ({args.pretrain_batch})"

    # Phase 1: pre-training (AugReg recipe)
    print("=" * 60)
    print("Phase 1: Pre-training with AugReg recipe")
    print("=" * 60)

    model = timm.create_model(
        'vit_base_patch16_224', pretrained=False, num_classes=1000,
        drop_rate=args.dropout, drop_path_rate=args.drop_path,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ViT-Base/16 (AugReg IN1K recipe)")
    print(f"  Parameters: {n_params:,}")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {device}")
    print(f"  Effective batch: {args.effective_batch} "
          f"({args.pretrain_batch} x {grad_accum} accum)")
    print(f"  Normalization: mean={NORM_MEAN}, std={NORM_STD}")

    train_loader, val_loader = get_pretrain_loaders(
        args.data_root, args.pretrain_batch, args.num_workers)
    print(f"  Train samples: {len(train_loader.dataset):,}")
    print(f"  Val samples: {len(val_loader.dataset):,}")

    mixup_fn = Mixup(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=0.0,
        label_smoothing=0.0,
        num_classes=1000,
    )
    criterion_train = SoftTargetCrossEntropy()

    optimizer = optim.AdamW(
        build_param_groups(model, args.weight_decay),
        lr=args.pretrain_lr, betas=(0.9, 0.999),
    )

    pretrain_epochs = 2 if args.dry_run else args.pretrain_epochs
    steps_per_epoch = len(train_loader) // grad_accum
    total_steps = pretrain_epochs * steps_per_epoch
    warmup_steps = min(args.warmup_steps, max(total_steps - 1, 1))

    # With warmup_prefix=True the schedule lasts warmup_t + t_initial steps, so the
    # cosine reaches lr_min at the last step; t_in_epochs=False is required for
    # step_update() to advance the LR.
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=max(total_steps - warmup_steps, 1),
        lr_min=1e-6,
        warmup_t=warmup_steps,
        warmup_lr_init=1e-6,
        warmup_prefix=True,
        t_in_epochs=False,
    )

    # Print the LR at the start, warmup end, middle and end of the schedule.
    _lr_at = {}
    for _s in (0, warmup_steps, total_steps // 2, total_steps - 1):
        scheduler.step_update(_s)
        _lr_at[_s] = optimizer.param_groups[0]['lr']
    print(f"  LR schedule: step 0={_lr_at[0]:.2e}, "
          f"warmup-end ({warmup_steps})={_lr_at[warmup_steps]:.2e}, "
          f"mid ({total_steps//2})={_lr_at[total_steps//2]:.2e}, "
          f"end ({total_steps-1})={_lr_at[total_steps-1]:.2e}")
    scheduler.step_update(0)  # reset

    os.makedirs(args.save_dir, exist_ok=True)
    pretrain_path = os.path.join(
        args.save_dir, f'vit_base_patch16_224_seed{args.seed}_pretrain.pth')
    save_path = os.path.join(
        args.save_dir, f'vit_base_patch16_224_seed{args.seed}.pth')

    best_acc1 = 0.0
    num_updates = 0
    t0 = time.time()

    for epoch in range(1, pretrain_epochs + 1):
        train_loss, num_updates = pretrain_one_epoch(
            model, train_loader, criterion_train, optimizer, device, epoch,
            mixup_fn, grad_accum, args.max_grad_norm, scheduler, num_updates)

        acc1, acc5 = evaluate(model, val_loader, device)

        if acc1 > best_acc1:
            best_acc1 = acc1
            torch.save(model.state_dict(), pretrain_path)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{pretrain_epochs} | "
              f"Loss {train_loss:.4f} | "
              f"Val Top-1 {acc1:.2f}% | Top-5 {acc5:.2f}% | "
              f"Best {best_acc1:.2f}% | LR {lr:.6f} | "
              f"{elapsed/3600:.1f}h", flush=True)

    print(f"\nPhase 1 complete. Best val top-1: {best_acc1:.2f}%")
    print(f"Pre-trained checkpoint: {pretrain_path}")

    # Phase 2: fine-tuning (SGD, no augmentation/regularization)
    print("\n" + "=" * 60)
    print("Phase 2: Fine-tuning (SGD, no augmentation)")
    print("=" * 60)

    model.load_state_dict(
        torch.load(pretrain_path, map_location=device, weights_only=True))
    disable_regularization(model)

    ft_loader, ft_val_loader = get_finetune_loaders(
        args.data_root, args.finetune_batch, args.num_workers)

    ft_optimizer = optim.SGD(
        model.parameters(), lr=args.finetune_lr, momentum=0.9)

    finetune_steps = 100 if args.dry_run else args.finetune_steps
    ft_warmup = min(args.finetune_warmup, max(finetune_steps - 1, 1))

    ft_scheduler = CosineLRScheduler(
        ft_optimizer,
        t_initial=max(finetune_steps - ft_warmup, 1),
        lr_min=1e-6,
        warmup_t=ft_warmup,
        warmup_lr_init=1e-6,
        warmup_prefix=True,
        t_in_epochs=False,
    )

    criterion_ft = nn.CrossEntropyLoss()
    loader_iter = iter(ft_loader)
    best_ft_acc1 = 0.0
    t1 = time.time()

    model.train()
    for step in range(1, finetune_steps + 1):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(ft_loader)
            x, y = next(loader_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        ft_optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion_ft(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm)
        ft_optimizer.step()
        ft_scheduler.step_update(step)

        if step % 1000 == 0 or step == finetune_steps:
            acc1, acc5 = evaluate(model, ft_val_loader, device)
            if acc1 > best_ft_acc1:
                best_ft_acc1 = acc1
                torch.save(model.state_dict(), save_path)
            elapsed = time.time() - t1
            lr = ft_optimizer.param_groups[0]['lr']
            print(f"Step {step:5d}/{finetune_steps} | "
                  f"Loss {loss.item():.4f} | "
                  f"Val Top-1 {acc1:.2f}% | Top-5 {acc5:.2f}% | "
                  f"Best {best_ft_acc1:.2f}% | LR {lr:.6f} | "
                  f"{elapsed/3600:.1f}h", flush=True)

    total_time = time.time() - t0
    print(f"\nTraining complete.")
    print(f"  Phase 1 best: {best_acc1:.2f}%")
    print(f"  Phase 2 best: {best_ft_acc1:.2f}%")
    print(f"  Total time: {total_time/3600:.1f}h")
    print(f"  Final checkpoint: {save_path}")


if __name__ == '__main__':
    main()
