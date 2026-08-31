import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torchvision.models import resnet18

from utils.seed import set_seed


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_imagenet_loaders(data_root, batch_size=256, num_workers=8):
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    val_set = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch} [{i}/{len(loader)}] "
                  f"Loss {loss.item():.4f} | "
                  f"Acc {100.0 * correct / total:.1f}% | "
                  f"{elapsed:.0f}s", flush=True)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, correct_top5, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        _, pred_top5 = logits.topk(5, dim=1)
        correct += (logits.argmax(1) == y).sum().item()
        correct_top5 += (pred_top5 == y.unsqueeze(1)).any(dim=1).sum().item()
        total += x.size(0)
    return 100.0 * correct / total, 100.0 * correct_top5 / total


def main():
    parser = argparse.ArgumentParser(
        description='Train ResNet-18 on ImageNet-1K (torchvision recipe)')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_step_size', type=int, default=30,
                        help='Decay LR by 10x every N epochs')
    parser.add_argument('--save_dir', type=str, default='pretrained')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to ImageNet root (must contain train/ and val/)')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--dry_run', action='store_true',
                        help='Run 2 epochs only for testing')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(f'cuda:{args.cuda_device}'
                          if torch.cuda.is_available() else 'cpu')

    model = resnet18(weights=None, num_classes=1000).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResNet-18 (torchvision)")
    print(f"  Parameters: {n_params:,}")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {device}")
    print(f"  Data: {args.data_root}")

    train_loader, val_loader = get_imagenet_loaders(
        args.data_root, args.batch_size, args.num_workers)
    print(f"  Train samples: {len(train_loader.dataset):,}")
    print(f"  Val samples: {len(val_loader.dataset):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    epochs = 2 if args.dry_run else args.epochs
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)

    best_acc1 = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir,
                             f'resnet18_imagenet_seed{args.seed}.pth')

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch)
        acc1, acc5 = evaluate(model, val_loader, device)
        scheduler.step()

        if acc1 > best_acc1:
            best_acc1 = acc1
            torch.save(model.state_dict(), save_path)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.1f}% | "
              f"Val Top-1 {acc1:.2f}% | Val Top-5 {acc5:.2f}% | "
              f"Best {best_acc1:.2f}% | "
              f"LR {scheduler.get_last_lr()[0]:.6f} | "
              f"Time {elapsed/3600:.1f}h", flush=True)

    elapsed = time.time() - t0
    print(f"\nTraining complete. Best val top-1: {best_acc1:.2f}%")
    print(f"Total time: {elapsed/3600:.1f}h")
    print(f"Checkpoint saved to: {save_path}")


if __name__ == '__main__':
    main()
