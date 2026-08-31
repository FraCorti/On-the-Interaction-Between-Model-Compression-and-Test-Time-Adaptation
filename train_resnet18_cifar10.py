import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms

from model_cifar10.resnet import ResNet18_Custom
from utils.seed import set_seed


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(batch_size=128, num_workers=4, data_root='./data'):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = datasets.CIFAR10(root=data_root, train=True,
                                 download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_root, train=False,
                                download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(
        description='Train ResNet-18 (CIFAR-adapted) on CIFAR-10')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--save_dir', type=str, default='pretrained')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--dry_run', action='store_true',
                        help='Run 2 epochs only for testing')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(f'cuda:{args.cuda_device}'
                          if torch.cuda.is_available() else 'cpu')

    model = ResNet18_Custom(num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResNet18_Custom (CIFAR-adapted)")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {device}")

    train_loader, test_loader = get_cifar10_loaders(
        args.batch_size, data_root=args.data_root)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    epochs = 2 if args.dry_run else args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir,
                             f'resnet18_cifar10_seed{args.seed}.pth')

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.1f}% | "
                  f"Test Acc {test_acc:.1f}% | Best {best_acc:.1f}% | "
                  f"LR {scheduler.get_last_lr()[0]:.6f} | "
                  f"Time {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\nTraining complete. Best test accuracy: {best_acc:.2f}%")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Checkpoint saved to: {save_path}")


if __name__ == '__main__':
    main()
