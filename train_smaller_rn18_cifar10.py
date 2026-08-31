import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms

from model_cifar10.resnet import BasicBlock_Custom
from utils.seed import set_seed


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def keep_channels(base: int, r: float) -> int:
    """Channels kept after removing a fraction r of the filters.

    torch_pruning removes ceil(base * r) channels, i.e. keeps floor(base * (1-r));
    int() truncation equals floor for the non-negative argument here.
    """
    return max(1, int(base * (1.0 - r)))


class ResNet_Custom_Narrow(nn.Module):
    """CIFAR ResNet-18 with per-layer widths shrunk by a uniform ratio r.

    Same layout as model_cifar10.resnet.ResNet_Custom (.fc head, 4x4 average
    pool), so the TTA and CKA pipelines accept it unchanged.
    """

    def __init__(self, block, num_blocks, r=0.0, num_classes=10):
        super().__init__()
        c = lambda b: keep_channels(b, r)
        c1, c2, c3, c4 = c(64), c(128), c(256), c(512)
        stem = c(64)

        self.in_planes = stem
        self.conv1 = nn.Conv2d(3, stem, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem)
        self.relu_conv = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, c1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, c2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, c3, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, c4, num_blocks[3], stride=2)
        # Custom variant uses .fc (not .linear) to match ResNet_Custom
        self.fc = nn.Linear(c4 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu_conv(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def get_cifar10_loaders(batch_size, num_workers, data_root):
    train_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    tr = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_t)
    te = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_t)
    return (
        torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=True),
        torch.utils.data.DataLoader(te, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True),
    )


def train_one(model, loader, crit, opt, dev):
    model.train()
    tot_l, corr, tot = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        tot_l += loss.item() * x.size(0)
        corr += (out.argmax(1) == y).sum().item()
        tot += x.size(0)
    return tot_l / tot, 100.0 * corr / tot


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    corr, tot = 0, 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        out = model(x)
        corr += (out.argmax(1) == y).sum().item()
        tot += x.size(0)
    return 100.0 * corr / tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--compression_ratio', type=float, required=True,
                   help='Per-layer filter-removal ratio r used by '
                        'utils.custom_structural_pruning (paper grid).')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--save_dir', type=str, default='pretrained')
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--cuda_device', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    set_seed(args.seed)
    dev = torch.device(f'cuda:{args.cuda_device}'
                       if torch.cuda.is_available() else 'cpu')

    model = ResNet_Custom_Narrow(BasicBlock_Custom, [2, 2, 2, 2],
                                 r=args.compression_ratio, num_classes=10).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Matched-dense narrow ResNet-18 (CIFAR variant)")
    print(f"  compression_ratio r = {args.compression_ratio}")
    print(f"  channels per stage: stem={keep_channels(64, args.compression_ratio)}, "
          f"L1={keep_channels(64, args.compression_ratio)}, "
          f"L2={keep_channels(128, args.compression_ratio)}, "
          f"L3={keep_channels(256, args.compression_ratio)}, "
          f"L4={keep_channels(512, args.compression_ratio)}")
    print(f"  total params: {n_params:,}")
    print(f"  seed: {args.seed}   device: {dev}", flush=True)

    tr_loader, te_loader = get_cifar10_loaders(
        args.batch_size, args.num_workers, args.data_root)
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=args.lr,
                    momentum=args.momentum, weight_decay=args.weight_decay)
    epochs = 2 if args.dry_run else args.epochs
    sched = CosineAnnealingLR(opt, T_max=epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"r{args.compression_ratio:.3f}".rstrip('0').rstrip('.')
    save_path = os.path.join(
        args.save_dir,
        f"smaller_rn18_cifar10_{tag}_seed{args.seed}.pth"
    )

    best = 0.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tl, ta = train_one(model, tr_loader, crit, opt, dev)
        va = evaluate(model, te_loader, dev)
        sched.step()
        if va > best:
            best = va
            torch.save({
                'model_state_dict': model.state_dict(),
                'compression_ratio': args.compression_ratio,
                'seed': args.seed,
                'epoch': epoch,
                'test_acc': va,
                'arch': 'ResNet_Custom_Narrow',
                'n_params': n_params,
            }, save_path)
        if epoch % 10 == 0 or epoch in (1, epochs):
            print(f"  ep {epoch:3d}/{epochs} | tr {ta:.1f}% | te {va:.1f}% | "
                  f"best {best:.1f}% | lr {sched.get_last_lr()[0]:.4f} | "
                  f"{time.time() - t0:.0f}s", flush=True)

    print(f"Done. Best test acc {best:.1f}%. Saved to {save_path}")


if __name__ == '__main__':
    main()
