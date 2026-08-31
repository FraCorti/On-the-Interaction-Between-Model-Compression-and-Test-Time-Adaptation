import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from utils.seed import set_seed


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def keep_channels(base: int, r: float) -> int:
    """Channels kept after removing a fraction r of the filters.

    torch_pruning removes ceil(base * r) channels, i.e. keeps floor(base * (1-r)).
    """
    return max(1, int(base * (1.0 - r)))


class BasicBlockTV(nn.Module):
    """torchvision-style BasicBlock; same as torchvision.models.resnet.BasicBlock
    but inlined so we don't have to construct via private helpers."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResNetTV_Narrow(nn.Module):
    """Imagenet-style narrow ResNet-18 (7x7 stem + maxpool + .fc head)."""

    def __init__(self, num_blocks=(2, 2, 2, 2), r=0.0, num_classes=1000):
        super().__init__()
        c1, c2, c3, c4 = (keep_channels(b, r) for b in (64, 128, 256, 512))
        stem = keep_channels(64, r)

        self.conv1 = nn.Conv2d(3, stem, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(stem)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_planes = stem
        self.layer1 = self._make_layer(c1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(c2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(c3, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(c4, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c4, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlockTV(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def get_imagenet_loaders(data_root, batch_size, num_workers):
    tr_t = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    va_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tr = datasets.ImageFolder(os.path.join(data_root, 'train'), tr_t)
    va = datasets.ImageFolder(os.path.join(data_root, 'val'), va_t)
    return (
        torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=True,
                                    persistent_workers=True),
        torch.utils.data.DataLoader(va, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True,
                                    persistent_workers=True),
    )


def train_one(model, loader, crit, opt, dev, epoch):
    model.train()
    tot_l, corr, tot = 0.0, 0, 0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        tot_l += loss.item() * x.size(0)
        corr += (out.argmax(1) == y).sum().item()
        tot += x.size(0)
        if i % 100 == 0:
            print(f"  ep{epoch} [{i:4d}/{len(loader)}] "
                  f"loss {loss.item():.3f} | acc {100.0*corr/tot:.1f}% | "
                  f"{time.time()-t0:.0f}s", flush=True)
    return tot_l / tot, 100.0 * corr / tot


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    corr, top5, tot = 0, 0, 0
    for x, y in loader:
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        out = model(x)
        _, p5 = out.topk(5, dim=1)
        corr += (out.argmax(1) == y).sum().item()
        top5 += (p5 == y.unsqueeze(1)).any(dim=1).sum().item()
        tot += x.size(0)
    return 100.0 * corr / tot, 100.0 * top5 / tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--compression_ratio', type=float, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--epochs', type=int, default=90)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--lr_step_size', type=int, default=30)
    p.add_argument('--save_dir', type=str, default='pretrained')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--cuda_device', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    set_seed(args.seed)
    dev = torch.device(f'cuda:{args.cuda_device}'
                       if torch.cuda.is_available() else 'cpu')

    model = ResNetTV_Narrow(r=args.compression_ratio, num_classes=1000).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Matched-dense narrow ResNet-18 (ImageNet variant)")
    print(f"  compression_ratio r = {args.compression_ratio}")
    print(f"  channels per stage: stem={keep_channels(64, args.compression_ratio)}, "
          f"L1={keep_channels(64, args.compression_ratio)}, "
          f"L2={keep_channels(128, args.compression_ratio)}, "
          f"L3={keep_channels(256, args.compression_ratio)}, "
          f"L4={keep_channels(512, args.compression_ratio)}")
    print(f"  total params: {n_params:,}")
    print(f"  seed: {args.seed}   device: {dev}", flush=True)

    tr_loader, va_loader = get_imagenet_loaders(
        args.data_root, args.batch_size, args.num_workers)

    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=args.lr,
                    momentum=args.momentum, weight_decay=args.weight_decay)
    epochs = 2 if args.dry_run else args.epochs
    sched = StepLR(opt, step_size=args.lr_step_size, gamma=0.1)

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"r{args.compression_ratio:.3f}".rstrip('0').rstrip('.')
    save_path = os.path.join(
        args.save_dir,
        f"smaller_rn18_imagenet_{tag}_seed{args.seed}.pth"
    )

    best = 0.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tl, ta = train_one(model, tr_loader, crit, opt, dev, epoch)
        v1, v5 = evaluate(model, va_loader, dev)
        sched.step()
        if v1 > best:
            best = v1
            torch.save({
                'model_state_dict': model.state_dict(),
                'compression_ratio': args.compression_ratio,
                'seed': args.seed,
                'epoch': epoch,
                'val_acc_top1': v1,
                'val_acc_top5': v5,
                'arch': 'ResNetTV_Narrow',
                'n_params': n_params,
            }, save_path)
        print(f"  ep {epoch:2d}/{epochs} | tr {ta:.1f}% | "
              f"v1 {v1:.1f}% v5 {v5:.1f}% | best {best:.1f}% | "
              f"lr {sched.get_last_lr()[0]:.4f} | {time.time()-t0:.0f}s",
              flush=True)

    print(f"Done. Best val top-1 {best:.1f}%. Saved to {save_path}")


if __name__ == '__main__':
    main()
