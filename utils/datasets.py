import os
import torch
import torchvision
from torchvision.transforms import v2  # Essential for GPU transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Configuration
DEFAULT_WORKERS = 7


class GPUWrapperDataLoader:
    """
    Wraps a standard DataLoader.
    Intercepts the batch on the CPU, moves it to GPU, and applies transforms.
    Yields (images, targets) already on device.
    """

    def __init__(self, loader, transform=None, device='cuda'):
        self.loader = loader
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset

    def __iter__(self):
        iterator = iter(self.loader)
        for batch in iterator:
            # Unpack batch (images, targets)
            x = batch[0]
            others = batch[1:]

            # Move to the GPU (uint8 when the loader provides it)
            x = x.to(self.device, non_blocking=True)

            # GPU transform: dtype conversion, normalize, resize
            if self.transform is not None:
                x = self.transform(x)

            # Move targets to the GPU
            others_gpu = [item.to(self.device, non_blocking=True) for item in others]

            yield (x, *others_gpu)


def get_cifar100(train=True, bs=512, num_workers=DEFAULT_WORKERS):
    path = os.path.dirname(os.path.abspath(__file__))

    # CPU: minimal load (uint8)
    cpu_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True)
    ])

    # GPU: augmentation and normalization
    stats = {'mean': [0.5071, 0.4865, 0.4409], 'std': [0.2673, 0.2564, 0.2762]}

    if train:
        gpu_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(15),
            v2.Normalize(mean=stats['mean'], std=stats['std'])
        ])
    else:
        gpu_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=stats['mean'], std=stats['std'])
        ])

    # Move transform to GPU immediately
    gpu_transform = gpu_transform.to('cuda')

    dataset = torchvision.datasets.CIFAR100(
        root=path + '/data',
        train=train,
        download=True,
        transform=cpu_transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return GPUWrapperDataLoader(loader, gpu_transform)


def get_imagenet_vit(datadir, transform=None, train=True, bs=128, num_workers=DEFAULT_WORKERS, transform_mode=None, persistent_workers=False, pin_memory=True):
    """
    Optimized loader for TIMM ViT models (vit_base_patch16_224).

    Args:
        datadir: Root directory
        transform: Ignored (Replaced by optimized GPU transform)
        train: If True, loads 'train' folder. If False, loads 'val' folder.
        transform_mode: (Optional) 'train' or 'val'.
                        If 'val', forces deterministic validation transforms (Resize 248->Crop).
                        If None, defaults to 'train' if train=True, else 'val'.
                        Use transform_mode='val' when evaluating sharpness on Training Data.
    """

    # Transform mode
    use_val_transforms = (transform_mode == 'val') or (not train and transform_mode is None)

    # CPU: geometric transforms (resize/crop) so that batched tensors have equal size
    if not use_val_transforms:
        # Training (CPU)
        cpu_transform = v2.Compose([
            v2.ToImage(),
            v2.RandomResizedCrop(
                size=(224, 224),
                scale=(0.08, 1.0),
                ratio=(0.75, 1.3333),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True
            ),
            v2.ToDtype(torch.uint8, scale=True)
        ])
    else:
        # Validation (CPU)
        cpu_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(248, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.ToDtype(torch.uint8, scale=True)
        ])

    # GPU: pixel-wise transforms (normalize, dtype, color jitter)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    if not use_val_transforms:
        # Training (GPU)
        gpu_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
            v2.Normalize(mean=mean, std=std)
        ])
    else:
        # Validation (GPU)
        gpu_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])

    gpu_transform = gpu_transform.to('cuda')

    # Data folder
    subdir = "train" if train else "val"
    dataset_path = os.path.join(datadir, subdir)
    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist."

    # Initialize dataset with CPU transform
    dataset = ImageFolder(root=dataset_path, transform=cpu_transform)

    # Optimized Loader
    data_loader = DataLoader(
        dataset,
        batch_size=bs,
        num_workers=num_workers,
        shuffle=(True if train else False),
        pin_memory=pin_memory
    )

    return GPUWrapperDataLoader(data_loader, gpu_transform)


def get_imagenet_sharpness_resnet18(datadir, train=True, bs=128, num_workers=DEFAULT_WORKERS, persistent_workers=False, pin_memory=True):
    # Sharpness calculation usually requires deterministic inputs (Validation Transform)
    # even on training data.

    # CPU: geometric transforms
    cpu_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(256, antialias=True),
        v2.CenterCrop(224),
        v2.ToDtype(torch.uint8, scale=True)
    ])

    # GPU: normalization
    gpu_transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]).to('cuda')

    subdir = "train" if train else "val"
    dataset_path = os.path.join(datadir, subdir)
    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist."

    dataset = ImageFolder(root=dataset_path, transform=cpu_transform)

    data_loader = DataLoader(
        dataset,
        batch_size=bs,
        num_workers=num_workers,
        shuffle=(True if train else False),
        pin_memory=True
    )

    return GPUWrapperDataLoader(data_loader, gpu_transform)


def get_imagenet(datadir, train=True, bs=128, num_workers=DEFAULT_WORKERS, persistent_workers=False, pin_memory=True):
    # CPU: geometric transforms (resize/crop)
    cpu_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(256, antialias=True),
        v2.CenterCrop(224),
        v2.ToDtype(torch.uint8, scale=True)
    ])

    # GPU: augmentations and normalization
    if train:
        gpu_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        subdir = "train"
        shuffle = True
    else:
        gpu_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        subdir = "val"
        shuffle = False

    gpu_transform = gpu_transform.to('cuda')

    dataset_path = os.path.join(datadir, subdir)
    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist."

    dataset = ImageFolder(root=dataset_path, transform=cpu_transform)

    data_loader = DataLoader(
        dataset,
        batch_size=bs,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory
    )

    return GPUWrapperDataLoader(data_loader, gpu_transform)


def get_cifar10_sharpness(train=True, bs=512, num_workers=DEFAULT_WORKERS, persistent_workers=False, pin_memory=True):
    path = os.path.dirname(os.path.abspath(__file__))

    # CPU
    cpu_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True)
    ])

    # GPU
    gpu_transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ]).to('cuda')

    dataset = torchvision.datasets.CIFAR10(
        root=path + '/data',
        train=train,
        download=True,
        transform=cpu_transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return GPUWrapperDataLoader(loader, gpu_transform)


def get_cifar10(train=True, bs=512, num_workers=DEFAULT_WORKERS, persistent_workers=False, pin_memory=True):
    path = os.path.dirname(os.path.abspath(__file__))

    # CPU
    cpu_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True)
    ])

    # GPU
    stats = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]}

    if train:
        gpu_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(15),
            v2.Normalize(mean=stats['mean'], std=stats['std'])
        ])
    else:
        gpu_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=stats['mean'], std=stats['std'])
        ])

    gpu_transform = gpu_transform.to('cuda')

    dataset = torchvision.datasets.CIFAR10(
        root=path + '/data',
        train=train,
        download=True,
        transform=cpu_transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return GPUWrapperDataLoader(loader, gpu_transform)