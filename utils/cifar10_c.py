import copy
import os
import torch
import numpy as np
from torchvision.transforms import v2  # Essential for GPU transforms
from torch.utils.data import Dataset

from .utils import AverageMeter


class GPUWrapperDataLoader:
    """
    Wraps a standard DataLoader to handle CPU->GPU transfer and transformations invisibly.
    This allows the TTA loop to remain unchanged (it just sees a loader yielding GPU tensors).
    """

    def __init__(self, loader, transform, device='cuda'):
        self.loader = loader
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset

    def __iter__(self):
        # Create the iterator from the underlying CPU loader
        iterator = iter(self.loader)

        for batch in iterator:
            # (images, targets) tuple
            x = batch[0]
            others = batch[1:]

            # Move uint8 images to the GPU before the float conversion
            x = x.to(self.device, non_blocking=True)

            # GPU transform: uint8 to float32, normalize
            x = self.transform(x)

            # Move targets to the GPU
            others_gpu = [item.to(self.device, non_blocking=True) for item in others]

            yield (x, *others_gpu)


corruptions_robustbench = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression"
]

cifar10_corruption_data_handlers = {}


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Cifar10CNpyDataset(Dataset):
    """
    Dataset for CIFAR-10-C .npy files; returns uint8 tensors and slices the
    50,000 images by severity.
    """

    def __init__(self, root, corruption, severity, transform=None):
        super(Cifar10CNpyDataset, self).__init__()

        # Files are laid out as root/corruption.npy
        imgs_path = os.path.join(root, f"{corruption}.npy")
        labels_path = os.path.join(root, "labels.npy")

        # Use mmap_mode='r' to keep memory usage low (doesn't load all 50k into RAM)
        self._imgs_all = np.load(imgs_path, mmap_mode="r")
        self._labels_all = np.load(labels_path, mmap_mode="r")

        # CIFAR-10-C structure: 5 severities stacked, 10k images each.
        # Severity 1: 0-10k, Severity 5: 40k-50k
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000

        self.images = self._imgs_all[start_idx:end_idx]
        self.labels = self._labels_all[start_idx:end_idx]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # np.array() copies the read-only mmap slice; torch.from_numpy on it would
        # give a non-writable tensor and in-place transforms would be undefined.
        img = np.array(self.images[idx])
        target = int(self.labels[idx])

        # v2.ToImage converts the HWC array to a CHW tensor
        if self.transform is not None:
            img = self.transform(img)

        return img, target


def set_cifar10_corruptions_data_handlers(
        corruptions,
        root_path="{}/Few-Shot-Adaptation-Learning/datasets/CIFAR-10-C/{}/{}",
        severity=5,
        encoder_name="resnet18",
):
    global cifar10_corruption_data_handlers
    print("Loading CIFAR-10-C corruptions: {}".format(corruptions))

    # Minimal CPU transform: HWC uint8 array to CHW uint8 tensor
    cpu_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True)
    ])

    base_root = root_path.format(os.path.dirname(os.getcwd()), "", "").rstrip("/")

    for corruption in corruptions:
        if cifar10_corruption_data_handlers.get(corruption) is not None:
            continue

        handler = Cifar10CNpyDataset(
            root=base_root,
            corruption=corruption,
            severity=severity,
            transform=cpu_transform
        )
        cifar10_corruption_data_handlers[corruption] = handler


def set_individual_corruption(corruption_name):
    global corruptions_robustbench
    corruptions_robustbench = [corruption_name]


def set_corruptions_robustbench_length(length):
    global corruptions_robustbench
    corruptions_robustbench = corruptions_robustbench[:length]
    return copy.deepcopy(corruptions_robustbench)


def set_and_return_corruptions_robustbench_start_index(start_index):
    global corruptions_robustbench
    corruptions_robustbench = corruptions_robustbench[start_index:]
    return copy.deepcopy(corruptions_robustbench)


def return_robustbench_corruptions_index(corruption_index):
    return corruptions_robustbench[corruption_index]


class Cifar10_C(Dataset):
    def __init__(self, image_size=32, encoder_name="resnet18", args=None):
        super(Cifar10_C, self).__init__()

        self.corruptions = corruptions_robustbench
        self.image_size = image_size
        self.encoder_name = encoder_name

        self.corruptions_data_handlers = {
            c: cifar10_corruption_data_handlers[c] for c in self.corruptions
        }

    def _get_gpu_transform(self, device):
        # Standard CIFAR-10 Mean/Std
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2471, 0.2435, 0.2616]

        # GPU Pipeline: uint8 -> float32 -> Normalize
        return v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ]).to(device)

    def get_corruption_data_loader(self, corruption, batch_size, shuffle=True, persistent_workers=True, pin_memory=True,
                                   num_workers=8):
        # CPU loader yields uint8 tensors
        cpu_loader = torch.utils.data.DataLoader(
            dataset=self.corruptions_data_handlers[corruption],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Wrapper yields GPU tensors
        gpu_transform = self._get_gpu_transform(device='cuda')
        return GPUWrapperDataLoader(cpu_loader, gpu_transform)

    def evaluate_model_all_corruption_data(self, model, device, batch_size, corruption, args, compute_acc=False,
                                            pass_labels=False):
        val_loader = self.get_corruption_data_loader(corruption, batch_size)

        if not compute_acc:
            top1, top5 = self._validate(val_loader=val_loader, model=model, device=device, args=args,
                                        pass_labels=pass_labels)
        else:
            top1, top5 = self._validate_acc(val_loader=val_loader, model=model, device=device, args=args)
        return top1.item(), top5.item()

    def _validate(self, val_loader, model, device, args, steps=None, pass_labels=False):
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        for i, dl in enumerate(val_loader):
            if args.debug and i >= 1:
                break

            # dl comes from wrapper: already on GPU
            images, target = dl[0], dl[1]

            # pass_labels=True for Oracle (supervised TTA): model.forward(x, y)
            if pass_labels:
                output = model(images, target)
            else:
                output = model(images)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        return top1.avg, top5.avg

    def _validate_acc(self, val_loader, model, device, args, steps=None):
        model.eval()
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        with torch.inference_mode():
            for i, dl in enumerate(val_loader):
                if args.debug and i >= 1:
                    break

                images, target = dl[0], dl[1]

                output = model(images)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

        return top1.avg, top5.avg
