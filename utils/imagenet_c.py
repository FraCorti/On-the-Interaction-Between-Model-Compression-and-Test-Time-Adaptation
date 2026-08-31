import copy
import os
import re
import torch
import torchvision.datasets as datasets
from torchvision.transforms import v2  # Essential for GPU transforms
from torch.utils.data import Dataset


_PLACEHOLDER_RE = re.compile(r'\{[^{}]*\}')

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
            # (images, targets) tuple from ImageFolder
            x = batch[0]
            others = batch[1:]

            # Move uint8 images to the GPU before the float conversion
            x = x.to(self.device, non_blocking=True)

            # GPU transform: resize, uint8 [0, 255] to float32 [0, 1], normalize
            x = self.transform(x)

            # Move targets to the GPU
            others_gpu = [item.to(self.device, non_blocking=True) for item in others]

            yield (x, *others_gpu)


corruptions_robustbench = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
                           "motion_blur",
                           "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform",
                           "pixelate", "jpeg_compression"]

imagenet_corruption_data_handlers = {}


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


def set_imagenet_corruptions_data_handlers(corruptions,
                                           root_path='{}/Few-Shot-Adaptation-Learning/datasets/imagenet-c/{}/{}',
                                           severity=5,
                                           encoder_name='resnet50_gn'):
    """Populate the global ImageNet-C corruption-handler registry.

    `root_path` may be (a) a 3-placeholder format string
    `'{}/.../imagenet-c/{}/{}'` (cwd-dir, corruption, severity), (b) a bare
    root such as `'/tmp/imagenet_c'`, or (c) a positional-placeholder string
    `'/tmp/imagenet_c/{1}/{2}'` with `{1}` = corruption and `{2}` = severity.
    All forms resolve to `.../<corruption>/<severity>/<wnid>/<img>.JPEG`.
    A bare root is composed explicitly because `str.format` on a string
    without placeholders returns it unchanged.
    """
    global imagenet_corruption_data_handlers
    print("Loading corruptions: {}".format(corruptions))
    print("Using pre-processing for encoder: {}".format(encoder_name))

    cpu_transform = v2.Compose([
        v2.ToImage(),
        v2.CenterCrop(224),
        v2.ToDtype(torch.uint8, scale=True)
    ])

    has_placeholders = bool(_PLACEHOLDER_RE.search(root_path))
    for corruption in corruptions:
        if has_placeholders:
            dir_path = root_path.format(
                os.path.dirname(os.getcwd()), corruption, severity)
        else:
            dir_path = os.path.join(root_path, corruption, str(severity))
        print("Preparing data handler for corruption: {} at path: {}".format(corruption, dir_path))
        if imagenet_corruption_data_handlers.get(corruption) is None:
            corruption_data_handler = datasets.ImageFolder(
                root=dir_path,
                transform=cpu_transform
            )
            imagenet_corruption_data_handlers[corruption] = corruption_data_handler


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


class ImageNet_C(Dataset):
    def __init__(self, image_size=224,
                 encoder_name='resnet18', args=None):
        super(ImageNet_C, self).__init__()

        self.corruptions = corruptions_robustbench
        self.image_size = image_size
        self.encoder_name = encoder_name

        self.corruptions_data_handlers = {}
        for corruption in self.corruptions:
            self.corruptions_data_handlers[corruption] = imagenet_corruption_data_handlers[corruption]

    def _get_gpu_transform(self, device):
        # ViT normalization: timm default 0.5/0.5
        if 'vit' in self.encoder_name or self.encoder_name == 'vit_base_patch16_224':
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        # ResNet normalization: standard ImageNet statistics
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        # GPU pipeline
        return v2.Compose([
            v2.ToDtype(torch.float32, scale=True),  # Converts [0, 255] uint8 -> [0.0, 1.0] float32
            v2.Normalize(mean=mean, std=std)
        ]).to(device)

    def _get_corruption_data_loader(self, corruption, batch_size, persistent_workers=True, pin_memory=True,
                                    num_workers=6):
        # CPU loader yields uint8 tensors
        cpu_loader = torch.utils.data.DataLoader(
            dataset=self.corruptions_data_handlers[corruption],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Wrapper yields GPU tensors
        gpu_transform = self._get_gpu_transform(device='cuda')
        return GPUWrapperDataLoader(cpu_loader, gpu_transform)

    def get_corruption_data_loader(self, corruption, batch_size, shuffle=True, persistent_workers=True, pin_memory=True,
                                   num_workers=6):
        cpu_loader = torch.utils.data.DataLoader(
            dataset=self.corruptions_data_handlers[corruption],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        gpu_transform = self._get_gpu_transform(device='cuda')
        return GPUWrapperDataLoader(cpu_loader, gpu_transform)

    def evaluate_model_all_corruption_data(self, model, device, batch_size, corruption, args, compute_acc=False,
                                            pass_labels=False):
        val_loader = self._get_corruption_data_loader(corruption, batch_size)
        if not compute_acc:
            top1, top5 = self._validate(val_loader=val_loader, model=model, device=device, args=args,
                                        pass_labels=pass_labels)
        else:
            top1, top5 = self._validate_acc(val_loader=val_loader, model=model, device=device, args=args)
        return top1.item(), top5.item()

    def _validate(self, val_loader, model, device, args, steps=None, pass_labels=False):
        # Adapted from the SAR reference implementation: https://github.com/mr-eggplant/SAR
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # See: https://github.com/mr-eggplant/SAR/issues/2
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
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        return top1.avg, top5.avg

    def _validate_acc(self, val_loader, model, device, args, steps=None):
        model.eval()

        # Adapted from the SAR reference implementation: https://github.com/mr-eggplant/SAR
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # See: https://github.com/mr-eggplant/SAR/issues/2
        with torch.inference_mode():

            for i, dl in enumerate(val_loader):

                if args.debug and i >= 1:
                    break

                # dl comes from wrapper: already on GPU
                images, target = dl[0], dl[1]

                # compute output
                output = model(images)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

        return top1.avg, top5.avg
