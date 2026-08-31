"""
Tool to compute Centered Kernel Alignment (CKA) in PyTorch w/ GPU (single or multi).

Repo: https://github.com/numpee/CKA.pytorch
Author: Dongwan Kim (Github: Numpee)
Year: 2022
"""

from __future__ import annotations
import torch
from typing import Tuple, Optional, Callable, Type, Union, TYPE_CHECKING, List

import torch.nn as nn
from tqdm.autonotebook import tqdm

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
from timm.models.vision_transformer import Block, Attention, Mlp
from torchvision.models.resnet import Bottleneck, BasicBlock

"""
Helper for CKA in PyTorch.
Adds hooks to modules of a given model.

Repo: https://github.com/numpee/CKA.pytorch
Author: Dongwan Kim (Github: Numpee)
Year: 2022
"""

_HOOK_LAYER_TYPES = (
    Bottleneck, BasicBlock, nn.Conv2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.modules.batchnorm._BatchNorm)

VIT_HOOK_LAYER_TYPES = (
    # container
    Block,
    # attention / MLP blocks
    Attention,
    Mlp,
    # basic building blocks
    nn.Linear,
    nn.Conv2d,
    nn.LayerNorm,
    # activations
    nn.GELU,
    nn.ReLU,
)


class AccumTensor(nn.Module):
    def __init__(self, default_value: torch.Tensor):
        super().__init__()
        # Keep it as a buffer so .cuda() / .to(device) still works
        self.register_buffer("val", default_value)

    def update(self, input_tensor: torch.Tensor):
        self.val += input_tensor

    def compute(self):
        return self.val


class HookManager:
    def __init__(self, model: nn.Module, hook_fn: Optional[Union[str, Callable]] = None,
                 hook_layer_types: Tuple[Type[nn.Module], ...] = _HOOK_LAYER_TYPES,
                 calculate_gram: bool = True) -> None:
        """
        Add hooks to models.
        Mainly supports ResNets.
        :param model: model to attach hooks to
        :param hook_fn: the hook function or string. Options: ("avgpool", "flatten"). Default: flatten
        :param hook_layer_types: layer types to register hooks. Should be nn.Module
        """
        self.model = model
        self.hook_fn = hook_fn
        self.hook_layer_types = hook_layer_types
        self.calculate_gram = calculate_gram
        for layer in self.hook_layer_types:
            if not issubclass(layer, nn.Module):
                raise TypeError(f"Class {layer} is not an nn.Module.")

        if self.hook_fn is None:
            self.hook_fn = self.flatten_hook_fn
        elif type(self.hook_fn) == str:
            hook_fn_dict = {'flatten': self.flatten_hook_fn, 'avgpool': self.avgpool_hook_fn}
            if self.hook_fn in hook_fn_dict:
                self.hook_fn = hook_fn_dict[self.hook_fn]
            else:
                raise ValueError(f"No hook function named {self.hook_fn}. Options: {list(hook_fn_dict.keys())}")

        # Not using dictionary because a single module may be used multiple times in forward
        self.features = []
        self.module_names = []
        self.handles = []

        self.register_hooks(self.hook_fn)

    def get_features(self) -> List[torch.Tensor]:
        return self.features

    def get_module_names(self) -> List[str]:
        return self.module_names

    def clear_features(self) -> None:
        self.features = []
        self.module_names = []

    def clear_all(self) -> None:
        self.clear_hooks()
        self.clear_features()

    def clear_hooks(self) -> None:
        num_handles = len(self.handles)
        for handle in self.handles:
            handle.remove()
        self.handles = []
        for m in self.model.modules():
            if hasattr(m, 'module_name'):
                delattr(m, 'module_name')

    def register_hooks(self, hook_fn: Callable) -> None:
        prev_num_handles = len(self.handles)
        self._register_hook_recursive(self.model, hook_fn, prev_name="")
        new_num_handles = len(self.handles)

    def _register_hook_recursive(self, module: nn.Module, hook_fn: Callable, prev_name: str = "") -> None:
        for name, child in module.named_children():
            curr_name = f"{prev_name}.{name}" if prev_name else name
            curr_name = curr_name.replace("_model.", "")
            num_grandchildren = len(list(child.children()))

            # skip Identity layers
            if isinstance(child, nn.Identity):
                continue

            if num_grandchildren > 0:
                self._register_hook_recursive(child, hook_fn, prev_name=curr_name)
            if isinstance(child, self.hook_layer_types):
                handle = child.register_forward_hook(hook_fn)
                self.handles.append(handle)
                setattr(child, 'module_name', curr_name)

    def flatten_hook_fn(self, module: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> None:
        batch_size = out.size(0)
        feature = out.reshape(batch_size, -1)
        if self.calculate_gram:
            feature = gram(feature)
        module_name = getattr(module, 'module_name')
        self.features.append(feature)
        self.module_names.append(module_name)

    def avgpool_hook_fn(self, module: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> None:
        if out.dim() == 4:
            feature = out.mean(dim=(-1, -2))
        elif out.dim() == 3:
            feature = out.mean(dim=-1)
        else:
            feature = out
        if self.calculate_gram:
            feature = gram(feature)
        module_name = getattr(module, 'module_name')
        self.features.append(feature)
        self.module_names.append(module_name)


def gram(x: torch.Tensor) -> torch.Tensor:
    return x.matmul(x.t())


class CKACalculator:
    def __init__(self, model1: nn.Module, model2: nn.Module, dataloader: DataLoader,
                 hook_fn: Optional[Union[str, Callable]] = None,
                 device: Optional[torch.device] = None,
                 debug: bool = False,
                 hook_layer_types: Tuple[Type[nn.Module], ...] = _HOOK_LAYER_TYPES, num_epochs: int = 10,
                 group_size: int = 512, epsilon: float = 1e-4, is_main_process: bool = True) -> None:
        """
        Class to extract intermediate features and calculate CKA Matrix.
        :param model1: model to evaluate. __call__ function should be implemented if NOT instance of `nn.Module`.
        :param model2: second model to evaluate. __call__ function should be implemented if NOT instance of `nn.Module`.
        :param dataloader: Torch DataLoader for dataloading. Assumes first return value contains input images.
        :param hook_fn: Optional - Hook function or hook name string for the HookManager. Options: [flatten, avgpool]. Default: flatten
        :param hook_layer_types: Types of layers (modules) to add hooks to.
        :param num_epochs: Number of epochs for cka_batch. Default: 10
        :param group_size: group_size for GPU acceleration. Default: 512
        :param epsilon: Small multiplicative value for HSIC. Default: 1e-4
        :param is_main_process: is current instance main process. Default: True
        """
        self.model1 = model1
        self.model2 = model2
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.group_size = group_size
        self.epsilon = epsilon
        self.is_main_process = is_main_process

        self.model1.eval()
        self.model2.eval()
        self.hook_manager1 = HookManager(self.model1, hook_fn, hook_layer_types, calculate_gram=True)
        self.hook_manager2 = HookManager(self.model2, hook_fn, hook_layer_types, calculate_gram=True)
        self.module_names_X = None
        self.module_names_Y = None
        self.num_layers_X = None
        self.num_layers_Y = None
        self.num_elements = None

        # Metrics to track
        self.cka_matrix = None
        self.hsic_matrix = None
        self.self_hsic_x = None
        self.self_hsic_y = None

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug

    @torch.no_grad()
    def calculate_cka_matrix(self) -> torch.Tensor:
        curr_hsic_matrix = None
        curr_self_hsic_x = None
        curr_self_hsic_y = None
        for epoch in range(self.num_epochs):
            loader = tqdm(self.dataloader, desc=f"Epoch {epoch}", disable=True)

            if self.debug and epoch >= 1:
                break

            for it, (imgs, *_) in enumerate(loader):

                if self.debug and it >= 1:
                    break

                imgs = imgs.to(self.device)
                self.model1(imgs)
                self.model2(imgs)
                all_layer_X, all_layer_Y = self.extract_layer_list_from_hook_manager()

                # Initialize values on first loop
                if self.num_layers_X is None:
                    curr_hsic_matrix, curr_self_hsic_x, curr_self_hsic_y = self._init_values(all_layer_X, all_layer_Y)

                # Get self HSIC values --> HSIC(K, K), HSIC(L, L)
                self._calculate_self_hsic(all_layer_X, all_layer_Y, curr_self_hsic_x, curr_self_hsic_y)

                # Get cross HSIC values --> HSIC(K, L)
                self._calculate_cross_hsic(all_layer_X, all_layer_Y, curr_hsic_matrix)

                self.hook_manager1.clear_features()
                self.hook_manager2.clear_features()
                curr_hsic_matrix.fill_(0)
                curr_self_hsic_x.fill_(0)
                curr_self_hsic_y.fill_(0)

        # Update values across GPUs
        hsic_matrix = self.hsic_matrix.compute()
        hsic_x = self.self_hsic_x.compute()
        hsic_y = self.self_hsic_y.compute()
        self.cka_matrix = hsic_matrix.reshape(self.num_layers_Y, self.num_layers_X) / torch.sqrt(hsic_x * hsic_y)
        return self.cka_matrix

    def extract_layer_list_from_hook_manager(self) -> Tuple[List, List]:
        all_layer_X, all_layer_Y = self.hook_manager1.get_features(), self.hook_manager2.get_features()
        return all_layer_X, all_layer_Y

    def hsic1(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        '''
        Batched version of HSIC.
        :param K: Size = (B, N, N) where N is the number of examples and B is the group/batch size
        :param L: Size = (B, N, N) where N is the number of examples and B is the group/batch size
        :return: HSIC tensor, Size = (B)
        '''
        assert K.size() == L.size()
        assert K.dim() == 3
        K = K.clone()
        L = L.clone()
        n = K.size(1)

        # K, L --> K~, L~ by setting diagonals to zero
        K.diagonal(dim1=-1, dim2=-2).fill_(0)
        L.diagonal(dim1=-1, dim2=-2).fill_(0)

        KL = torch.bmm(K, L)
        trace_KL = KL.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)
        middle_term = K.sum((-1, -2), keepdim=True) * L.sum((-1, -2), keepdim=True)
        middle_term /= (n - 1) * (n - 2)
        right_term = KL.sum((-1, -2), keepdim=True)
        right_term *= 2 / (n - 2)
        main_term = trace_KL + middle_term - right_term
        hsic = main_term / (n ** 2 - 3 * n)
        return hsic.squeeze(-1).squeeze(-1)

    def reset(self) -> None:
        # Set values to none, clear feature and hooks
        self.cka_matrix = None
        self.hsic_matrix = None
        self.self_hsic_x = None
        self.self_hsic_y = None
        self.hook_manager1.clear_all()
        self.hook_manager2.clear_all()

    def _init_values(self, all_layer_X, all_layer_Y):
        self.num_layers_X = len(all_layer_X)
        self.num_layers_Y = len(all_layer_Y)
        self.module_names_X = self.hook_manager1.get_module_names()
        self.module_names_Y = self.hook_manager2.get_module_names()
        self.num_elements = self.num_layers_Y * self.num_layers_X
        curr_hsic_matrix = torch.zeros(self.num_elements).to(device=self.device)
        curr_self_hsic_x = torch.zeros(1, self.num_layers_X).to(device=self.device)
        curr_self_hsic_y = torch.zeros(self.num_layers_Y, 1).to(device=self.device)
        self.hsic_matrix = AccumTensor(torch.zeros_like(curr_hsic_matrix)).to(device=self.device)
        self.self_hsic_x = AccumTensor(torch.zeros_like(curr_self_hsic_x)).to(device=self.device)
        self.self_hsic_y = AccumTensor(torch.zeros_like(curr_self_hsic_y)).to(device=self.device)
        return curr_hsic_matrix, curr_self_hsic_x, curr_self_hsic_y

    def _calculate_self_hsic(self, all_layer_X, all_layer_Y, curr_self_hsic_x, curr_self_hsic_y):
        for start_idx in range(0, self.num_layers_X, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_layers_X)
            K = torch.stack([all_layer_X[i] for i in range(start_idx, end_idx)], dim=0)
            curr_self_hsic_x[0, start_idx:end_idx] += self.hsic1(K, K) * self.epsilon
        for start_idx in range(0, self.num_layers_Y, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_layers_Y)
            L = torch.stack([all_layer_Y[i] for i in range(start_idx, end_idx)], dim=0)
            curr_self_hsic_y[start_idx:end_idx, 0] += self.hsic1(L, L) * self.epsilon

        self.self_hsic_x.update(curr_self_hsic_x)
        self.self_hsic_y.update(curr_self_hsic_y)

    def _calculate_cross_hsic(self, all_layer_X, all_layer_Y, curr_hsic_matrix):
        for start_idx in range(0, self.num_elements, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_elements)
            K = torch.stack([all_layer_X[i % self.num_layers_X] for i in range(start_idx, end_idx)], dim=0)
            L = torch.stack([all_layer_Y[j // self.num_layers_X] for j in range(start_idx, end_idx)], dim=0)
            curr_hsic_matrix[start_idx:end_idx] += self.hsic1(K, L) * self.epsilon
        self.hsic_matrix.update(curr_hsic_matrix)
