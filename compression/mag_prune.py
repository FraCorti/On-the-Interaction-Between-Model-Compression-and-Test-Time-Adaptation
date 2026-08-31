import torch
import torch.nn as nn


from compression.base_resnet import BaseResNetCompression
from compression.base_preact_resnet import BasePreActResNetCompression
from compression.base_vit import BaseViTCompression



class ResNet18_MagnitudePruning(BaseResNetCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5, p=2,
                 per_block_ratios=None):
        super().__init__(model, min_channels, compression_ratio,
                         per_block_ratios=per_block_ratios)
        self.p = p

    def apply(self):
        # Initial conv + BN
        conv1 = self.model_layers['conv1']
        k1 = max(int(conv1.out_channels * self._get_keep_ratio('conv1')), self.min_channels)
        scores1 = torch.norm(conv1.weight.view(conv1.out_channels, -1), p=self.p, dim=1)
        keep = self._get_keep_indices(scores1, k1)
        self._rebuild_conv('conv1', out_keep=keep)
        self._adjust_bn('bn1', keep)
        prev_keep = keep

        # Residual blocks
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            blocks = self.model_layers[layer]
            for i, block in enumerate(blocks):
                prefix = f"{layer}.{i}"
                block_module = self.model_layers[prefix]

                # conv1 pruning
                conv1 = block.conv1
                conv1_scores = torch.norm(conv1.weight.view(conv1.out_channels, -1), p=self.p, dim=1)
                conv1_k = max(int(len(conv1_scores) * self._get_keep_ratio(f'{layer}.{i}.conv1')), self.min_channels)
                keep1 = self._get_keep_indices(conv1_scores, conv1_k)
                in_keep = prev_keep if i == 0 else keep2

                self._rebuild_conv(f"{prefix}.conv1", in_keep=in_keep, out_keep=keep1)
                self._adjust_bn(f"{prefix}.bn1", keep1)

                # conv2 pruning
                conv2 = block.conv2
                conv2_scores = torch.norm(conv2.weight.view(conv2.out_channels, -1), p=self.p, dim=1)
                conv2_k = max(int(len(conv2_scores) * self._get_keep_ratio(f'{layer}.{i}.conv2')), self.min_channels)

                if hasattr(block_module, 'downsample') and isinstance(block_module.downsample, nn.Sequential):
                    downsample_conv_name = f"{prefix}.downsample.0"
                    if downsample_conv_name in self.model_layers:
                        keep2 = self._get_keep_indices(conv2_scores, conv2_k)
                        self._rebuild_conv(downsample_conv_name, in_keep=prev_keep, out_keep=keep2)
                        self._adjust_bn(f"{prefix}.downsample.1", keep2)
                    else:
                        keep2 = in_keep
                else:
                    keep2 = in_keep

                self._rebuild_conv(f"{prefix}.conv2", in_keep=keep1, out_keep=keep2)
                self._adjust_bn(f"{prefix}.bn2", keep2)

                prev_keep = keep2

        # Final FC
        self._prune_linear("fc", prev_keep)
        return self.model


class PreActResNet18_MagnitudePruning(BasePreActResNetCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5, p=2):
        super().__init__(model, min_channels, compression_ratio)
        self.p = p

    def apply(self):
        # --- Initial conv (no BN at root in PreActResNet) ---
        conv1 = self.model_layers['conv1']
        k1 = max(int(conv1.out_channels * self.keep_ratio), self.min_channels)
        scores1 = torch.norm(conv1.weight.view(conv1.out_channels, -1), p=self.p, dim=1)
        keep = self._get_keep_indices(scores1, k1)
        self._rebuild_conv('conv1', out_keep=keep)
        prev_keep = keep

        # Residual blocks
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            blocks = self.model_layers[layer]

            for i, block in enumerate(blocks):
                prefix = f"{layer}.{i}"

                conv1_scores = torch.norm(block.conv1.weight.view(block.conv1.out_channels, -1), p=self.p, dim=1)
                conv1_k = max(int(len(conv1_scores) * self.keep_ratio), self.min_channels)
                keep1 = self._get_keep_indices(conv1_scores, conv1_k)
                in_keep = prev_keep

                self._rebuild_conv(f"{prefix}.conv1", in_keep=in_keep, out_keep=keep1)
                self._adjust_bn(f"{prefix}.bn1", in_keep)

                conv2_scores = torch.norm(block.conv2.weight.view(block.conv2.out_channels, -1), p=self.p, dim=1)
                conv2_k = max(int(len(conv2_scores) * self.keep_ratio), self.min_channels)

                if hasattr(block, 'shortcut') and isinstance(block.shortcut, nn.Sequential):
                    keep2 = self._get_keep_indices(conv2_scores, conv2_k)
                else:
                    keep2 = in_keep

                self._rebuild_conv(f"{prefix}.conv2", in_keep=keep1, out_keep=keep2)
                self._adjust_bn(f"{prefix}.bn2", keep1)

                if hasattr(block, 'shortcut') and isinstance(block.shortcut, nn.Sequential):
                    self._rebuild_conv(f"{prefix}.shortcut.0", in_keep=in_keep, out_keep=keep2)

                prev_keep = keep2

        self._adjust_bn("bn", prev_keep)
        self._prune_linear("linear", prev_keep)
        return self.model


class ViT_MagnitudePruning(BaseViTCompression):
    """
    Magnitude-based pruning for timm ViT MLPs:
    - Score hidden units by ||row(fc1.weight)||_p
    - Keep top-k rows in fc1; select the same columns in fc2
    """
    def __init__(self, model, min_channels=1, compression_ratio=0.5, p=2):
        super().__init__(model, min_channels, compression_ratio)
        self.p = p

    def compress_function(self, axes, params):
        compressed, merge_sizes = {}, {}

        name_fc1, name_fc2 = axes
        W_fc1 = params[name_fc1 + '.weight']  # [H, D]
        W_fc2 = params[name_fc2 + '.weight']  # [D, H]

        H = W_fc1.shape[0]
        k = max(int(H * self.keep_ratio), self.min_channels)
        k = min(k, H)

        # Lp norm of rows
        scores = torch.norm(W_fc1, dim=1, p=self.p)  # [H]
        topk = torch.topk(scores, k, largest=True).indices.sort()[0].to(W_fc1.device)

        new_fc1 = W_fc1[topk, :]          # [H', D]
        new_fc2 = W_fc2[:, topk]          # [D, H']

        compressed[name_fc1 + '.weight'] = new_fc1
        compressed[name_fc2 + '.weight'] = new_fc2

        # biases
        b1 = params.get(name_fc1 + '.bias', None)
        if b1 is not None:
            compressed[name_fc1 + '.bias'] = b1[topk]
        b2 = params.get(name_fc2 + '.bias', None)
        if b2 is not None:
            compressed[name_fc2 + '.bias'] = b2  # unchanged

        merge_sizes[name_fc1] = new_fc1.shape[0]
        merge_sizes[name_fc2] = new_fc2.shape[1]
        # Store original hidden-unit positions for barrier reconstruction
        merge_sizes['keep_indices'] = {name_fc1: topk.cpu()}
        return compressed, merge_sizes
