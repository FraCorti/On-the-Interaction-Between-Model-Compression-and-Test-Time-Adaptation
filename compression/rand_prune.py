import torch
import torch.nn as nn
from compression.base_resnet import BaseResNetCompression
from compression.base_preact_resnet import BasePreActResNetCompression

class ResNet18_RandomPruning(BaseResNetCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5,
                 per_block_ratios=None):
        super().__init__(model, min_channels, compression_ratio,
                         per_block_ratios=per_block_ratios)
        self.model_layers = dict(self.model.named_modules())

    def _get_random_indices(self, n, k):
        """Select k random channel indices (no sorting)."""
        perm = torch.randperm(n, device=self.device)
        return perm[:k]

    def _get_shortcut_attr(self, block):
        """Get the shortcut/downsample attribute name that exists in the block."""
        if hasattr(block, 'downsample') and isinstance(block.downsample, nn.Sequential) and len(block.downsample) > 0:
            return 'downsample'
        elif hasattr(block, 'shortcut') and isinstance(block.shortcut, nn.Sequential) and len(block.shortcut) > 0:
            return 'shortcut'
        return None

    def _get_fc_name(self):
        """Get the final FC layer name (fc or linear)."""
        if 'fc' in self.model_layers and isinstance(self.model_layers['fc'], nn.Linear):
            return 'fc'
        elif 'linear' in self.model_layers and isinstance(self.model_layers['linear'], nn.Linear):
            return 'linear'
        raise ValueError("Could not find final FC layer (expected 'fc' or 'linear')")

    def apply(self):
        # Initial conv + BN
        conv1 = self.model_layers['conv1']
        k1 = max(int(conv1.out_channels * self._get_keep_ratio('conv1')), self.min_channels)
        keep = self._get_random_indices(conv1.out_channels, k1)
        self._rebuild_conv('conv1', out_keep=keep)
        self._adjust_bn('bn1', keep)
        prev_keep = keep

        # Residual blocks
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            blocks = self.model_layers[layer]
            for i, block in enumerate(blocks):
                prefix = f"{layer}.{i}"
                block_module = self.model_layers[prefix]

                # conv1 random pruning
                conv1 = block.conv1
                k1 = max(int(conv1.out_channels * self._get_keep_ratio(f'{layer}.{i}.conv1')), self.min_channels)
                keep1 = self._get_random_indices(conv1.out_channels, k1)
                in_keep = prev_keep if i == 0 else keep2

                self._rebuild_conv(f"{prefix}.conv1", in_keep=in_keep, out_keep=keep1)
                self._adjust_bn(f"{prefix}.bn1", keep1)

                # conv2 random pruning
                conv2 = block.conv2
                k2 = max(int(conv2.out_channels * self._get_keep_ratio(f'{layer}.{i}.conv2')), self.min_channels)

                # Check for shortcut/downsample - handle both naming conventions
                shortcut_attr = self._get_shortcut_attr(block_module)
                if shortcut_attr is not None:
                    shortcut_conv_name = f"{prefix}.{shortcut_attr}.0"
                    shortcut_bn_name = f"{prefix}.{shortcut_attr}.1"
                    if shortcut_conv_name in self.model_layers:
                        keep2 = self._get_random_indices(conv2.out_channels, k2)
                        self._rebuild_conv(shortcut_conv_name, in_keep=prev_keep, out_keep=keep2)
                        self._adjust_bn(shortcut_bn_name, keep2)
                    else:
                        keep2 = in_keep
                else:
                    keep2 = in_keep

                self._rebuild_conv(f"{prefix}.conv2", in_keep=keep1, out_keep=keep2)
                self._adjust_bn(f"{prefix}.bn2", keep2)

                prev_keep = keep2

        # Final FC - handle both naming conventions
        fc_name = self._get_fc_name()
        self._prune_linear(fc_name, prev_keep)
        return self.model



class PreActResNet18_RandomPruning(BasePreActResNetCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        super().__init__(model, min_channels, compression_ratio)
        self.model_layers = dict(self.model.named_modules())

    def _get_random_indices(self, n, k):
        """Randomly select k channel indices."""
        return torch.randperm(n, device=self.device)[:k]

    def apply(self):
        # --- Initial conv (no BN at root in PreActResNet) ---
        conv1 = self.model_layers['conv1']
        k1 = max(int(conv1.out_channels * self.keep_ratio), self.min_channels)
        keep = self._get_random_indices(conv1.out_channels, k1)
        self._rebuild_conv('conv1', out_keep=keep)
        prev_keep = keep

        # Residual blocks
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            blocks = self.model_layers[layer]

            for i, block in enumerate(blocks):
                prefix = f"{layer}.{i}"

                # --- conv1 ---
                conv1_k = max(int(block.conv1.out_channels * self.keep_ratio), self.min_channels)
                keep1 = self._get_random_indices(block.conv1.out_channels, conv1_k)
                in_keep = prev_keep

                self._rebuild_conv(f"{prefix}.conv1", in_keep=in_keep, out_keep=keep1)
                self._adjust_bn(f"{prefix}.bn1", in_keep)

                # --- conv2 ---
                conv2_k = max(int(block.conv2.out_channels * self.keep_ratio), self.min_channels)
                # If shortcut exists, conv2 output must match shortcut output
                if hasattr(block, 'shortcut') and isinstance(block.shortcut, nn.Sequential):
                    keep2 = self._get_random_indices(block.conv2.out_channels, conv2_k)
                else:
                    keep2 = in_keep

                self._rebuild_conv(f"{prefix}.conv2", in_keep=keep1, out_keep=keep2)
                self._adjust_bn(f"{prefix}.bn2", keep1)

                # --- shortcut (downsample) ---
                if hasattr(block, 'shortcut') and isinstance(block.shortcut, nn.Sequential):
                    self._rebuild_conv(f"{prefix}.shortcut.0", in_keep=in_keep, out_keep=keep2)

                prev_keep = keep2

        # Final BN and FC
        self._adjust_bn("bn", prev_keep)
        self._prune_linear("linear", prev_keep)
        return self.model


from compression.base_vit import BaseViTCompression


class ViT_RandomPruning(BaseViTCompression):
    """
    Random structural pruning for Vision Transformers (timm).
    Randomly selects MLP hidden dimensions to keep in each transformer block.
    """

    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        super().__init__(model, min_channels, compression_ratio)

    def compress_function(self, axes, params):
        """
        Randomly select hidden dimensions to keep for fc1/fc2 pair.

        Args:
            axes: tuple (name_fc1, name_fc2)
            params: dict with keys like 'blocks.i.mlp.fc1.weight', etc.

        Returns:
            compressed_params: dict with pruned weights/biases
            merge_sizes: dict (empty for pruning)
        """
        name_fc1, name_fc2 = axes

        W_fc1 = params[name_fc1 + '.weight']  # [H, D]
        W_fc2 = params[name_fc2 + '.weight']  # [D, H]

        H = W_fc1.shape[0]
        k = max(int(H * self.keep_ratio), self.min_channels)

        # Randomly select k indices to keep
        perm = torch.randperm(H, device=self.device)
        keep_idx = perm[:k]

        # Prune fc1 output and fc2 input
        new_W_fc1 = W_fc1[keep_idx, :]  # [k, D]
        new_W_fc2 = W_fc2[:, keep_idx]  # [D, k]

        compressed = {
            name_fc1 + '.weight': new_W_fc1,
            name_fc2 + '.weight': new_W_fc2,
        }

        # Handle biases
        if name_fc1 + '.bias' in params:
            compressed[name_fc1 + '.bias'] = params[name_fc1 + '.bias'][keep_idx]
        if name_fc2 + '.bias' in params:
            compressed[name_fc2 + '.bias'] = params[name_fc2 + '.bias']  # unchanged

        return compressed, {}
