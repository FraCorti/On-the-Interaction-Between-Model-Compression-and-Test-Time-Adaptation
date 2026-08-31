import torch.nn as nn


class BYOLWrapper(nn.Module):
    """BYOL-style wrapper used by SPA. ViT path is the original Niu et al.
    Self-Bootstrapping Adaptation (timm `forward_features` -> CLS token ->
    `fc_norm` -> predictor -> `head`). ResNet path is the architecturally
    analogous variant: penultimate features (output of avgpool, flattened)
    -> predictor -> classifier head (`.fc` for torchvision or `.linear` for
    the CIFAR ResNet_Custom). The wrapper detects which path to use from
    the presence of `forward_features` / `head` (ViT) versus `.fc` /
    `.linear` (ResNet)."""

    def __init__(self, model, projector_dim):
        super().__init__()
        self.model = model
        self.predictor = nn.Linear(projector_dim, projector_dim, bias=False)
        nn.init.eye_(self.predictor.weight)

        # Path selection: prefer the timm/ViT path if `forward_features`
        # exists and a `head` linear is exposed; otherwise drop to the
        # ResNet path keyed on `.fc` or `.linear`.
        if hasattr(model, 'forward_features') and hasattr(model, 'head'):
            self._mode = 'vit'
        elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            self._mode = 'resnet_fc'
        elif hasattr(model, 'linear') and isinstance(model.linear, nn.Linear):
            self._mode = 'resnet_linear'
        else:
            raise ValueError(
                "BYOLWrapper: model exposes neither a ViT-style "
                "(forward_features + head) nor a ResNet-style classifier "
                "(.fc / .linear). Add an explicit feature/head accessor."
            )

    def _resnet_features(self, x):
        """Penultimate features for torchvision-style ResNets and the
        in-house ResNet_Custom (model_cifar10/resnet.py). Both topologies
        match: conv1 -> bn1 -> relu -> [maxpool] -> layer1..4 -> pool ->
        flatten -> classifier. We avoid `get_rep` (defined only on the
        in-house ResNet) to keep this path valid for torchvision too."""
        m = self.model
        if hasattr(m, 'maxpool'):  # torchvision resnet18 has the 7x7 stem + maxpool
            x = m.conv1(x)
            x = m.bn1(x)
            x = m.relu(x)
            x = m.maxpool(x)
            x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
            x = m.avgpool(x)
            return x.flatten(1)
        # CIFAR ResNet_Custom: 3x3 stem, no maxpool, F.avg_pool2d(out, 4)
        import torch.nn.functional as F
        x = m.relu_conv(m.bn1(m.conv1(x)))
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        x = F.avg_pool2d(x, 4)
        return x.flatten(1)

    def forward(self, x, use_predictor=False):
        if not use_predictor:
            return self.model(x)
        if self._mode == 'vit':
            x = self.model.forward_features(x)[:, 0]
            x = self.model.fc_norm(x)
            x = self.predictor(x)
            x = self.model.head(x)
            return x
        # ResNet paths share the predictor + head structure
        feats = self._resnet_features(x)
        feats = self.predictor(feats)
        head = self.model.fc if self._mode == 'resnet_fc' else self.model.linear
        return head(feats)
