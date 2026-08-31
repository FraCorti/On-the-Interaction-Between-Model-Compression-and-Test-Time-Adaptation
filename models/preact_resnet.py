import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.mu, self.std = mu, std

    def forward(self, x):
        return (x - self.mu) / self.std

# Pre-activation residual block
class PreActBlock(nn.Module):
    """ Pre-activation version of the BasicBlock. """
    expansion = 1

    def __init__(self, in_planes, planes, bn, learnable_bn, stride=1, activation='relu', droprate=0.0, gn_groups=32):
        super(PreActBlock, self).__init__()
        self.collect_preact = True
        self.activation = activation
        self.droprate = droprate
        self.avg_preacts = []
        self.bn1 = nn.BatchNorm2d(in_planes, affine=learnable_bn) if bn else nn.GroupNorm(gn_groups, in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not learnable_bn)
        self.bn2 = nn.BatchNorm2d(planes, affine=learnable_bn) if bn else nn.GroupNorm(gn_groups, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not learnable_bn)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=not learnable_bn)
            )

    def act_function(self, preact):
        if self.activation == 'relu':
            act = F.relu(preact)
        else:
            assert self.activation[:8] == 'softplus'
            beta = int(self.activation.split('softplus')[1])
            act = F.softplus(preact, beta=beta)
        return act

    def forward(self, x):
        out = self.act_function(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x  # shortcut takes the pre-activated out, not x
        out = self.conv1(out)
        out = self.act_function(self.bn2(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out += shortcut
        return out


# Pre-activation ResNet
class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, n_cls, model_width=64, cuda=True, half_prec=False, activation='relu',
                 droprate=0.0, bn_flag=True, normalize_features=False, normalize_logits=False):
        super(PreActResNet, self).__init__()
        self.half_prec = half_prec
        self.bn_flag = bn_flag
        self.gn_groups = model_width // 2  # in particular, 32 for model_width=64 as in the original GroupNorm paper
        self.learnable_bn = True  # doesn't matter if self.bn=False
        self.in_planes = model_width
        self.avg_preact = None
        self.activation = activation
        self.n_cls = n_cls
        self.mu = torch.tensor((0.0, 0.0, 0.0)).view(1, 3, 1, 1)
        self.std = torch.tensor((1.0, 1.0, 1.0)).view(1, 3, 1, 1)
        self.normalize_logits = normalize_logits
        self.normalize_features = normalize_features

        if cuda:
            self.mu, self.std = self.mu.cuda(), self.std.cuda()

        self.normalize = Normalize(self.mu, self.std)
        self.conv1 = nn.Conv2d(3, model_width, kernel_size=3, stride=1, padding=1, bias=not self.learnable_bn)
        self.layer1 = self._make_layer(block, model_width, num_blocks[0], 1, droprate)
        self.layer2 = self._make_layer(block, 2 * model_width, num_blocks[1], 2, droprate)
        self.layer3 = self._make_layer(block, 4 * model_width, num_blocks[2], 2, droprate)
        final_layer_factor = 8
        self.layer4 = self._make_layer(block, final_layer_factor * model_width, num_blocks[3], 2, droprate)
        self.bn = nn.BatchNorm2d(final_layer_factor * model_width * block.expansion) if self.bn_flag \
            else nn.GroupNorm(self.gn_groups, final_layer_factor * model_width * block.expansion)
        self.linear = nn.Linear(final_layer_factor * model_width * block.expansion, 1 if n_cls == 2 else n_cls)

    def _make_layer(self, block, planes, num_blocks, stride, droprate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.bn_flag, self.learnable_bn, stride, self.activation,
                                droprate, self.gn_groups))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False, return_block=5):
        assert return_block in [1, 2, 3, 4, 5], 'wrong return_block'
        for layer in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            layer.avg_preacts = []

        out = self.normalize(x)
        out = self.conv1(out)
        out = self.layer1(out)
        if return_features and return_block == 1:
            return out
        out = self.layer2(out)
        if return_features and return_block == 2:
            return out
        out = self.layer3(out)
        if return_features and return_block == 3:
            return out
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        if return_features and return_block == 4:
            return out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if return_features and return_block == 5:
            return out

        if self.normalize_features:
            out = out / out.norm(dim=-1, keepdim=True)
        out = self.linear(out)
        if self.normalize_logits:
            out = out - out.mean(dim=-1, keepdim=True)
            out_norms = out.norm(dim=-1, keepdim=True)
            out_norms = torch.max(out_norms, 10 ** -10 * torch.ones_like(out_norms))
            out = out / out_norms
        if out.shape[1] == 1:
            out = torch.cat([torch.zeros_like(out), out], dim=1)

        return out

# Factory for PreActResNet18
def PreActResNet18(n_cls, model_width=64, cuda=True, half_prec=False, activation='relu', droprate=0.0,
                   normalize_features=False, normalize_logits=False):
    bn_flag = True
    return PreActResNet(PreActBlock, [2, 2, 2, 2], n_cls=n_cls, model_width=model_width, cuda=cuda, half_prec=half_prec,
                        activation=activation, droprate=droprate, bn_flag=bn_flag,
                        normalize_features=normalize_features, normalize_logits=normalize_logits)


# Axis-to-permutation mapping (for compression)
def get_axis_to_perm_PreActResNet18(override=True):
    conv = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in, None, None)
    }

    norm = lambda name, p: {
        f"{name}.weight": (p,),
        f"{name}.bias": (p,),
        f"{name}.running_mean": (p,),
        f"{name}.running_var": (p,)
    }

    dense = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,)
    }

    # Basic block: BN1 (input), Conv1 -> BN2 (conv1 output), Conv2 (block output)
    def basicblock(name, p_in, p_out):
        return {
            **norm(f"{name}.bn1", p_in),                      # BN1 aligns with block input
            **conv(f"{name}.conv1", p_in, f"{name}.relu1"),   # conv1 output
            **norm(f"{name}.bn2", f"{name}.relu1"),           # BN2 aligns with conv1 output
            **conv(f"{name}.conv2", f"{name}.relu1", p_out)   # conv2 output = block output
        }

    axis_to_perm = {
        # Initial conv
        **conv("conv1", None, "relu_conv"),

        # Layer 1
        **basicblock("layer1.0", "relu_conv", "relu1_out"),
        **basicblock("layer1.1", "relu1_out", "relu1_out"),

        # Layer 2
        **basicblock("layer2.0", "relu1_out", "relu2_out"),
        **conv("layer2.0.shortcut.0", "relu1_out", "relu2_out"),
        **basicblock("layer2.1", "relu2_out", "relu2_out"),

        # Layer 3
        **basicblock("layer3.0", "relu2_out", "relu3_out"),
        **conv("layer3.0.shortcut.0", "relu2_out", "relu3_out"),
        **basicblock("layer3.1", "relu3_out", "relu3_out"),

        # Layer 4
        **basicblock("layer4.0", "relu3_out", "relu4_out"),
        **conv("layer4.0.shortcut.0", "relu3_out", "relu4_out"),
        **basicblock("layer4.1", "relu4_out", "relu4_out"),

        # Final BN and linear
        **norm("bn", "relu4_out"),
        **dense("linear", "relu4_out", None)
    }

    return axis_to_perm










def get_module_by_name_PreActResNet18(model, name: str):
    module = model
    for part in name.split('.'):
        if part.isdigit():
            idx = int(part)
            children = list(module.children())
            if idx >= len(children):
                raise IndexError(f"Index {idx} out of range for {module}, available: {len(children)}")
            module = children[idx]
        else:
            if not hasattr(module, part):
                raise AttributeError(f"{name} not found (missing {part})")
            module = getattr(module, part)
    return module





