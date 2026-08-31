import torch
import torch.nn as nn


class BasePreActResNetCompression:
    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        self.model = model
        self.min_channels = min_channels
        self.keep_ratio = 1.0 - compression_ratio
        self.device = next(model.parameters()).device
        self._rename_layers(self.model)
        self.model_layers = dict(self.model.named_modules())

    # --- Setup ---
    def _rename_layers(self, model):
        for name, module in model.named_modules():
            if not hasattr(model, '_module_dict'):
                model._module_dict = {}
            model._module_dict[name] = module

    def _replace(self, name, new_layer):
        parts = name.split('.')
        mod = self.model
        for part in parts[:-1]:
            mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
        last = parts[-1]
        if last.isdigit():
            mod[int(last)] = new_layer
        else:
            setattr(mod, last, new_layer)

    # --- Abstract method ---
    def apply(self):
        raise NotImplementedError

    # ---- Helpers: Conv, BN, FC folding ----
    def _fold_bn_params(self, original_bn, cluster_labels, n_clusters):
        device = original_bn.weight.device
        new_bn = nn.BatchNorm2d(n_clusters).to(device)

        for param_name in ['weight', 'bias']:
            original = getattr(original_bn, param_name).data
            fused = torch.zeros(n_clusters, device=device)
            for k in range(n_clusters):
                mask = cluster_labels == k
                if mask.sum() > 0:
                    fused[k] = original[mask].mean()
            setattr(new_bn, param_name, nn.Parameter(fused))

        for stat_name in ['running_mean', 'running_var']:
            original = getattr(original_bn, stat_name).data
            fused = torch.zeros(n_clusters, device=device)
            for k in range(n_clusters):
                mask = cluster_labels == k
                if mask.sum() > 0:
                    fused[k] = original[mask].mean()
            getattr(new_bn, stat_name).data.copy_(fused)

        return new_bn

    def _rebuild_module(self, name, old_module, param_dict, cluster_labels=None, n_clusters=None):
        if isinstance(old_module, nn.Conv2d):
            new_weight = param_dict.get('weight')
            new_out_channels = new_weight.shape[0]
            new_in_channels = new_weight.shape[1]
            new_conv = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=new_out_channels,
                kernel_size=old_module.kernel_size,
                stride=old_module.stride,
                padding=old_module.padding,
                dilation=old_module.dilation,
                groups=old_module.groups,
                bias='bias' in param_dict,
                padding_mode=old_module.padding_mode
            ).to(self.device)
            new_conv.weight.data = param_dict['weight'].clone()
            if 'bias' in param_dict:
                new_conv.bias.data = param_dict['bias'].clone()
            return new_conv

        elif isinstance(old_module, nn.BatchNorm2d):
            if cluster_labels is not None and n_clusters is not None:
                return self._fold_bn_params(old_module, cluster_labels, n_clusters)
            else:
                new_num_features = param_dict['weight'].shape[0]
                new_bn = nn.BatchNorm2d(new_num_features).to(self.device)
                for pname in ['weight', 'bias', 'running_mean', 'running_var']:
                    if pname in param_dict:
                        getattr(new_bn, pname).data = param_dict[pname].clone()
                return new_bn

        elif isinstance(old_module, nn.Linear):
            new_weight = param_dict.get('weight')
            new_out_features = new_weight.shape[0]
            new_in_features = new_weight.shape[1]
            new_fc = nn.Linear(new_in_features, new_out_features).to(self.device)
            new_fc.weight.data = param_dict['weight'].clone()
            if 'bias' in param_dict:
                new_fc.bias.data = param_dict['bias'].clone()
            return new_fc

        return old_module

    # ---- Helpers: Conv, BN, FC pruning ----
    def _get_keep_indices(self, scores, k):
        return torch.argsort(scores, descending=True)[:k]

    def _prune_linear(self, name, keep):
        layer = self.model_layers[name]
        assert isinstance(layer, nn.Linear), f"{name} is not Linear but {type(layer)}"

        new_weight = layer.weight[:, keep].clone()
        new_linear = nn.Linear(in_features=len(keep), out_features=layer.out_features).to(self.device)
        new_linear.weight = nn.Parameter(new_weight)

        if layer.bias is not None:
            new_linear.bias = nn.Parameter(layer.bias.detach().clone())

        self._replace(name, new_linear)
        self.model_layers[name] = new_linear

    def _rebuild_conv(self, name, in_keep=None, out_keep=None):
        layer = self.model_layers[name]
        assert isinstance(layer, nn.Conv2d), f"{name} is not Conv2d but {type(layer)}"

        weight = layer.weight.detach()
        orig_out, orig_in = weight.shape[:2]

        out_indices = out_keep if out_keep is not None else torch.arange(orig_out, device=weight.device)
        in_indices = in_keep if in_keep is not None else torch.arange(orig_in, device=weight.device)

        new_weight = weight.index_select(0, out_indices).index_select(1, in_indices).clone()

        new_conv = nn.Conv2d(
            in_channels=len(in_indices),
            out_channels=len(out_indices),
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode
        ).to(self.device)

        new_conv.weight = nn.Parameter(new_weight)
        if layer.bias is not None:
            new_conv.bias = nn.Parameter(layer.bias[out_indices].clone())

        self._replace(name, new_conv)
        self.model_layers[name] = new_conv

    def _adjust_bn(self, name, keep):
        bn = self.model_layers[name]
        assert isinstance(bn, nn.BatchNorm2d), f"{name} is not BatchNorm2d but {type(bn)}"
        new_bn = nn.BatchNorm2d(len(keep)).to(self.device)
        new_bn.weight = nn.Parameter(bn.weight[keep].detach().clone())
        new_bn.bias = nn.Parameter(bn.bias[keep].detach().clone())
        new_bn.running_mean = bn.running_mean[keep].detach().clone()
        new_bn.running_var = bn.running_var[keep].detach().clone()
        self._replace(name, new_bn)
        self.model_layers[name] = new_bn
