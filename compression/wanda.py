import torch
import torch.nn as nn

from compression.base_resnet import BaseResNetCompression
from compression.base_vit import BaseViTCompression


class ResNet18_WandaPruning(BaseResNetCompression):
    """
    ResNet-18 structured channel pruning with Wanda scores.
    - Same layer-by-layer flow and ratios as ResNet18_MagnitudePruning.
    - Only the scores change: score_out[j] = sum_i ( ||W_{j,i,...}||_1 * s_in[i] ),
      where s_in[i] is the L2 norm of input activation channel i (from calibration).
    """

    def __init__(self, model, min_channels=1, compression_ratio=0.5,
                 per_block_ratios=None):
        super().__init__(model, min_channels, compression_ratio,
                         per_block_ratios=per_block_ratios)
        self.act_scales = {}  # layer_name -> 1D tensor [Cin] of L2 norms
        self._hooks = []

    # -------- Calibration (collect per-input-channel L2 norms) --------
    def _register_activation_hooks(self):
        self._act_sum_sq = {}

        def make_hook(name):
            def hook(mod, inputs):
                x = inputs[0]
                with torch.no_grad():
                    if isinstance(mod, nn.Conv2d):
                        # x: [B, Cin, H, W] -> per-Cin sum of squares
                        ss = (x.float() ** 2).sum(dim=(0, 2, 3)).detach().cpu()
                    elif isinstance(mod, nn.Linear):
                        # x: [B, Cin]
                        ss = (x.float() ** 2).sum(dim=0).detach().cpu()
                    else:
                        return
                    self._act_sum_sq[name] = self._act_sum_sq.get(name, 0)
                    self._act_sum_sq[name] = self._act_sum_sq[name] + ss

            return hook

        for lname, layer in self.model_layers.items():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                self._hooks.append(layer.register_forward_pre_hook(make_hook(lname)))

    def _clear_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except:
                pass
        self._hooks = []

    @torch.no_grad()
    def run_calibration(self, dataloader, device, num_batches=10):
        self.model.eval().to(device)
        self._register_activation_hooks()
        seen = 0
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            _ = self.model(x.to(device))
            seen += 1
            if seen >= num_batches:
                break
        self._clear_hooks()
        # finalize L2 norms (sqrt of sum of squares)
        self.act_scales = {name: ss.sqrt() for name, ss in self._act_sum_sq.items()}

    # ---------------- Wanda scoring helpers ----------------
    @staticmethod
    def _wanda_scores_conv(weight: torch.Tensor, s_in: torch.Tensor):
        # weight: [Cout, Cin, kH, kW], s_in: [Cin]
        w_abs_sum = weight.abs().view(weight.size(0), weight.size(1), -1).sum(dim=2)  # [Cout, Cin]
        return torch.matmul(w_abs_sum, s_in.to(weight.device))  # [Cout]

    @staticmethod
    def _wanda_scores_linear(weight: torch.Tensor, s_in: torch.Tensor):
        return torch.matmul(weight.abs(), s_in.to(weight.device))  # [Cout]

    @staticmethod
    def _slice_weight_in(weight: torch.Tensor, in_keep):
        if in_keep is None:
            return weight
        idx = in_keep.to(weight.device)
        if weight.ndim == 4:  # Conv2d
            return weight.index_select(1, idx)
        else:  # Linear
            return weight.index_select(1, idx)

    def _get_s_in(self, layer_name, in_keep=None, fallback_weight=None):
        """
        Get per-input-channel L2 norms for a layer (CPU tensor).
        If absent (no calibration), fall back to a weight-based proxy.
        """
        if layer_name in self.act_scales:
            s = self.act_scales[layer_name]
            if in_keep is not None:
                s = s[in_keep.cpu()]
            return s.clamp_min(1e-8)
        # Fallback proxy from weights: average |W| over Cout & kernel dims per Cin
        w = fallback_weight
        if w is None:
            raise RuntimeError(f"No activation stats or fallback weights for '{layer_name}'. "
                               "Run calibration first.")
        if w.ndim == 4:
            proxy = w.abs().view(w.size(0), w.size(1), -1).sum(dim=2).mean(dim=0)  # [Cin]
        else:
            proxy = w.abs().mean(dim=0)  # [Cin]
        if in_keep is not None:
            proxy = proxy[in_keep.to(proxy.device)]
        return proxy.detach().cpu().clamp_min(1e-8)

    # ---------------- Apply (same flow as magnitude pruning) ----------------
    @torch.no_grad()
    def apply(self):
        # ----- Initial conv + BN (Wanda scores on conv1 OUT channels) -----
        conv1 = self.model_layers['conv1']
        k1 = max(int(conv1.out_channels * self._get_keep_ratio('conv1')), self.min_channels)
        s_in_conv1 = self._get_s_in('conv1', fallback_weight=conv1.weight)
        scores1 = self._wanda_scores_conv(conv1.weight, s_in_conv1)
        keep = self._get_keep_indices(scores1, k1)
        self._rebuild_conv('conv1', out_keep=keep)
        self._adjust_bn('bn1', keep)
        prev_keep = keep  # indices in the original conv1 basis

        # ----- Residual blocks (same mapping as magnitude pruning; only the scores differ) -----
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            blocks = self.model_layers[layer]
            for i, block in enumerate(blocks):
                prefix = f"{layer}.{i}"
                block_module = self.model_layers[prefix]

                # conv1 pruning (Wanda on OUT channels; inputs subset by in_keep)
                in_keep = prev_keep if i == 0 else keep2
                conv1 = block.conv1
                s_in1 = self._get_s_in(f"{prefix}.conv1", in_keep=in_keep, fallback_weight=conv1.weight)
                w1 = self._slice_weight_in(conv1.weight, in_keep)
                assert w1.size(1) == s_in1.numel(), f"Cin mismatch @ {prefix}.conv1"
                conv1_scores = self._wanda_scores_conv(w1, s_in1)
                conv1_k = max(int(conv1.out_channels * self._get_keep_ratio(f'{layer}.{i}.conv1')), self.min_channels)
                keep1 = self._get_keep_indices(conv1_scores, conv1_k)

                self._rebuild_conv(f"{prefix}.conv1", in_keep=in_keep, out_keep=keep1)
                self._adjust_bn(f"{prefix}.bn1", keep1)

                # conv2 pruning (Wanda on OUT channels; inputs subset by keep1)
                conv2 = block.conv2
                s_in2 = self._get_s_in(f"{prefix}.conv2", in_keep=keep1, fallback_weight=conv2.weight)
                w2 = self._slice_weight_in(conv2.weight, keep1)
                assert w2.size(1) == s_in2.numel(), f"Cin mismatch @ {prefix}.conv2"
                conv2_scores = self._wanda_scores_conv(w2, s_in2)
                conv2_k = max(int(conv2.out_channels * self._get_keep_ratio(f'{layer}.{i}.conv2')), self.min_channels)

                if hasattr(block_module, 'downsample') and isinstance(block_module.downsample, nn.Sequential):
                    downsample_conv_name = f"{prefix}.downsample.0"
                    if downsample_conv_name in self.model_layers:
                        keep2 = self._get_keep_indices(conv2_scores, conv2_k)
                        self._rebuild_conv(downsample_conv_name, in_keep=prev_keep, out_keep=keep2)
                        self._adjust_bn(f"{prefix}.downsample.1", keep2)
                    else:
                        keep2 = in_keep
                else:
                    keep2 = in_keep  # identity shortcut: keep the skip width and order

                self._rebuild_conv(f"{prefix}.conv2", in_keep=keep1, out_keep=keep2)
                self._adjust_bn(f"{prefix}.bn2", keep2)

                prev_keep = keep2

        # ----- Final FC (prune inputs only; outputs = classes unchanged) -----
        self._prune_linear("fc", prev_keep)
        return self.model


class ViT_WandaPruning(BaseViTCompression):
    """
    Wanda pruning for timm ViT MLPs:
    - Score hidden units via weights + input activation scales.
    - mode='proj_cols' (recommended): use |fc2.weight| column sums weighted by scales of fc2 inputs
      (post-activation hidden), then keep top-k.
    - mode='fc_rows': use |fc1.weight| * s_in (scales of fc1 inputs).
    """

    def __init__(self, model, min_channels=1, compression_ratio=0.5, mode="proj_cols"):
        super().__init__(model, min_channels, compression_ratio)
        assert mode in {"proj_cols", "fc_rows"}
        self.mode = mode
        self.act_scales = {}
        self._hooks = []
        self._named_modules = dict(self.model.named_modules())

    # --------- activation calibration ---------

    def _register_activation_hooks(self):
        """
        Collect sum of squares per input channel for every Linear module.
        Works with inputs shaped [B, N, C] or [*, C].
        """
        self._sum_sq = {}

        def make_hook(name):
            def hook(mod, inputs):
                x = inputs[0]
                with torch.no_grad():
                    x2 = x.float().reshape(-1, x.shape[-1])  # [T, C]
                    ss = (x2 ** 2).sum(dim=0).detach().cpu()  # [C]
                    if name in self._sum_sq:
                        self._sum_sq[name] += ss
                    else:
                        self._sum_sq[name] = ss

            return hook

        for name, mod in self._named_modules.items():
            if isinstance(mod, nn.Linear):
                self._hooks.append(mod.register_forward_pre_hook(make_hook(name)))

    def _clear_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    @torch.no_grad()
    def run_calibration(self, dataloader, device=None, num_batches=50):
        """
        Run a few (clean) batches to estimate input-channel L2 norms for each Linear.
        """
        device = device or self.device
        self.model.eval().to(device)
        self._register_activation_hooks()

        seen = 0
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            _ = self.model(x.to(device))
            seen += 1
            if seen >= num_batches:
                break

        self._clear_hooks()

        # L2 = sqrt(sum of squares); clamp for numerical stability
        self.act_scales = {name: ss.sqrt().clamp_min(1e-8) for name, ss in self._sum_sq.items()}

    # --------- wanda scoring utilities ---------

    @staticmethod
    def _wanda_row_scores_fc(W_fc: torch.Tensor, s_in: torch.Tensor):
        # |W| @ s_in
        return torch.matmul(W_fc.abs(), s_in.to(W_fc.device))

    def _get_scale(self, module_name: str, expected_dim: int):
        s = self.act_scales.get(module_name, None)
        if s is None or s.numel() != expected_dim:
            return None
        return s.clamp_min(1e-8)

    # --------- main compress for one pair ---------

    def compress_function(self, axes, params):
        compressed, merge_sizes = {}, {}
        name_fc1, name_fc2 = axes  # fc1 then fc2

        W_fc1 = params[name_fc1 + '.weight']  # [H, D]
        W_fc2 = params[name_fc2 + '.weight']  # [D, H]
        H, D = W_fc1.shape
        k = max(int(H * self.keep_ratio), self.min_channels)
        k = min(k, H)

        if self.mode == "fc_rows":
            s_in = self._get_scale(name_fc1, D)
            if s_in is None:  # fallback proxy
                s_in = W_fc1.abs().mean(dim=0).detach().cpu()
            scores = self._wanda_row_scores_fc(W_fc1, s_in)  # [H]

        elif self.mode == "proj_cols":
            # Use scales at fc2 input (the hidden features after GELU and dropout)
            s_hidden = self._get_scale(name_fc2, H)
            if s_hidden is None:
                # near-uniform fallback
                s_hidden = torch.ones(H)
            col_sum = W_fc2.abs().sum(dim=0)  # [H]
            scores = col_sum * s_hidden.to(W_fc2.device)  # [H]

        topk = torch.topk(scores, k, largest=True).indices.sort()[0].to(W_fc1.device)

        # apply selection
        new_fc1 = W_fc1[topk, :]  # [H', D]
        new_fc2 = W_fc2[:, topk]  # [D, H']

        compressed[name_fc1 + '.weight'] = new_fc1
        compressed[name_fc2 + '.weight'] = new_fc2

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
