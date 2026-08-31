import timm
import torch
import torch.nn as nn

try:
    import torch_pruning as tp
except ImportError:
    tp = None
import torch.nn.functional as F


def build_importance(compression_method: str):
    k = compression_method.lower()
    if k == "mag-l1":  return tp.importance.GroupMagnitudeImportance(p=1)  # noqa: E701
    if k == "mag-l2":  return tp.importance.GroupMagnitudeImportance(p=2)  # noqa: E701
    if k == "taylor":  return tp.importance.GroupTaylorImportance()  # noqa: E701
    if k == "hessian": return tp.importance.GroupHessianImportance()  # noqa: E701
    raise ValueError(f"Unknown importance: {compression_method}")


def wanda_like_ignored_layers_resnet18(model):
    """
    Return layers that torch_pruning.BasePruner should not prune.

    Only the final classifier (fc) is ignored. All convolutional layers
    participate; DepGraph enforces the residual-connection constraint, which
    matches the pruning scope of Fold/Wanda/Mag-L2.
    """
    ignored = []
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        ignored.append(model.fc)
    return ignored


def one_shot_structural_pruning_resnet18(model, compression_method, train_loader, compression_ratio, device,
                                         num_batches=50, per_block_ratios=None):
    """One-shot structural pruning of a ResNet-18 using torch_pruning.

    Args:
        model: ResNet-18 model to prune (modified in-place).
        compression_method: 'taylor' or 'hessian'.
        train_loader: calibration DataLoader supplying (x, y) batches.
        compression_ratio: scalar ratio used when per_block_ratios is None.
        device: torch.device.
        num_batches: number of calibration batches.
        per_block_ratios: optional dict mapping block keys ('conv1', 'layer1',
            ..., 'layer4') to per-block compression ratios.  When provided,
            the scalar compression_ratio is ignored and a pruning_ratio_dict
            is built for each Conv2d layer.

    Returns:
        (model, keep_indices) tuple.
        keep_indices maps conv layer names to LongTensor of surviving
        output-channel positions for barrier reconstruction.
    """
    input_res = 32
    if hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
        k = model.conv1.kernel_size
        if isinstance(k, tuple): k = k[0]
        if k == 7:
            input_res = 224

    example_inputs = torch.randn(1, 3, input_res, input_res)
    example_inputs = example_inputs.to(device)
    ex_in = example_inputs

    ignored_layers = wanda_like_ignored_layers_resnet18(model=model)
    importance = build_importance(compression_method=compression_method)

    # Build per-layer pruning_ratio_dict when a non-uniform schedule is provided
    pruning_ratio_dict = None
    if per_block_ratios is not None:
        pruning_ratio_dict = {}
        block_order = ('conv1', 'layer1', 'layer2', 'layer3', 'layer4')
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module not in ignored_layers:
                matched_ratio = compression_ratio  # fallback
                for block_key in block_order:
                    if name == block_key or name.startswith(block_key + '.'):
                        matched_ratio = per_block_ratios.get(block_key, compression_ratio)
                        break
                pruning_ratio_dict[module] = matched_ratio

    # Record pre-pruning output channel counts for keep_indices extraction
    pre_channels = {}
    module_to_name = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module not in ignored_layers:
            pre_channels[name] = module.out_channels
            module_to_name[id(module)] = name

    pruner = tp.pruner.BasePruner(
        model=model,
        example_inputs=ex_in,
        importance=importance,
        iterative_steps=1,  # one-shot
        pruning_ratio=compression_ratio,  # scalar fallback (ignored when dict provided)
        pruning_ratio_dict=pruning_ratio_dict,  # per-layer override (None = uniform)
        ignored_layers=ignored_layers,
        global_pruning=False
    )
    # Taylor/Hessian need (second-order) grads from a tiny calibration set
    needs_grad = isinstance(importance, (tp.importance.GroupTaylorImportance,
                                         tp.importance.GroupHessianImportance))
    if needs_grad:
        assert train_loader is not None, \
            "Provide a small calibration loader."
        model.zero_grad()

        for param in model.parameters():
            param.requires_grad = True

        if isinstance(importance, tp.importance.GroupHessianImportance):
            importance.zero_grad()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)

            if isinstance(importance, tp.importance.GroupHessianImportance):
                loss = torch.nn.functional.cross_entropy(logits, yb, reduction='none')
                for l in loss:
                    model.zero_grad()
                    l.backward(retain_graph=True)
                    importance.accumulate_grad(model)
            elif isinstance(importance, tp.importance.GroupTaylorImportance):
                loss = torch.nn.functional.cross_entropy(logits, yb)
                loss.backward()

            if num_batches is not None:
                num_batches -= 1
                if num_batches <= 0:
                    break

    # interactive=True returns the groups without applying them, so keep_indices can be derived
    keep_indices = {}
    groups = pruner.step(interactive=True)
    for group in groups:
        for dep, idxs in group:
            target_module = dep.target.module
            mid = id(target_module)
            if mid in module_to_name and isinstance(target_module, nn.Conv2d):
                name = module_to_name[mid]
                if name in pre_channels:
                    n_out = pre_channels[name]
                    # idxs contains the channels being PRUNED
                    pruned_set = set(idxs.tolist()) if hasattr(idxs, 'tolist') else set()
                    kept = sorted([i for i in range(n_out) if i not in pruned_set])
                    if len(kept) < n_out:
                        keep_indices[name] = torch.tensor(kept, dtype=torch.long)
        group.prune()

    # Attached to the model so callers can retrieve it without a signature change
    model._keep_indices_ = keep_indices

    return model


def one_shot_structural_pruning_small_model(model, compression_method, train_loader,
                                            compression_ratio, device, num_batches=50):
    """One-shot DepGraph structural pruning for small sequential models (MLP-BN, LeNet5-BN).

    Same as one_shot_structural_pruning_resnet18 but with 32x32 CIFAR-10 inputs;
    only the final classifier (last nn.Linear) is ignored.
    """
    example_inputs = torch.randn(1, 3, 32, 32).to(device)

    # Ignore the final classifier layer
    ignored_layers = []
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        ignored_layers.append(last_linear)

    importance = build_importance(compression_method)

    pruner = tp.pruner.BasePruner(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=1,
        pruning_ratio=compression_ratio,
        ignored_layers=ignored_layers,
        global_pruning=False,
    )

    needs_grad = isinstance(importance, (tp.importance.GroupTaylorImportance,
                                         tp.importance.GroupHessianImportance))
    if needs_grad:
        assert train_loader is not None, "Provide a calibration loader."
        model.zero_grad()
        for param in model.parameters():
            param.requires_grad = True

        if isinstance(importance, tp.importance.GroupHessianImportance):
            importance.zero_grad()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)

            if isinstance(importance, tp.importance.GroupHessianImportance):
                loss = torch.nn.functional.cross_entropy(logits, yb, reduction='none')
                for l in loss:
                    model.zero_grad()
                    l.backward(retain_graph=True)
                    importance.accumulate_grad(model)
            elif isinstance(importance, tp.importance.GroupTaylorImportance):
                loss = torch.nn.functional.cross_entropy(logits, yb)
                loss.backward()

            if num_batches is not None:
                num_batches -= 1
                if num_batches <= 0:
                    break

    pruner.step()
    return model


def forward(self, x):
    """Adapted from https://github.com/huggingface/pytorch-image-models/blob/054c763fcaa7d241564439ae05fbe919ed85e614/timm/models/vision_transformer.py"""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, -1)  # original implementation: x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def one_shot_structural_pruning_vit(model, compression_method, train_loader, compression_ratio, device, num_batches=50):
    example_inputs = torch.randn(1, 3, 224, 224)
    example_inputs = example_inputs.to(device)

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if m.out_features == model.num_classes:
                ignored_layers.append(m)  # do not prune the classifier layer
            if hasattr(m, 'fc1') or hasattr(m, 'fc2'):
                ignored_layers.append(m)  # only prune the internal layers of FFN & Attention

    importance = build_importance(compression_method=compression_method)

    num_heads = {}
    ignored_layers = [model.head]
    for m in model.modules():
        if isinstance(m, timm.models.vision_transformer.Attention):
            m.forward = forward.__get__(m,
                                        timm.models.vision_transformer.Attention)  # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
            num_heads[m.qkv] = m.num_heads

            ignored_layers.extend([m.qkv, m.proj])

        if isinstance(m, timm.models.vision_transformer.Mlp):
            ignored_layers.append(m.fc2)  # only prune the internal layers of FFN & Attention

    pruner = tp.pruner.BasePruner(
        model,
        example_inputs,
        global_pruning=False,
        # If False, a uniform pruning ratio will be assigned to different layers.
        importance=importance,  # importance criterion for parameter selection
        pruning_ratio=compression_ratio,  # target pruning ratio
        ignored_layers=ignored_layers,
        num_heads=num_heads,  # number of heads in self attention
        prune_num_heads=False,  # reduce num_heads by pruning entire heads (default: False)
        prune_head_dims=False,  # reduce head_dim by pruning features dims of each head (default: True)
        head_pruning_ratio=0.0,
        round_to=None
    )

    if isinstance(importance, (tp.importance.GroupTaylorImportance, tp.importance.GroupHessianImportance)):
        model.zero_grad()
        if isinstance(importance, tp.importance.GroupHessianImportance):
            importance.zero_grad()
        for k, (imgs, lbls) in enumerate(train_loader):
            if k >= num_batches: break
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            output = model(imgs)
            if isinstance(importance, tp.importance.GroupHessianImportance):
                loss = torch.nn.functional.cross_entropy(output, lbls, reduction='none')
                for l in loss:
                    model.zero_grad()
                    l.backward(retain_graph=True)
                    importance.accumulate_grad(model)
            elif isinstance(importance, tp.importance.GroupTaylorImportance):
                loss = torch.nn.functional.cross_entropy(output, lbls)
                loss.backward()

    for g in pruner.step(interactive=True):
        g.prune()

    # Modify the attention head size and all head size aftering pruning
    head_id = 0
    for m in model.modules():
        if isinstance(m, timm.models.vision_transformer.Attention):
            m.num_heads = pruner.num_heads[m.qkv]
            m.head_dim = m.qkv.out_features // (3 * m.num_heads)
            head_id += 1

    return model
