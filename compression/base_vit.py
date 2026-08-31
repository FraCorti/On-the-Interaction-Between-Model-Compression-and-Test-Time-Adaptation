import torch
import torch.nn as nn

def get_axis_to_perm_ViT_TIMM(model: nn.Module):
    """
    Return mapping {group_name: (fc1_path, fc2_path)} for every transformer block.
    We pair the MLP layers: blocks.i.mlp.fc1 <-> blocks.i.mlp.fc2
    """
    axis = {}
    # model.blocks is a ModuleList in timm ViTs
    for i, _ in enumerate(model.blocks):
        axis[f'block{i}_mlp'] = (f'blocks.{i}.mlp.fc1', f'blocks.{i}.mlp.fc2')
    return axis


def get_module_by_name(root: nn.Module, name: str) -> nn.Module:
    """
    Resolve dotted paths like 'blocks.0.mlp.fc1' on a timm ViT.
    Supports numeric indices for ModuleList/Sequential.
    """
    cur = root
    for part in name.split('.'):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def get_parent_and_attr(root: nn.Module, name: str):
    """
    Return (parent_module, attribute_name) for a dotted path.
    """
    parts = name.split('.')
    parent_path, attr = '.'.join(parts[:-1]), parts[-1]
    parent = get_module_by_name(root, parent_path) if parent_path else root
    return parent, attr


class BaseViTCompression(nn.Module):
    """
    Base class for ViT MLP compression (magnitude/Wanda/etc.) compatible with timm ViTs.
    - Iterates over all blocks and targets (fc1, fc2) pairs in each MLP.
    - Subclasses implement `compress_function(axes, params)` which must return
      (compressed_params_dict, merge_sizes_dict).
    """
    def __init__(self, model: nn.Module, min_channels: int = 1, compression_ratio: float = 0.5):
        super().__init__()
        self.model = model
        self.min_channels = int(min_channels)
        self.keep_ratio = float(1.0 - compression_ratio)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.keep_indices_ = {}  # fc1_name -> LongTensor of original hidden-unit positions kept

        # cache names->modules for Wanda calibration, etc.
        self._named_modules = dict(self.model.named_modules())

    # to be implemented by subclasses
    def compress_function(self, axes, params):
        raise NotImplementedError

    @torch.no_grad()
    def apply(self):
        self.model.to(self.device)
        self.model.eval()  # safe default; keep shapes deterministic

        axis_to_perm = get_axis_to_perm_ViT_TIMM(self.model)
        self.cluster_labels_ = {}  # fc1_name -> LongTensor[H_original]

        for group_name, (name_fc1, name_fc2) in axis_to_perm.items():
            # fetch layers
            fc1 = get_module_by_name(self.model, name_fc1)  # Linear(in=D, out=H)
            fc2 = get_module_by_name(self.model, name_fc2)  # Linear(in=H, out=D)

            # only Linear pairs are supported
            if not (isinstance(fc1, nn.Linear) and isinstance(fc2, nn.Linear)):
                print(f"[WARN] Skip {group_name}: non-Linear pair ({type(fc1)}, {type(fc2)})")
                continue

            W_fc1 = fc1.weight.data   # [H, D]
            W_fc2 = fc2.weight.data   # [D, H]
            raw_params = {
                name_fc1 + '.weight': W_fc1,
                name_fc2 + '.weight': W_fc2,
            }
            if fc1.bias is not None:
                raw_params[name_fc1 + '.bias'] = fc1.bias.data
            if fc2.bias is not None:
                raw_params[name_fc2 + '.bias'] = fc2.bias.data

            # delegate to subclass
            compressed, merge_info = self.compress_function((name_fc1, name_fc2), raw_params)

            # Accumulate cluster labels if subclass provides them (e.g. Fold)
            if isinstance(merge_info, dict) and 'cluster_labels' in merge_info:
                self.cluster_labels_.update(merge_info['cluster_labels'])

            # Accumulate keep_indices if subclass provides them (non-Fold pruning)
            if isinstance(merge_info, dict) and 'keep_indices' in merge_info:
                self.keep_indices_.update(merge_info['keep_indices'])

            # build new modules with pruned hidden size
            new_fc1_W = compressed[name_fc1 + '.weight']   # [H', D]
            new_fc2_W = compressed[name_fc2 + '.weight']   # [D, H']

            H_prime, D = new_fc1_W.shape
            D2, H_prime2 = new_fc2_W.shape
            assert D2 == D and H_prime2 == H_prime, \
                f"Shape mismatch after compression: fc1 {new_fc1_W.shape}, fc2 {new_fc2_W.shape}"

            new_fc1 = nn.Linear(in_features=D, out_features=H_prime, bias=fc1.bias is not None).to(self.device, self.dtype)
            new_fc1.weight.data.copy_(new_fc1_W.to(self.device, self.dtype))
            if fc1.bias is not None:
                b1 = compressed.get(name_fc1 + '.bias', None)
                if b1 is None:
                    # fallback: keep bias for surviving units if provided later
                    raise RuntimeError(f"Missing bias for {name_fc1} in compressed params")
                new_fc1.bias.data.copy_(b1.to(self.device, self.dtype))

            new_fc2 = nn.Linear(in_features=H_prime, out_features=D, bias=fc2.bias is not None).to(self.device, self.dtype)
            new_fc2.weight.data.copy_(new_fc2_W.to(self.device, self.dtype))
            if fc2.bias is not None:
                b2 = compressed.get(name_fc2 + '.bias', None)
                if b2 is None:
                    b2 = fc2.bias.data  # unchanged
                new_fc2.bias.data.copy_(b2.to(self.device, self.dtype))

            # replace in the model
            parent1, attr1 = get_parent_and_attr(self.model, name_fc1)
            parent2, attr2 = get_parent_and_attr(self.model, name_fc2)
            setattr(parent1, attr1, new_fc1)
            setattr(parent2, attr2, new_fc2)

            # refresh cache
            self._named_modules = dict(self.model.named_modules())

        return self.model

