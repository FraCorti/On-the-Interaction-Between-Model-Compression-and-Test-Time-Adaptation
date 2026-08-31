from collections import defaultdict

import numpy as np
import torch

try:
    from hkmeans import HKMeans
except ImportError:
    HKMeans = None

from compression.base_resnet import BaseResNetCompression
from compression.base_vit import BaseViTCompression
import warnings

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch, DBSCAN
from sklearn.linear_model._cd_fast import ConvergenceWarning

from models.resnet import get_module_by_name_ResNet
from utils.weight_clustering import merge_channel_clustering


def axes2perm_to_perm2axes(axes_to_perm):
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return perm_to_axes


def get_axis_to_perm_ResNet18(approx_repair=False, override=True):
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,)}

    if approx_repair is True:
        conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,), f"{name}.bias": (p_out,)}

    norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,), f"{name}.running_mean": (p,),
                            f"{name}.running_var": (p,)}

    sh = "shortcut"
    fc = "linear"
    if override is False:
        sh = "downsample"
        fc = "fc"

    dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out,)}
    basicblock = lambda name, p_in, p_out: {
        **conv(f"{name}.conv1", p_in, f"{name}.relu1"),
        **norm(f"{name}.bn1", f"{name}.relu1"),
        **conv(f"{name}.conv2", f"{name}.relu1", p_out),
        **norm(f"{name}.bn2", p_out),
    }
    axis_to_perm = {
        **conv('conv1', None, 'relu_conv'),
        **norm('bn1', 'relu_conv'),

        **basicblock('layer1.0', 'relu_conv', 'relu_conv'),
        **basicblock('layer1.1', 'relu_conv', 'relu_conv'),

        **basicblock('layer2.0', 'relu_conv', 'layer2.0.relu2'),
        **conv(f'layer2.0.{sh}.0', 'relu_conv', 'layer2.0.relu2'),
        **norm(f'layer2.0.{sh}.1', 'layer2.0.relu2'),
        **basicblock('layer2.1', 'layer2.0.relu2', 'layer2.0.relu2'),

        **basicblock('layer3.0', 'layer2.0.relu2', 'layer3.0.relu2'),
        **conv(f'layer3.0.{sh}.0', 'layer2.0.relu2', 'layer3.0.relu2'),
        **norm(f'layer3.0.{sh}.1', 'layer3.0.relu2'),
        **basicblock('layer3.1', 'layer3.0.relu2', 'layer3.0.relu2'),

        **basicblock('layer4.0', 'layer3.0.relu2', 'layer4.0.relu2'),
        **conv(f'layer4.0.{sh}.0', 'layer3.0.relu2', 'layer4.0.relu2'),
        **norm(f'layer4.0.{sh}.1', 'layer4.0.relu2'),
        **basicblock('layer4.1', 'layer4.0.relu2', 'layer4.0.relu2'),

        **dense(f'{fc}', 'layer4.0.relu2', None)
    }

    return axis_to_perm


def concat_weights(perm_to_axes, params, p_name, n, approx_repair=False):
    A = None
    for wk, axis in perm_to_axes[p_name]:
        w_a = params[wk]

        if "running" in wk or "identity_transform" in wk:
            continue

        if approx_repair is True:
            if axis == 1:
                continue

            if len(w_a.shape) < 2:
                continue

        w_a = torch.movedim(w_a, axis, 0).reshape((n, -1))

        if A is None:
            A = w_a
        else:
            A = torch.hstack((A, w_a))

    return A.cpu()


class WeightClustering:
    """
    Unified clustering for folding/pruning:
      - Supports HKMeans (default) and multiple sklearn clusterers
      - Optional PCA (default True) and normalization (default False)
      - Can return either cluster labels or binary merge matrix
    """

    def __init__(self, n_clusters: int, n_features: int = None,
                 method: str = "hkmeans",
                 # Options: "hkmeans", "kmeans", "agglomerative", "spectral", "birch", "dbscan"
                 normalize: bool = False, use_pca: bool = False, return_matrix: bool = False):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.method = method.lower()
        self.normalize = normalize
        self.use_pca = use_pca
        self.return_matrix = return_matrix

    def _get_clustering_algorithm(self):
        """Return clustering object based on selected method."""
        if self.method == "hkmeans":
            if HKMeans is None:
                print("Warning: hkmeans not installed, falling back to kmeans")
                return KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10, max_iter=300)
            return HKMeans(n_clusters=self.n_clusters, random_state=42, n_init=1, n_jobs=-1,
                           max_iter=10, verbose=False)
        if self.method == "kmeans":
            return KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10, max_iter=300)
        if self.method == "agglomerative":
            return AgglomerativeClustering(n_clusters=self.n_clusters)
        if self.method == "spectral":
            return SpectralClustering(n_clusters=self.n_clusters, assign_labels='kmeans')
        if self.method == "birch":
            return Birch(n_clusters=self.n_clusters)
        if self.method == "dbscan":
            return DBSCAN()
        raise ValueError(f"Unsupported clustering method: {self.method}")

    def __call__(self, tensor: torch.Tensor):
        """
        Cluster input tensor and return either labels or merge matrix.
        Merge matrix is always [clusters, channels].
        """
        # --- Flatten tensor to [channels, features] ---
        X = tensor.view(tensor.shape[0], -1).cpu().numpy()

        # --- Normalize if enabled ---
        if self.normalize:
            X = preprocessing.RobustScaler().fit_transform(X)

        # Replace NaNs/Infs before PCA
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # --- PCA if enabled ---
        if self.use_pca:
            n_samples, n_features = X.shape
            X = PCA(n_components=min(n_samples, n_features)).fit_transform(X)

        # --- Clustering ---
        clustering = self._get_clustering_algorithm()
        # Some clustering methods (e.g., HKMeans) may raise warnings during convergence.
        # E.g., when finding fewer clusters than requested.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clustering.fit(X)
        labels = clustering.labels_

        # Handle fewer-than-expected clusters (KMeans collapse)
        unique_labels = np.unique(labels)
        if len(unique_labels) < self.n_clusters:
            # Remap labels to be contiguous 0..num_clusters-1
            label_map = {old: new for new, old in enumerate(unique_labels)}
            labels = np.array([label_map[l] for l in labels])

        labels_torch = torch.tensor(labels, dtype=torch.int64, device=tensor.device)

        # Return either labels or merge matrix
        if not self.return_matrix:
            return labels_torch

        # Build merge matrix (clusters x channels)
        num_clusters = labels_torch.max().item() + 1
        num_channels = labels_torch.shape[0]
        merge = torch.zeros((num_clusters, num_channels), device=tensor.device, dtype=torch.float32)
        merge.scatter_(0, labels_torch.unsqueeze(0), 1.0)
        merge /= merge.sum(dim=1, keepdim=True).clamp(min=1)

        return merge


class ViT_ModelFolding(BaseViTCompression):
    """
    Cluster hidden units in each block's MLP (fc1), replace them by cluster means,
    and sum corresponding columns in fc2. Bias for fc1 is averaged per cluster;
    fc2 bias is kept unchanged.
    """

    def __init__(self, model, min_channels=1, compression_ratio=0.5,
                 cluster_method="hkmeans", use_pca=False, normalize=False):
        super().__init__(model, min_channels, compression_ratio)
        self.cluster_method = cluster_method
        self.cluster_use_pca = use_pca
        self.cluster_normalize = normalize

    def compress_function(self, axes, params):
        compressed, merge_sizes = {}, {}
        name_fc1, name_fc2 = axes

        # W_fc1: [H, D], W_fc2: [D, H]
        W_fc1 = params[name_fc1 + '.weight']
        W_fc2 = params[name_fc2 + '.weight']
        device, dtype = W_fc1.device, W_fc1.dtype

        H, D = W_fc1.shape
        n_clusters = max(int(H * self.keep_ratio), self.min_channels)
        n_clusters = min(n_clusters, H)

        # --- column-wise z-score on fc1 so each input dim contributes comparably
        eps = torch.finfo(dtype).eps
        col_mean = W_fc1.mean(dim=0, keepdim=True)  # [1, D]
        col_std = W_fc1.std(dim=0, unbiased=False, keepdim=True) + eps
        W_norm = (W_fc1 - col_mean) / col_std  # [H, D]

        # --- clustering on normalized rows
        clusterer = WeightClustering(
            n_clusters=n_clusters,
            method=self.cluster_method,
            use_pca=self.cluster_use_pca,
            normalize=False  # already normalized above
        )
        labels = clusterer(W_norm).to(device).long()

        uniq = torch.unique(labels, sorted=True)
        if uniq.numel() < n_clusters:
            # collapse to actually used cluster ids
            remap = {int(lbl): i for i, lbl in enumerate(uniq.tolist())}
            labels = torch.tensor([remap[int(l.item())] for l in labels], device=device, dtype=torch.long)
            n_clusters = uniq.numel()

        # --- build new fc1 rows (de-normalized cluster means) and fc2 columns (sum)
        cluster_means = []
        proj_cols = []
        for cid in range(n_clusters):
            members = (labels == cid).nonzero(as_tuple=True)[0]  # indices in [0..H-1]
            mean_norm = W_norm[members, :].mean(dim=0, keepdim=False)  # [D]
            mean_vec = mean_norm * col_std.squeeze(0) + col_mean.squeeze(0)  # [D]
            cluster_means.append(mean_vec)

            proj_sum = W_fc2[:, members].sum(dim=1, keepdim=True)  # [D, 1]
            proj_cols.append(proj_sum)

        new_fc1 = torch.stack(cluster_means, dim=0)  # [H', D]
        new_fc2 = torch.cat(proj_cols, dim=1)  # [D, H']

        compressed[name_fc1 + '.weight'] = new_fc1.to(device=device, dtype=dtype)
        compressed[name_fc2 + '.weight'] = new_fc2.to(device=device, dtype=dtype)

        # --- fc1 bias averaged per cluster; fc2 bias unchanged
        if name_fc1 + '.bias' in params and params[name_fc1 + '.bias'] is not None:
            b1 = params[name_fc1 + '.bias']
            new_b1 = []
            for cid in range(n_clusters):
                members = (labels == cid).nonzero(as_tuple=True)[0]
                new_b1.append(b1[members].mean())
            compressed[name_fc1 + '.bias'] = torch.stack(new_b1, dim=0).to(device=device, dtype=dtype)

        if name_fc2 + '.bias' in params and params[name_fc2 + '.bias'] is not None:
            compressed[name_fc2 + '.bias'] = params[name_fc2 + '.bias']  # unchanged

        merge_sizes[name_fc1] = new_fc1.shape[0]
        merge_sizes[name_fc2] = new_fc2.shape[1]
        # fc1 rows and fc2 columns share one cluster assignment; the labels are
        # kept for barrier reconstruction (cf. Wang et al., arXiv:2502.10216).
        merge_sizes['cluster_labels'] = {
            name_fc1: labels.detach().cpu(),
            name_fc2: labels.detach().cpu(),
        }
        return compressed, merge_sizes


class ResNet18_ModelFolding(BaseResNetCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5,
                 per_block_ratios=None):
        super().__init__(model, min_channels, compression_ratio,
                         per_block_ratios=per_block_ratios)

    def _get_keep_ratio_for_axes(self, axes):
        """Infer the keep ratio for a permutation group from its representative axis.

        The first axis name in the group (e.g. 'conv1.weight', 'layer2.0.conv1.weight')
        is stripped of the '.weight' suffix and used as input to _get_keep_ratio(),
        which resolves the block key ('conv1', 'layer1', ..., 'layer4').
        """
        # axes is a list of (module_name, axis_index) tuples; take the first module_name
        representative = axes[0][0]  # e.g. 'conv1', 'layer2.0.conv1', 'layer2.0.downsample.0'
        return self._get_keep_ratio(representative)

    def compress_function(self, axes, params):
        """
        Folding logic: perform clustering and merge weights.
        """
        # Always cluster on output channels
        n_channels = params[axes[0][0]].shape[0]
        keep_ratio = self._get_keep_ratio_for_axes(axes)
        n_clusters = max(int(n_channels * keep_ratio), 1)

        # Flatten weights across layers (output dim)
        weight = concat_weights({0: axes}, params, 0, n_channels)

        # Cluster
        clusterer = WeightClustering(n_clusters=n_clusters, n_features=n_channels,
                                     method="hkmeans", normalize=False, use_pca=True)
        labels = clusterer(weight).to(self.device).long()

        # Build merge matrix
        merge_matrix = torch.zeros((n_clusters, n_channels), device=self.device)
        merge_matrix.scatter_(0, labels.unsqueeze(0), 1.0)
        cluster_counts = merge_matrix.sum(dim=1, keepdim=True)
        merge_matrix /= cluster_counts.clamp(min=1)  # avoid div by zero

        # Merge params
        from utils.weight_clustering import NopMerge
        compressed_params = merge_channel_clustering({0: axes}, params, 0, merge_matrix, custom_merger=NopMerge())

        return compressed_params, {'cluster_labels': labels}

    def apply(self):
        print(f"[INFO] Starting {self.__class__.__name__}...")
        axis_to_perm = get_axis_to_perm_ResNet18(override=False)
        perm_to_axes = axes2perm_to_perm2axes(axis_to_perm)

        for perm_id, axes in perm_to_axes.items():
            # --- Collect raw params ---
            raw_params = {}
            for module_name, axis in axes:
                module = get_module_by_name_ResNet(self.model, module_name)
                weight = module.weight.data if hasattr(module, 'weight') else module.data
                raw_params[module_name] = weight

            # --- Call specific compression/pruning logic ---
            compressed_params, meta = self.compress_function(axes, raw_params)

            # --- Rebuild modules ---
            param_groups = defaultdict(dict)
            for full_name, tensor in compressed_params.items():
                module_name, pname = full_name.rsplit('.', 1)
                param_groups[module_name][pname] = tensor

            cluster_labels = meta.get('cluster_labels') if meta else None

            # Store labels only for modules clustered along axis 0 (output channels);
            # for axis-1 modules the labels describe input channels.
            if cluster_labels is not None:
                # axes entries are (state_dict_key, axis) e.g. ('conv1.weight', 0)
                # Strip suffix to get module names matching param_groups keys
                axis0_modules = set()
                for sd_key, axis in axes:
                    if axis == 0:
                        for suffix in ('.weight', '.bias', '.running_mean', '.running_var'):
                            if sd_key.endswith(suffix):
                                axis0_modules.add(sd_key[:-len(suffix)])
                                break
                for module_name in param_groups:
                    if module_name in axis0_modules:
                        self.cluster_labels_[module_name] = cluster_labels.detach().cpu()

            for module_name, param_dict in param_groups.items():
                module = get_module_by_name_ResNet(self.model, module_name)
                n_clusters = param_dict['weight'].shape[0]
                new_module = self._rebuild_module(module, param_dict, cluster_labels, n_clusters)
                self._replace(module_name, new_module)

        return self.model
