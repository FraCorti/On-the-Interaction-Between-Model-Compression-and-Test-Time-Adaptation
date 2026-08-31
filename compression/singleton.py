import torch
from utils.weight_clustering import merge_channel_clustering, NopMerge

from compression.fold import ResNet18_ModelFolding


class ResNet18_Singleton(ResNet18_ModelFolding):
    def compress_function(self, axes, params):
        """
        Singleton folding:
        - Top (k-1) channels (highest L2 norm) form singleton clusters.
        - Remaining channels grouped into one cluster.
        """
        # Number of channels and clusters
        n_channels = params[axes[0][0]].shape[axes[0][1]]
        n_clusters = max(int(n_channels * self.keep_ratio), 1)

        # Compute per-channel L2 norm (flatten along remaining dimensions)
        weight = params[axes[0][0]]
        magnitudes = torch.norm(weight.view(n_channels, -1), p=2, dim=1)

        # Get indices of top (k-1) channels
        if n_clusters > 1:
            topk = torch.topk(magnitudes, n_clusters - 1, largest=True).indices
        else:
            topk = torch.tensor([], device=self.device, dtype=torch.long)

        # Build labels: top-k are unique, others share one cluster
        labels = torch.full((n_channels,), n_clusters - 1, device=self.device, dtype=torch.long)
        for idx, ch in enumerate(topk):
            labels[ch] = idx

        # Build merge matrix
        merge_matrix = torch.zeros((n_clusters, n_channels), device=self.device)
        merge_matrix.scatter_(0, labels.unsqueeze(0), 1.0)
        merge_matrix /= merge_matrix.sum(dim=1, keepdim=True).clamp(min=1)

        # Merge weights (folding)
        compressed_params = merge_channel_clustering({0: axes}, params, 0, merge_matrix, custom_merger=NopMerge())

        return compressed_params, {'cluster_labels': labels}
