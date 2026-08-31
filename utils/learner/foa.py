"""
Copyright to FOA Authors ICML 2024
Forward-Only Adaptation: Test-Time Adaptation without Backpropagation

Reference: https://arxiv.org/abs/2404.01650
GitHub: https://github.com/mr-eggplant/FOA
"""

import torch
import torch.nn as nn
import torch.jit

from utils.vpt import PromptViT
try:
    import cma
except ImportError:
    cma = None
    print("Warning: CMA-ES not installed. Install with: pip install cma")
import numpy as np
from .learner import register

# Optional quantization library imports (for compatibility with quantized models)
try:
    from quant_library.quant_layers.matmul import PTQSLBatchingQuantMatMul, SoSPTQSLBatchingQuantMatMul
except ImportError:
    PTQSLBatchingQuantMatMul = None
    SoSPTQSLBatchingQuantMatMul = None

__all__ = ['foa']

RUNNING_IMAGNET_R = False


@register('foa')
def foa(model, optimizer=None, fitness_lambda=0.4):
    return FOA(model=model, fitness_lambda=fitness_lambda)


class FOA(nn.Module):
    """test-time Forward Optimization Adaptation
    FOA devises both input level and output level adaptation.
    It avoids modification to model weights and adapts in a backpropogation-free manner.
    """

    def __init__(self, model: PromptViT, fitness_lambda=0.4):
        super().__init__()
        self.fitness_lambda = fitness_lambda

        self.model = model
        self.es = self._init_cma()  # initialization for CMA-ES

        self.best_prompts = model.prompts
        self.best_loss = np.inf
        self.hist_stat = None  # which is used for calculating the shift direction in Eqn. (8)
        self.imagenet_mask = None  # used for ImageNet-R

    def _init_cma(self):
        """CMA-ES initialization"""
        dim = self.model.prompts.numel()
        popsize = 27  # which is equal to 4 + 3 * np.log(dim) when #prompts=3
        cma_opts = {
            'seed': 2020,
            'popsize': popsize,
            'maxiter': -1,
            'verbose': -1,
        }
        es = cma.CMAEvolutionStrategy(dim * [0], 1, inopts=cma_opts)
        self.popsize = es.popsize
        return es

    def _update_hist(self, batch_mean):
        """Update overall test statistics, Eqn. (9)"""
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean

    def _get_shift_vector(self):
        """Calculate shift direction, Eqn. (8)"""
        if self.hist_stat is None:
            return None
        else:
            return self.train_info[1][-768:] - self.hist_stat

    @torch.no_grad()
    def forward(self, x):
        """Forward pass with CMA-ES prompt optimization, Eqn. (8)"""
        device = x.device
        shift_vector = self._get_shift_vector()

        self.best_loss, self.best_outputs, batch_means = np.inf, None, []

        """Sampling from CMA-ES and evaluate the new solutions.
        Note that we also compare the current solutions with the previous best one"""
        # Detach best_prompts to avoid numpy conversion error in CMA-ES
        prompts, losses = self.es.ask() + [self.best_prompts.flatten().detach().cpu()], []
        for j, prompt in enumerate(prompts):
            self.model.prompts = torch.nn.Parameter(torch.tensor(prompt, dtype=torch.float).
                                                    reshape_as(self.model.prompts).to(device))
            self.model.prompts.requires_grad_(False)

            outputs, loss, batch_mean = forward_and_get_loss(images=x, model=self.model,
                                                             fitness_lambda=self.fitness_lambda,
                                                             train_info=self.train_info,
                                                             shift_vector=shift_vector,
                                                             imagenet_mask=self.imagenet_mask)
            batch_means.append(batch_mean[-768:].unsqueeze(0))
            del batch_mean

            if self.best_loss > loss.item():
                self.best_prompts = self.model.prompts
                self.best_loss = loss.item()
                self.best_outputs = outputs
                outputs = None
            losses.append(loss.item())
            del outputs


        """CMA-ES updates, Eqn. (6)"""
        self.es.tell(prompts, losses)

        """Update overall test statistics, Eqn. (9)"""
        batch_means = torch.cat(batch_means, dim=0).mean(0)
        self._update_hist(batch_means)
        return self.best_outputs

    def obtain_origin_stat(self, train_loader, device='cuda', max_batches=None):
        """Collect training set statistics for FOA adaptation.

        Args:
            max_batches: cap on the number of source mini-batches used for
              the (std, mean) statistics of Eqn. (8).
        """
        features = []
        with torch.no_grad():
            for i, dl in enumerate(train_loader):
                if max_batches is not None and i >= int(max_batches):
                    break
                images = dl[0].to(device)
                feature = self.model.layers_cls_features(images)
                features.append(feature)
            features = torch.cat(features, dim=0)
            self.train_info = torch.std_mean(features, dim=0)
        del features
        
        # Preparing quantized model for prompt adaptation (only if quant modules present)
        if PTQSLBatchingQuantMatMul is not None or SoSPTQSLBatchingQuantMatMul is not None:
            for _, m in self.model.vit.named_modules():
                if PTQSLBatchingQuantMatMul is not None and type(m) == PTQSLBatchingQuantMatMul:
                    m._get_padding_parameters(
                        torch.zeros((1, 12, 197 + self.model.num_prompts, 64)).to(device),
                        torch.zeros((1, 12, 64, 197 + self.model.num_prompts)).to(device))
                elif SoSPTQSLBatchingQuantMatMul is not None and type(m) == SoSPTQSLBatchingQuantMatMul:
                    m._get_padding_parameters(
                        torch.zeros((1, 12, 197 + self.model.num_prompts, 197 + self.model.num_prompts)).to(device),
                        torch.zeros((1, 12, 197 + self.model.num_prompts, 64)).to(device))

    def reset(self):
        self.es = self._init_cma()
        self.hist_stat = None

        self.model.reset()
        self.best_prompts = self.model.prompts


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x / temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


criterion_mse = nn.MSELoss(reduction='none').cuda()


@torch.no_grad()
def forward_and_get_loss(images, model: PromptViT, train_info, shift_vector, imagenet_mask, fitness_lambda=0.4):
    features = model.layers_cls_features_with_prompts(images)

    """discrepancy loss for Eqn. (5)"""
    batch_std, batch_mean = torch.std_mean(features, dim=0)
    std_mse, mean_mse = criterion_mse(batch_std, train_info[0]), criterion_mse(batch_mean, train_info[1])
    # lambda should be 0.2 for ImageNet-R
    discrepancy_loss = fitness_lambda * (std_mse.sum() + mean_mse.sum()) * images.shape[0] / 64

    cls_features = features[:, -768:]  # the feature of classification token
    output = model.vit.head(cls_features)

    """entropy loss for Eqn. (5)"""
    if imagenet_mask is not None:
        output = output[:, imagenet_mask]
    entropy_loss = softmax_entropy(output).sum()
    loss = discrepancy_loss + entropy_loss

    """activation shifting, Eqn. (7)"""
    if shift_vector is not None:
        output = model.vit.head(cls_features + 1. * shift_vector)
        if imagenet_mask is not None:
            output = output[:, imagenet_mask]

    return output, loss, batch_mean
