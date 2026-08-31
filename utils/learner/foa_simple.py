"""
Simplified Forward-Only Adaptation (FOA) for Vision Transformers, inspired by
https://arxiv.org/abs/2404.01650 (ICML 2024). Uses only the BP-free activation
shifting component: test-batch feature statistics are aligned to training
statistics; no visual prompts or CMA-ES.
"""

import torch
import torch.nn as nn
import torch.jit
import numpy as np
from .learner import register
from copy import deepcopy

__all__ = ['foa_simple']


@register('foa_simple')
def foa_simple(model, optimizer=None, alpha=0.9, **kwargs):
    """
    Factory function for simplified FOA adaptation (BP-free).
    
    Args:
        model: ViT model to adapt
        optimizer: Ignored (FOA is BP-free)
        alpha: EMA momentum for updating history statistics
    
    Returns:
        FOASimple wrapper around the model
    """
    return FOASimple(model, alpha=alpha)


class FOASimple(nn.Module):
    """
    Simplified Forward-Only Adaptation for Vision Transformers.
    
    BP-free adaptation via activation shifting based on feature statistics.
    Does not require visual prompts or CMA-ES optimization.
    
    Key idea: 
    - Collect training set feature statistics (mean, std)
    - At test time, compute batch statistics and shift features to align
    """
    
    def __init__(self, model, alpha=0.9):
        super().__init__()
        self.model = model
        self.alpha = alpha  # EMA momentum for history
        
        # Training statistics (to be set via obtain_origin_stat)
        self.train_mean = None
        self.train_std = None
        
        # Running history of test statistics
        self.hist_mean = None
        
        # Store original state for reset
        self.model_state = deepcopy(self.model.state_dict())
        
        # ImageNet-R mask for class subset (optional)
        self.imagenet_mask = None
    
    def forward(self, x):
        """
        Forward pass with activation shifting.
        """
        # Get model output
        outputs = self.model(x)
        
        if self.imagenet_mask is not None:
            outputs = outputs[:, self.imagenet_mask]
        
        # If we have training statistics, apply activation shifting
        if self.train_mean is not None:
            # Compute batch statistics
            batch_mean = outputs.mean(dim=0)
            
            # Update history
            self._update_hist(batch_mean)
            
            # Compute shift vector
            shift_vector = self._get_shift_vector()
            
            # Apply shift if available
            if shift_vector is not None:
                outputs = outputs + shift_vector
        
        return outputs
    
    def _update_hist(self, batch_mean):
        """Update running history of test statistics."""
        if self.hist_mean is None:
            self.hist_mean = batch_mean.detach()
        else:
            self.hist_mean = self.alpha * self.hist_mean + (1 - self.alpha) * batch_mean.detach()
    
    def _get_shift_vector(self):
        """Calculate shift direction based on training vs test statistics."""
        if self.hist_mean is None or self.train_mean is None:
            return None
        return self.train_mean - self.hist_mean
    
    def obtain_origin_stat(self, train_loader, device='cuda'):
        """
        Compute training set feature statistics.
        
        Args:
            train_loader: DataLoader for training/calibration set
            device: Device to use
        """
        print('====> Computing training feature statistics for FOA (BP-free)')
        outputs_list = []
        
        with torch.no_grad():
            for batch_idx, (images, *_) in enumerate(train_loader):
                images = images.to(device)
                outputs = self.model(images)
                
                if self.imagenet_mask is not None:
                    outputs = outputs[:, self.imagenet_mask]
                
                outputs_list.append(outputs)
                
                # Limit samples for efficiency
                if batch_idx >= 10:
                    break
        
        all_outputs = torch.cat(outputs_list, dim=0)
        self.train_std = all_outputs.std(dim=0).detach()
        self.train_mean = all_outputs.mean(dim=0).detach()
        
        print(f'====> Collected statistics from {all_outputs.shape[0]} samples')
        del outputs_list, all_outputs
    
    def reset(self):
        """Reset model and adaptation state."""
        self.model.load_state_dict(self.model_state, strict=True)
        self.hist_mean = None


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
