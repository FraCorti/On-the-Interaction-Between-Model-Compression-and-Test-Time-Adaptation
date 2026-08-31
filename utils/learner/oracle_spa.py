"""
Supervised Oracle for SPA: identical augmentation pipeline (BYOLWrapper,
frequency masking, adversarial patch erasing) and optimizer as SPA; the
self-bootstrapping KL teacher is replaced by cross-entropy on the target
labels for each augmented view.
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from .learner import register

from ..maskers import LearnablePatchErasing

__all__ = ['oracle_spa']


@register('oracle_spa')
def oracle_spa(model, optimizer, steps=1, episodic=False):
    return OracleSPA(model=model, optimizer=optimizer, steps=steps, episodic=episodic)


class OracleSPA(nn.Module):
    """Supervised SPA.

    Loss: CE(predictor(model(freq_masked_x)), y) + CE(predictor(model(patch_erased_x)), y).
    The clean-view cross-entropy is detached, matching SPA's zero-gradient clean path.
    SPA's confidence-based sample filtering is not applied since labels are available.
    """

    def __init__(self, model, optimizer, noise_ratio=0.4, freq_mask_ratio=0.2,
                 steps=1, episodic=False, imagenet_mask=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "Oracle-SPA requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        # Augmentation pipeline, identical to SPA.
        self.prompts = Prompt(prompt_alpha=0.2).cuda()

        self.learnable_patch_erasing = LearnablePatchErasing(patch_size=16).cuda()
        self.noise_ratio = noise_ratio
        self.patch_optimizer = torch.optim.SGD([self.learnable_patch_erasing.alpha], 1, momentum=0.9)
        self.keep_prob = 1 - freq_mask_ratio
        self.imagenet_mask = imagenet_mask

    def forward(self, x, y=None):
        """Forward with supervised adaptation via SPA's augmentation pipeline.

        Args:
            x: input tensor [B, C, H, W]
            y: ground-truth labels [B]. If None, just forward (no adaptation).
        """
        if self.episodic:
            self.reset()

        if y is not None:
            for _ in range(self.steps):
                outputs = self.forward_and_adapt_supervised(x, y)
        else:
            with torch.no_grad():
                outputs = self.model(x)

        return outputs

    @torch.enable_grad()
    def forward_and_adapt_supervised(self, x, y):
        """One supervised step through SPA's augmentation pipeline."""
        # Clean forward in eval mode, as in SPA.
        self.model.eval()
        outputs = self.model(x)
        self.model.train()

        if self.imagenet_mask is not None:
            outputs = outputs[:, self.imagenet_mask]

        # Clean outputs are detached: SPA's clean path carries no gradient.
        loss = F.cross_entropy(outputs.detach(), y)

        # Frequency-masked view.
        noise_x = self.prompts.masking(x, keep_prob=self.keep_prob)
        low_outputs_freq = self.model(noise_x, use_predictor=True)
        if self.imagenet_mask is not None:
            low_outputs_freq = low_outputs_freq[:, self.imagenet_mask]
        loss += F.cross_entropy(low_outputs_freq, y)

        # Patch-erased view.
        noise_x = self.learnable_patch_erasing(x)
        low_outputs_patch = self.model(noise_x, use_predictor=True)
        if self.imagenet_mask is not None:
            low_outputs_patch = low_outputs_patch[:, self.imagenet_mask]
        loss += F.cross_entropy(low_outputs_patch, y)

        # Gradient negation makes the patch erasing adversarial, as in SPA.
        loss.backward()
        self.learnable_patch_erasing.alpha.grad = -self.learnable_patch_erasing.alpha.grad

        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.patch_optimizer.step()
        self.patch_optimizer.zero_grad()
        self.fix_patch_alpha()
        return outputs

    def predict(self, feats):
        return self.model(feats)

    @torch.no_grad()
    def fix_patch_alpha(self):
        """Identical to SPA.fix_patch_alpha."""
        current_mean = (self.learnable_patch_erasing.alpha ** 2).mean()
        div = current_mean / self.noise_ratio
        self.learnable_patch_erasing.alpha /= torch.sqrt(div)
        torch.clip_(self.learnable_patch_erasing.alpha, -1, 1)

    def reset(self):
        if self.model_state is None:
            raise Exception("cannot reset without saved model state")
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)
        self.learnable_patch_erasing.alpha.data = torch.ones_like(self.learnable_patch_erasing.alpha) * math.sqrt(
            self.noise_ratio)
        self.patch_optimizer = torch.optim.SGD([self.learnable_patch_erasing.alpha], 1, momentum=0.9)


# Utilities shared with spa.py.

class Prompt(nn.Module):
    """Frequency masking prompt, identical to SPA's Prompt class."""
    def __init__(self, prompt_alpha=0.5, num_range=200):
        super().__init__()
        imageH, imageW = 224, 224
        promptH = int(imageH * prompt_alpha) if int(imageH * prompt_alpha) > 1 else 1
        promptW = int(imageW * prompt_alpha) if int(imageW * prompt_alpha) > 1 else 1
        paddingH = (imageH - promptH) // 2
        paddingW = (imageW - promptW) // 2
        self.source_frequency, self.target_frequency = None, None

        self.init_para = torch.ones((1, 3, promptH, promptW)).cuda()
        self.low_mask = F.pad(self.init_para, [imageH - paddingH - promptH, paddingH,
                                               imageW - paddingW - promptW, paddingW],
                              mode='constant', value=0.).contiguous()
        self.high_mask = 1 - self.low_mask

    def iFFT(self, amp_src_, pha_src, imgH, imgW):
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)
        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
        return src_in_trg

    def masking(self, x, keep_prob=0.9):
        _, _, imgH, imgW = x.size()
        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src)
        amp_src = self._random_mask_on_low_frequency_only(amp_src, keep_prob)
        amp_src = torch.fft.ifftshift(amp_src)
        x = self.iFFT(amp_src, pha_src, imgH, imgW)
        return x

    def _random_mask_on_low_frequency_only(self, amp_src, keep_prob):
        mask = self.low_mask * torch.empty_like(amp_src).bernoulli_(keep_prob)
        mask = mask + self.high_mask
        mask = self.__convert_symmertry_mask(mask)
        amp_src = amp_src * mask
        return amp_src

    def __convert_symmertry_mask(self, mask):
        startH = 1 - (mask.shape[-2] % 2)
        startW = 1 - (mask.shape[-1] % 2)
        sub_mask = mask[:, :, startH:, startW:]
        assert (sub_mask.shape[-1] % 2 == 1 and sub_mask.shape[-2] % 2 == 1)
        maskH, maskW = sub_mask.shape[-2] // 2, sub_mask.shape[-1] // 2
        sub_mask[:, :, -maskH:, :] = 1
        sub_mask[:, :, maskH, :maskW] = 1
        sub_mask = sub_mask * torch.flip(sub_mask, dims=[-1, -2])
        mask[:, :, startH:, startW:] = sub_mask
        return mask


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    if optimizer is not None:
        optimizer_state = deepcopy(optimizer.state_dict())
    else:
        optimizer_state = None
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
