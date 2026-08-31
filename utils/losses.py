import torch
import torch.nn as nn
import torch.nn.functional as F


class SAR_EntropyLoss(nn.Module):
    """
    Computes the Reliable Entropy Loss for SAR analysis.
    Only computes entropy for samples that are 'reliable' (low entropy)
    under the current model state.
    """

    def __init__(self, margin_e0, fallback_on_empty=True):
        super().__init__()
        # Default margin from SAR code: 0.4 * ln(1000) ≈ 2.76 for ImageNet (1000 classes)
        self.margin_e0 = margin_e0
        self.fallback_on_empty = fallback_on_empty

    def forward(self, logits, y=None):
        """
        Args:
            logits: Model output
            y: Ignored (unsupervised)
        """
        # Standard entropy
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=1)

        # Reliable-sample mask, computed dynamically as in forward_and_adapt_sar
        reliable_mask = entropy < self.margin_e0

        # No reliable samples
        if reliable_mask.sum() == 0:
            if self.fallback_on_empty:
                # Fallback to TENT (Standard Entropy Minimization)
                return entropy.mean()
            else:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Mean entropy over reliable samples
        reliable_loss = entropy[reliable_mask].mean()

        return reliable_loss


class SPAConsistencyLoss(torch.nn.Module):
    """
    Computes the SPA Consistency Objective.
    Can optionally bypass the linear predictor to measure intrinsic backbone consistency.
    """

    requires_model_x_output = True

    def __init__(self, spa_wrapper, bypass_predictor=False):
        super().__init__()
        self.prompts = spa_wrapper.prompts
        self.patch_erasing = spa_wrapper.learnable_patch_erasing
        self.imagenet_mask = spa_wrapper.imagenet_mask
        self.keep_prob = spa_wrapper.keep_prob
        self.criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.bypass_predictor = bypass_predictor

    def forward(self, model, x, clean_logits):
        loss = 0.0
        # Frequency masking consistency
        noise_x_freq = self.prompts.masking(x, keep_prob=self.keep_prob)
        loss += self._compute_consistency(model, noise_x_freq, clean_logits)

        # Learnable patch erasing consistency
        noise_x_patch = self.patch_erasing(x)
        loss += self._compute_consistency(model, noise_x_patch, clean_logits)
        return loss

    def _compute_consistency(self, model, augmented_x, clean_logits):
        # Determine if we use the predictor head or standard forward
        # Pre-TTA: bypass_predictor=True (Avoid random head noise)
        # Post-TTA: bypass_predictor=False (Measure actual TTA objective)
        if self.bypass_predictor:
            aug_logits = model(augmented_x)
        elif "use_predictor" in model.forward.__code__.co_varnames:
            aug_logits = model(augmented_x, use_predictor=True)
        else:
            # Fallback if model doesn't support predictor
            aug_logits = model(augmented_x)

        if self.imagenet_mask is not None:
            aug_logits = aug_logits[:, self.imagenet_mask]
            if clean_logits.shape[1] != aug_logits.shape[1]:
                clean_logits_masked = clean_logits[:, self.imagenet_mask]
            else:
                clean_logits_masked = clean_logits
        else:
            clean_logits_masked = clean_logits

        clean_probs = clean_logits_masked.softmax(dim=-1)
        aug_probs = aug_logits.softmax(dim=-1)

        clean_conf, _ = clean_probs.max(dim=-1)
        aug_conf, _ = aug_probs.max(dim=-1)

        conf_mask = torch.where(clean_conf > aug_conf)

        if conf_mask[0].shape[0] == 0:
            return torch.tensor(0.0, device=clean_logits.device, requires_grad=True)

        return self.criterion_kl(
            aug_logits[conf_mask].log_softmax(dim=-1),
            clean_logits_masked[conf_mask].softmax(dim=-1).detach()
        )


class OracleSPALoss(torch.nn.Module):
    """Supervised counterpart of SPAConsistencyLoss for matched-oracle gradient alignment.

    Cross-entropy on the frequency-masked and patch-erased views, using the same
    Prompt and LearnablePatchErasing instances as the paired SPAConsistencyLoss so
    both losses see identical augmentation noise. The clean-view term of Oracle-SPA
    is omitted since it carries no gradient. Used as `oracle_loss_fn` in
    gradient_alignment.compute_gradient_alignment (`requires_model_x_y`).
    """

    requires_model_x_y = True

    def __init__(self, spa_wrapper, bypass_predictor=False):
        super().__init__()
        self.prompts = spa_wrapper.prompts
        self.patch_erasing = spa_wrapper.learnable_patch_erasing
        self.imagenet_mask = spa_wrapper.imagenet_mask
        self.keep_prob = spa_wrapper.keep_prob
        self.bypass_predictor = bypass_predictor

    def forward(self, model, x, y):
        noise_x_freq = self.prompts.masking(x, keep_prob=self.keep_prob)
        loss = self._compute_supervised(model, noise_x_freq, y)
        noise_x_patch = self.patch_erasing(x)
        loss = loss + self._compute_supervised(model, noise_x_patch, y)
        return loss

    def _compute_supervised(self, model, augmented_x, y):
        if self.bypass_predictor:
            aug_logits = model(augmented_x)
        elif "use_predictor" in model.forward.__code__.co_varnames:
            aug_logits = model(augmented_x, use_predictor=True)
        else:
            aug_logits = model(augmented_x)

        if self.imagenet_mask is not None:
            aug_logits = aug_logits[:, self.imagenet_mask]

        return F.cross_entropy(aug_logits, y)
