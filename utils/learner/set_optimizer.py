import torch

from utils.byol_wrapper import BYOLWrapper
from utils.learner import sar, tent, spa
from utils.optimizers import sam_collect_params, SAM
from utils.vpt import PromptViT


def get_adaptation_optimizer_configure_model(tta_method_name, model, learning_rate=0.001, args=None, device=None, cifar10=False):
    optimizer = None
    if tta_method_name == 'sar':
        net = sar.configure_model(model=model)
        params, _ = sam_collect_params(net, freeze_top=True)

        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, lr=learning_rate, momentum=0.9)
    elif tta_method_name == "tent":
        net = tent.configure_model(model=model, args=args)
        # TENT hyper-parameters: Adam for CIFAR10-C (https://github.com/DequanWang/tent),
        # SGD with momentum for ImageNet-C (https://github.com/mr-eggplant/SAR).
        if not cifar10:
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif tta_method_name == "spa":
        # Classifier head is `.head` (timm ViT), `.fc` (torchvision / CIFAR ResNet)
        # or `.linear` (older CIFAR ResNet). SPA's augmentations assume 224x224 inputs.
        if hasattr(model, 'head') and hasattr(model.head, 'in_features'):
            projector_dim = model.head.in_features
        elif hasattr(model, 'fc') and hasattr(model.fc, 'in_features'):
            projector_dim = model.fc.in_features
        elif hasattr(model, 'linear') and hasattr(model.linear, 'in_features'):
            projector_dim = model.linear.in_features
        else:
            raise ValueError(
                "SPA: model exposes no recognized classifier head with "
                "`in_features` (looked for `.head`, `.fc`, `.linear`)."
            )
        net = BYOLWrapper(model, projector_dim=projector_dim)
        net = net.to(device=device)
        net = spa.configure_model(net)
        params, param_names = spa.collect_params(net)
        optimizer = torch.optim.SGD([
            {'params': params, 'lr': 0.01, 'momentum': 0.9},
            {'params': net.predictor.parameters(), 'lr': 0.05, 'momentum': 0.9},
        ])
        return optimizer, net
    elif tta_method_name == "foa":
        # FOA (Niu et al., ICML 2024): https://github.com/mr-eggplant/FOA
        # Official hyper-parameters: 3 prompts, fitness_lambda 0.4 for ImageNet-C.
        from utils.learner.foa import FOA

        num_prompts = 3
        fitness_lambda = 0.4

        prompt_vit = PromptViT(vit=model, num_prompts=num_prompts).to(device)
        # FOA is BP-free; leaving parameters with requires_grad=True would make
        # autograd record an activation tape for each of the CMA-ES candidates.
        prompt_vit.requires_grad_(False)
        prompt_vit.eval()
        foa_model = FOA(model=prompt_vit, fitness_lambda=fitness_lambda)

        return None, foa_model
    elif tta_method_name == "no_adapt":
        from utils.learner.no_adapt import NoAdapt
        no_adapt_model = NoAdapt(model=model)
        return None, no_adapt_model
    elif tta_method_name == "oracle_tent":
        # Oracle-TENT: same optimizer and trainable parameters as TENT,
        # supervised cross-entropy instead of entropy.
        from utils.learner import oracle_tent as oracle_tent_mod
        net = oracle_tent_mod.configure_model(model=model, args=args)
        if not cifar10:
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        return optimizer, None
    elif tta_method_name == "oracle":
        # Oracle: same SAM optimizer and parameter set as SAR,
        # supervised cross-entropy instead of entropy.
        from utils.learner import oracle as oracle_mod
        net = oracle_mod.configure_model(model=model, args=args)
        params, _ = sam_collect_params(net, freeze_top=True)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, lr=learning_rate, momentum=0.9)
        return optimizer, None
    elif tta_method_name == "oracle_spa":
        # Oracle-SPA: same BYOLWrapper and optimizer groups as SPA; the
        # bootstrapping KL is replaced by supervised cross-entropy on all views.
        if hasattr(model, 'head') and hasattr(model.head, 'in_features'):
            projector_dim = model.head.in_features
        elif hasattr(model, 'fc') and hasattr(model.fc, 'in_features'):
            projector_dim = model.fc.in_features
        elif hasattr(model, 'linear') and hasattr(model.linear, 'in_features'):
            projector_dim = model.linear.in_features
        else:
            raise ValueError(
                "Oracle-SPA: model exposes no recognized classifier head."
            )
        net = BYOLWrapper(model, projector_dim=projector_dim)
        net = net.to(device=device)
        net = spa.configure_model(net)
        params, param_names = spa.collect_params(net)
        optimizer = torch.optim.SGD([
            {'params': params, 'lr': 0.01, 'momentum': 0.9},
            {'params': net.predictor.parameters(), 'lr': 0.05, 'momentum': 0.9},
        ])
        return optimizer, net
    elif tta_method_name == "norm":
        # NORM: BatchNorm statistics recalibration, https://github.com/DequanWang/tent
        from utils.learner.norm import Norm
        norm_model = Norm(model=model)
        return None, norm_model
    elif tta_method_name == "lame":
        # LAME (Boudiaf et al., CVPR 2022): https://github.com/fiveai/LAME
        # Returns probabilities rather than logits; argmax-based accuracy is unaffected.
        from utils.learner.lame import LAME
        lame_model = LAME(model=model)
        return None, lame_model
    elif tta_method_name in ("pea_resnet18", "pea_vit"):
        # PEA (Xiao et al., ICLR 2026): https://github.com/TheMaXiao/PEA_TTA
        # The caller must run `precompute_source_stats(train_loader, device)`
        # once per model before adaptation.
        from utils.learner.pea import PEAResNetWrapper, PEAViTWrapper
        if tta_method_name == "pea_resnet18":
            pea_model = PEAResNetWrapper(model=model)
        else:
            pea_model = PEAViTWrapper(model=model)
        return None, pea_model
    else:
        raise ValueError("Unknown TTA method: {}".format(tta_method_name))
    return optimizer, None

