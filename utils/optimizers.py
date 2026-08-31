import torch
from torch import nn

from torch.optim import SGD, RMSprop, Adam
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

"""
from https://github.com/taeckyung/SoTTA/blob/main/utils/sam_optimizer.py 
"""


def sam_collect_params(model, freeze_top=False):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if freeze_top:
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


"""
from https://github.com/davda54/sam
"""


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p)) * p.grad).norm(p=2)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def make(name, params, lr, weight_decay=0.,
         schedule='step', milestones=None, gamma=0.1):
    """
    Prepares an optimizer and its learning-rate scheduler.

    Args:
      name (str): name of the optimizer. Options: 'sgd', 'rmsprop', 'adam'
      params (iterable): parameters to optimize.
      lr (float): initial learning rate.
      weight_decay (float, optional): weight decay. Default: 0.
      schedule (str, optional): type of learning-rate schedule. Default: 'step'
        Options: 'step', 'cosine'
        (This argument is ignored if milestones=None.)
      milestones (int list, optional): a list of epoches when learning rate
        is altered. Default: None
      gamma (float, optional): multiplicative factor of learning rate decay.
        Default: 0.1
    """
    if name == 'sgd':
        optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'rmsprop':
        optimizer = RMSprop(params, lr, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = Adam(params, lr, weight_decay=weight_decay)
    else:
        raise ValueError('invalid optimizer')

    if milestones is not None:
        if schedule == 'step':
            lr_scheduler = MultiStepLR(optimizer, milestones, gamma)
        elif schedule == 'cosine':
            lr_scheduler = CosineAnnealingLR(optimizer, milestones[-1])
    else:
        lr_scheduler = None

    return optimizer, lr_scheduler


def load(ckpt, params):
    train = ckpt['training']
    optimizer, lr_scheduler = make(
        train['optimizer'], params, **train['optimizer_args'])
    optimizer.load_state_dict(train['optimizer_state_dict'])

    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(train['lr_scheduler_state_dict'])

    return optimizer, lr_scheduler
