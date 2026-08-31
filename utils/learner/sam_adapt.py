import torch.nn as nn

from utils.learner.learner import register

__all__ = ['sam_adapt']


@register('sam_adapt')
def sam_adapt(model):
    return SAM_Adapt(model=model)


class SAM_Adapt:

    def __init__(self, model=None):
        self.net = model

        # turn on grad for BN params only
        for param in self.net.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False

        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

                # With below, this module always uses the test batch statistics (no momentum)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.InstanceNorm2d):
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.LayerNorm):
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

    def predict(self, feats):
        self.net.eval()
        return self.net(feats)

    def step(self, loss_fn, feats, optimizer=None):
        assert (feats is not None)

        self.net.train()


        preds_of_data_first = self.net(feats)

        loss_first = loss_fn(preds_of_data_first).mean(0)

        optimizer.zero_grad()

        loss_first.backward()

        # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
        optimizer.first_step(zero_grad=True)

        preds_of_data_second = self.net(feats)

        # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
        loss_second = loss_fn(preds_of_data_second).mean(0)

        loss_second.backward()
        optimizer.second_step(zero_grad=True)

