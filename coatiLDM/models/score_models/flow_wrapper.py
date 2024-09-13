import torch
from torch import nn


class ODEWrapper(nn.Module):

    def __init__(self, score_net):
        super(ODEWrapper, self).__init__()
        self.score_net = score_net

    def forward(self, t, x):
        device = next(self.score_net.parameters()).device
        t = t * torch.ones(len(x), device=device)
        return self.score_net(x, t)
