#
# A non-convolutional u-net.
#
import numpy as np
import torch
from torch import nn
import math

from coatiLDM.models.score_models.score_model import get_time_embedding, SwiGLUNet

import torch
import torch.nn as nn


class BottleNeck(nn.Module):
    def __init__(
        self,
        in_dim=256,
        out_dim=256,
        const_dim=256,
        dropout=0.0,
        use_weight_norm=False,
        bias=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.const_dim = const_dim
        self.use_weight_norm = use_weight_norm
        self.wedge_in = SwiGLUNet(
            self.in_dim + self.const_dim,
            self.out_dim,
            dropout=dropout,
            use_weight_norm=use_weight_norm,
            bias=bias,
        )
        self.wedge_out = SwiGLUNet(
            self.out_dim + self.const_dim,
            self.in_dim,
            dropout=dropout,
            use_weight_norm=use_weight_norm,
            bias=bias,
        )

    def forward(self, x, const):
        """
        Downsamples and residuals like a U-Net.
        """
        tocat = [const]
        z = self.wedge_in(torch.cat([x, *tocat], -1))
        z2 = self.wedge_out(torch.cat([torch.nn.functional.silu(z), *tocat], -1))
        out_ = x + z2
        return out_, z


class Flat(nn.Module):
    def __init__(
        self, in_dim=256, const_dim=256, dropout=0.0, use_weight_norm=False, bias=True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.const_dim = const_dim
        self.wedge_in = SwiGLUNet(
            self.in_dim + self.const_dim,
            self.in_dim,
            dropout=dropout,
            use_weight_norm=use_weight_norm,
            bias=bias,
        )
        self.wedge_out = SwiGLUNet(
            self.in_dim + self.const_dim,
            self.in_dim,
            dropout=dropout,
            use_weight_norm=use_weight_norm,
            bias=bias,
        )
        self.use_weight_norm = use_weight_norm

    def forward(self, x, const):
        tocat = [const]
        z = self.wedge_in(torch.cat([x, *tocat], -1))
        z2 = self.wedge_out(torch.cat([x + torch.nn.functional.silu(z), *tocat], -1))
        out = x + z2
        return out


class Up(nn.Module):
    def __init__(
        self,
        in_dim=256,
        out_dim=256,
        const_dim=256,
        dropout=0.0,
        use_weight_norm=False,
        bias=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.const_dim = const_dim
        self.wedge_in = SwiGLUNet(
            self.in_dim + self.const_dim,
            self.out_dim,
            dropout=dropout,
            use_weight_norm=use_weight_norm,
            bias=bias,
        )
        self.wedge_out = SwiGLUNet(
            self.out_dim + self.const_dim,
            self.out_dim,
            dropout=dropout,
            use_weight_norm=use_weight_norm,
            bias=bias,
        )
        self.use_weight_norm = use_weight_norm

    def forward(self, x, const):
        tocat = [const]
        z = self.wedge_in(torch.cat([x, *tocat], -1))
        z2 = self.wedge_out(torch.cat([torch.nn.functional.silu(z), *tocat], -1))
        out = z + z2
        return out


class OU(nn.Module):
    def __init__(
        self,
        x_dim=256,
        cond_dim=64,
        time_dim=64,
    ):
        super().__init__()
        # Allow learning of OU-like nearly constant DOFs.
        # score(X) \propto mu - X
        self.ou_const = nn.Parameter(torch.zeros(x_dim).normal_())
        self.ou_dec = SwiGLUNet(
            time_dim + cond_dim,
            x_dim,
            residual=False,
            dropout=False,
            use_weight_norm=False,
            bias=True,
        )

    def forward(self, x, t, cond=None):
        tocat = [t]
        if (not cond is None) and len(cond):
            tocat.append(cond)
        const = torch.cat(tocat, -1)
        fac = torch.nn.functional.softplus(self.ou_dec(const))
        return (x - self.ou_const.unsqueeze(0)) * fac


class NonConvUNet(nn.Module):
    def __init__(
        self,
        x_dim=256,
        cond_dim=1,
        time_max=1.0,
        time_dim=None,
        dropout=0.0,
        use_weight_norm=False,
        scheduler=None,
        bias=True,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.cond_dim = cond_dim
        self.time_max = time_max
        self.time_dim = time_dim
        self.use_weight_norm = use_weight_norm
        self.scheduler = scheduler

        self.ou = OU(x_dim=x_dim, time_dim=time_dim, cond_dim=cond_dim)

        # bias is just set to true for the bottleneck layers. This aligned with whatever was going on in John branch pre-merge.
        self.steps_down = nn.ModuleList(
            [
                BottleNeck(
                    x_dim,
                    x_dim // 2,
                    time_dim + cond_dim,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                    bias=True,
                ),
                BottleNeck(
                    x_dim // 2,
                    x_dim // 4,
                    time_dim + cond_dim,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                    bias=True,
                ),
                BottleNeck(
                    x_dim // 4,
                    x_dim // 8,
                    time_dim + cond_dim,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                    bias=True,
                ),
            ]
        )

        self.flat = nn.ModuleList(
            [
                Flat(
                    x_dim // 8,
                    time_dim + cond_dim,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                    bias=bias,
                ),
                Flat(
                    x_dim,
                    time_dim + cond_dim,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                    bias=bias,
                ),
            ]
        )

        self.steps_up = nn.ModuleList(
            [
                Up(
                    x_dim // 8,
                    x_dim // 4,
                    time_dim + cond_dim,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                    bias=bias,
                ),
                Up(
                    x_dim // 4,
                    x_dim // 2,
                    time_dim + cond_dim,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                    bias=bias,
                ),
                Up(
                    x_dim // 2,
                    x_dim,
                    time_dim + cond_dim,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                    bias=bias,
                ),
            ]
        )

    def forward(self, x, t, cond=None):
        time = get_time_embedding(
            t, max_time=self.time_max, embedding_dim=self.time_dim
        )
        tocat = [time]
        if (not cond is None) and len(cond):
            tocat.append(cond)
        const = torch.cat(tocat, -1)

        z0 = self.ou(x, time, cond=cond)

        x0, d0 = self.steps_down[0](x, const)
        x1, d1 = self.steps_down[1](d0, const)
        x2, d2 = self.steps_down[2](d1, const)

        x3 = self.flat[0](d2, const)

        y2 = self.steps_up[0](x3, const)
        y1 = self.steps_up[1](y2 + x2, const)
        y0 = self.steps_up[2](x1 + y1, const)

        return self.flat[1](x0 + y0 + z0, const)
