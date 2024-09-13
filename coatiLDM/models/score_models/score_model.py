import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * x


class SwiGLUNet(nn.Module):
    def __init__(
        self, d_in, d_out, residual=False, dropout=0.0, use_weight_norm=False, bias=True
    ):

        super().__init__()
        self.residual = residual
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            torch.nn.Dropout(p=dropout),
            # should this one be weight-normed as well? (vs just the second)
            (
                weight_norm(nn.Linear(d_in, 2 * d_out, bias=bias), dim=None)
                if use_weight_norm
                else nn.Linear(d_in, 2 * d_out, bias=bias)
            ),
            SwiGLU(),
            (
                weight_norm(nn.Linear(d_out, d_out, bias=bias), dim=None)
                if use_weight_norm
                else nn.Linear(d_out, d_out, bias=bias)
            ),
        )

    def forward(self, x):
        if self.residual:
            return self.net(x) + x
        else:
            return self.net(x)


def get_time_embedding(
    timesteps,
    embedding_dim: int,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
    max_time=1.0,
):
    # Adapted from tensor2tensor and VDM codebase.

    timesteps *= (
        1000.0 / max_time
    )  # In DDPM the time step is in [0, 1000], in BFN [0, 1]
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(  # or exp(-linspace(log(min), log(max), n))
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        device=timesteps.device,
    )
    emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # (T, D/2)
    return torch.cat([emb.sin(), emb.cos()], dim=1)  # (T, D)
