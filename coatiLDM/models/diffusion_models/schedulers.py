import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

EPSILON = 1e-7
DEFAULT_BETA_START = 1e-4
DEFAULT_BETA_END = 0.02


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.99)


class DDPMScheduler(torch.nn.Module):
    def __init__(self, schedule, timesteps, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.diff = "ddpm"
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.schedule = schedule
        if self.schedule == "linear":
            self.register_buffer(
                "all_betas",
                torch.linspace(self.beta_start, self.beta_end, self.timesteps),
            )
        elif self.schedule == "cosine":
            self.register_buffer(
                "all_betas",
                torch.tensor(cosine_beta_schedule(self.timesteps), dtype=torch.float),
            )
        else:
            raise ValueError("unknown noise schedule type")
        self.register_buffer("all_alphas", 1.0 - self.all_betas)
        self.register_buffer(
            "all_bar_alphas", torch.cumprod(self.all_alphas, 0).clamp(0.0, 1.0)
        )

    def beta(self, T):
        """
        Exactly the beta schedule of Ho & Abeel.
         Args:
            T: torch. int tensor batch_size
        """
        beta = self.all_betas[T.long()]
        return beta.unsqueeze(-1)

    def alpha(self, T):

        return self.all_alphas[T.long()].unsqueeze(-1)

    def bar_alpha(self, T):

        return self.all_bar_alphas[T.long()].clamp(0.0, 1.0).unsqueeze(-1)

    def is_same(self, other):
        """
        Check if two instances of DDPMScheduler are functionally the same.
        Args:
            other: Another instance of DDPMScheduler.
        Returns:
            True if all values in all_betas are equal for both instances, False otherwise.
        """
        # Ensure that the other instance is of the same class
        if not isinstance(other, DDPMScheduler):
            raise ValueError(
                "Comparison is only supported between instances of DDPMScheduler."
            )

        # Check if all_betas are equal for both instances
        return torch.equal(self.all_betas, other.all_betas)
