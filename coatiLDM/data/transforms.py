import torch
import numpy as np
from scipy.interpolate import PchipInterpolator


class cdf:
    # broadcasted/vectorized version of the CDF computation.
    # scales much better than using np.grid.
    def __init__(self, x, npts=800, max_bin=None, min_bin=None, verbose=False):

        if verbose:
            print(
                f"Computing empirical CDF from {x.shape[0]} points on {npts} grid points..."
            )

        X = np.nan_to_num(x)
        if min_bin is None:
            min_bin = X.min()
        if max_bin is None:
            max_bin = X.max()
        left_barrier = min_bin - (max_bin - min_bin) / 10.0
        right_barrier = max_bin + (max_bin - min_bin) / 10.0
        self.grid = np.linspace(left_barrier, right_barrier, npts)
        self.count = (X[:, None] < self.grid).sum(0)
        self.scdf = self.count / self.count[-1] + np.linspace(
            0, 1e-9, self.count.shape[0]
        )
        if verbose:
            print("done! Computing function approximations....")
        # compute spline approximations to cdfs
        self.invcdf = PchipInterpolator(self.scdf, self.grid)
        self.cdf = PchipInterpolator(self.grid, self.scdf)
        # compute smoothed cdf, useful for computing pdf
        w = self.grid.shape[0] // 200
        self.smoothed_scdf = np.convolve(
            self.scdf, np.hamming(w) / np.sum(np.hamming(w)), mode="same"
        )
        self.smoothed_scdf[-w:] = 1.0
        self.smoothed_cdf = PchipInterpolator(self.grid, self.smoothed_scdf)

        if verbose:
            print("done!")

    def pdf(self, pts, smooth=True):
        """
        The pdf of the cdf
        """
        cdf_approx = self.smoothed_cdf if smooth else self.cdf
        return cdf_approx.derivative()(pts)

    def to_unit_interval(self, pts):
        return self.cdf(pts)

    def quantile_class_boundaries(
        self, bounds=np.array([0.1, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999])
    ):
        return self.invcdf(bounds)

    def sample(self, n_sample=4000):
        ent = np.random.random(n_sample)
        return self.invcdf(ent)


def embed_scalar(
    timesteps,
    embedding_dim: int = 16,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
    max_time=1.0,
):
    # Adapted from tensor2tensor and VDM codebase.
    assert timesteps.ndim == 1
    assert embedding_dim % 2 == 0
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


def safe_embed_scalar(
    timesteps,
    embedding_dim: int = 16,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
    max_time=1.0,
):
    # Adapted from tensor2tensor and VDM codebase.
    assert timesteps.ndim == 1
    assert embedding_dim % 2 == 0
    transformed_timesteps = timesteps * (
        1000.0 / max_time
    )  # In DDPM the time step is in [0, 1000], in BFN [0, 1]
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(  # or exp(-linspace(log(min), log(max), n))
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        device=timesteps.device,
    )
    emb = transformed_timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # (T, D/2)
    return torch.cat([emb.sin(), emb.cos()], dim=1)  # (T, D)


def cg_xform_routine(
    batch,
    x_field="normd_vector",
    scalar_field="normd_logp",
    bar_alphas=None,
    timesteps=None,
    device=torch.device("cpu"),
    no_noise=False,
):
    batch_size = len(batch)
    assert batch_size > 0
    T = torch.randint(low=0, high=timesteps, size=(batch_size,), device=device)
    stacked = {}
    unnoised_sample = torch.tensor(
        np.stack([row[x_field] for row in batch], 0), device=device, dtype=torch.float
    )
    noise = torch.randn((batch_size, unnoised_sample.shape[-1]), device=device)
    b_alph_reshape = bar_alphas[T.long()].clamp(0.0, 1.0).unsqueeze(-1)
    noisy_samples = (
        b_alph_reshape.sqrt() * unnoised_sample + (1.0 - b_alph_reshape).sqrt() * noise
    )
    stacked["unnoised_samples"] = unnoised_sample
    stacked["noised_samples"] = noisy_samples
    stacked["T"] = T
    if no_noise:
        stacked["noised_samples"] = unnoised_sample
        stacked["T"] = torch.zeros_like(T)
    C = torch.tensor(
        [row[scalar_field] for row in batch], device=device, dtype=torch.float
    )
    # make uniform
    stacked["target"] = C
    return stacked


def xform_basic(
    batch,
    x_field="emb_smiles",
    scalar_cond_fields=["logp"],
    cond_emb_dim=16,
    device=torch.device("cpu"),
):
    """
    Stacks and vector embeds. assumes no normalization.
    """
    batch_size = len(batch)
    assert batch_size > 0
    stacked = {}
    stacked["samples"] = torch.tensor(
        np.stack([row[x_field] for row in batch], 0), device=device, dtype=torch.float
    )
    cond_vectors = []
    if len(scalar_cond_fields):
        for c in scalar_cond_fields:
            if c in batch[0]:
                stacked[c] = torch.tensor(
                    [row[c] for row in batch], device=device, dtype=torch.float
                )
            C = torch.tensor(
                [row[c] for row in batch], device=device, dtype=torch.float
            )
            cond_vectors.append(embed_scalar(C, embedding_dim=cond_emb_dim))
    if len(cond_vectors):
        cond_vectors = torch.cat(cond_vectors, -1)
        stacked["cond_vector"] = cond_vectors
    else:
        stacked["cond_vector"] = None
    return stacked
