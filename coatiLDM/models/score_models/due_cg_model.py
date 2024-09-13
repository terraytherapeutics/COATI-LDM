from typing import Dict, Any

import numpy as np
import torch
from due.dkl import DKL, GP, _get_initial_inducing_points, _get_initial_lengthscale
from due.fc_resnet import FCResNet
from gpytorch.likelihoods import GaussianLikelihood
from torch import nn
from coatiLDM.data.transforms import cg_xform_routine
from coatiLDM.models.score_models.score_model import get_time_embedding
import pickle
from smart_open import open
from coatiLDM.data.datapipe import get_base_pipe
import matplotlib.pyplot as plt


def initial_values(
    train_sample: torch.Tensor, feature_extractor: nn.Module, n_inducing_points: int
):
    with torch.no_grad():
        f_X_samples = feature_extractor(train_sample)

    initial_inducing_points = _get_initial_inducing_points(
        f_X_samples.numpy(), n_inducing_points
    )
    initial_lengthscale = _get_initial_lengthscale(f_X_samples)

    return initial_inducing_points, initial_lengthscale


class DueCG(nn.Module):
    def __init__(
        self,
        scheduler,
        time_embed_dim,
        train_data_sample: torch.Tensor = None,  # their implementation only uses 1k samples
        x_dim: int = 256,
        scalar_name: str = "logp",
        # features: int = 256,
        depth: int = 4,
        num_outputs: int = 1,
        spectral_normalization: bool = True,
        n_inducing_points=60,
        soft_norm_coeff: float = 0.95,
        n_power_iterations=2,
        dropout_rate=0.03,
        kernel="RBF",
    ) -> None:
        super().__init__()

        self.x_dim = x_dim
        self.scalar_name = scalar_name
        self.time_embed_dim = time_embed_dim
        self.input_dim = x_dim + time_embed_dim
        self.features = x_dim + time_embed_dim
        self.depth = depth
        self.num_outputs = num_outputs
        self.spectral_normalization = spectral_normalization
        self.n_inducing_points = n_inducing_points
        self.soft_norm_coeff = soft_norm_coeff
        self.n_power_iterations = n_power_iterations
        self.dropout_rate = dropout_rate
        self.kernel = kernel
        self.scheduler = scheduler
        self.initalized = False

        self.feature_extractor = FCResNet(
            input_dim=self.input_dim,
            features=self.features,
            depth=depth,
            spectral_normalization=spectral_normalization,
            coeff=soft_norm_coeff,
            n_power_iterations=n_power_iterations,
            dropout_rate=dropout_rate,
        )
        self.gp = None
        self.dkl = None
        self.n_inducing_points = n_inducing_points
        self.num_outputs = num_outputs
        self.kernel = kernel
        self.initalized = False
        self.likelihood = GaussianLikelihood()

        if train_data_sample is not None:
            self.initalize_model(train_data_sample["X"], train_data_sample=["T"])

    def noise_train_data_sample(self, train_data_sample):
        batch_size = train_data_sample.shape[0]
        T = torch.randint(
            low=0, high=self.scheduler.timesteps, size=(batch_size,), device=self.device
        )
        t_embed = get_time_embedding(T.float(), self.time_embed_dim)
        noise = torch.randn((batch_size, self.x_dim), device=self.device)
        noisy_samples = (
            self.scheduler.bar_alpha(T).sqrt() * train_data_sample
            + (1.0 - self.scheduler.bar_alpha(T)).sqrt() * noise
        )
        samp_with_noise = torch.cat([noisy_samples, t_embed], dim=1)
        return samp_with_noise

    def initalize_model(self, X: torch.Tensor, T: torch.Tensor):
        with torch.no_grad():
            t_embed = get_time_embedding(T.float(), self.time_embed_dim)
        train_data_sample = torch.cat([X, t_embed], dim=1)
        initial_inducing_points, initial_lengthscale = initial_values(
            train_data_sample, self.feature_extractor, self.n_inducing_points
        )
        self.gp = GP(
            num_outputs=self.num_outputs,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=self.kernel,
        )
        self.dkl = DKL(self.feature_extractor, self.gp)
        self.initalized = True

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, T):
        t_embed = get_time_embedding(T.float(), self.time_embed_dim)
        full_rep = torch.cat([x, t_embed], dim=1)
        return self.dkl(full_rep)


def save_due(params, model) -> bytes:
    scheduler_args = {
        "schedule": model.scheduler.schedule,
        "timesteps": model.scheduler.timesteps,
        "beta_start": model.scheduler.beta_start,
        "beta_end": model.scheduler.beta_end,
    }
    due_serialized = {
        "model_kwargs": {key: val for key, val in params.items() if key != "n_samples"},
        # "scheduler_kwargs": scheduler_args,
        "model": model.to("cpu").state_dict(),
    }
    return pickle.dumps(due_serialized)


def get_due_batch_pipe(
    pickle_path, due, x_field="emb_smiles", batch_size=2048, load_type="pickle"
):

    base_pipe = get_base_pipe(pickle_path, load_type)
    sched_bar_alphas = due.scheduler.all_bar_alphas.clone().detach().cpu()
    datapipe = (
        base_pipe.shuffle()
        .batch(batch_size)
        .collate(
            lambda batch: cg_xform_routine(
                batch,
                x_field=x_field,
                scalar_field=due.scalar_name,
                timesteps=due.scheduler.timesteps,
                bar_alphas=sched_bar_alphas,
            )
        )
    )

    return datapipe
