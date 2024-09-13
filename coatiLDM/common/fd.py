from scipy import linalg
from coatiLDM.models.diffusion_models import ddpm_sample_routines
from coatiLDM.models.diffusion_models.dflow import dflow, dflow_multi
import numpy as np
import torch


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all() or not np.allclose(
        np.diagonal(covmean).imag, 0, atol=1e-3
    ):
        msg = (
            "fd calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calc_fd(s1: np.ndarray, s2: np.ndarray):
    """
    Automatically checks the convergence of the FID
    WRT samples for you... by calculating it over 4 samples.

    Args:
        s1: np samples in rows.
        s2: same
    """
    print(f"S1: {s1.shape}")
    print(f"S2: {s2.shape}")
    sample_shape = np.min([s1.shape[0], s2.shape[0]])
    n_samples = sample_shape // 4
    mu1 = s1[:n_samples].mean(0)
    mu2 = s2[:n_samples].mean(0)
    sigma1 = np.cov(s1[:n_samples], rowvar=False)
    sigma2 = np.cov(s2[:n_samples], rowvar=False)
    FID1 = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    n_samples = sample_shape // 2
    mu1 = s1[:n_samples].mean(0)
    mu2 = s2[:n_samples].mean(0)
    sigma1 = np.cov(s1[:n_samples], rowvar=False)
    sigma2 = np.cov(s2[:n_samples], rowvar=False)
    FID2 = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    n_samples = 3 * sample_shape // 4
    mu1 = s1[:n_samples].mean(0)
    mu2 = s2[:n_samples].mean(0)
    sigma1 = np.cov(s1[:n_samples], rowvar=False)
    sigma2 = np.cov(s2[:n_samples], rowvar=False)
    FID3 = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    n_samples = sample_shape
    mu1 = s1[:n_samples].mean(0)
    mu2 = s2[:n_samples].mean(0)
    sigma1 = np.cov(s1[:n_samples], rowvar=False)
    sigma2 = np.cov(s2[:n_samples], rowvar=False)
    FID4 = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(
        f"FID at 25\%{FID1:.3f} 50\%{FID2:.3f} 75\%{FID3:.3f} 100\%{FID4:.3f} of {n_samples} samples"
    )
    return FID4


def run_fd(
    target_embs, score_model, n_samples=20_000, samples_per_batch=256, pg_weight=0.0
):
    print(
        f"Running FID with {n_samples} samples against {target_embs.shape[0]} target embeddings"
    )
    samples = torch.zeros(n_samples, score_model.x_dim)
    idx = 0
    while idx < n_samples:
        batch_size = min([samples_per_batch, n_samples - idx])
        sample_batch = (
            ddpm_sample_routines.ddpm_basic_sample(
                score_model, cond=None, batch_size=batch_size, pg_weight=pg_weight
            )
            .detach()
            .cpu()
        )
        samples[idx : idx + batch_size] = sample_batch
        idx = idx + samples_per_batch
    fd = calc_fd(target_embs, samples.numpy())
    return fd


def run_fd_cond(
    target_embs,
    score_model,
    cond_set,
    n_samples=20_000,
    samples_per_batch=256,
    pg_weight=0.0,
):
    print(
        f"Running FID with {n_samples} samples against {target_embs.shape[0]} target embeddings"
    )
    samples = torch.zeros(n_samples, score_model.x_dim)
    idx = 0
    while idx < n_samples:
        batch_size = min([samples_per_batch, n_samples - idx])
        sample_batch = (
            ddpm_sample_routines.ddpm_basic_sample(
                score_model,
                cond_set[idx : idx + batch_size],
                batch_size=batch_size,
                pg_weight=pg_weight,
            )
            .detach()
            .cpu()
        )
        samples[idx : idx + batch_size] = sample_batch
        idx = idx + samples_per_batch
    fd = calc_fd(target_embs, samples.numpy())
    return fd


def run_fd_cg(
    target_embs,
    score_model,
    cond_set,
    cg_regressor,
    cg_weight,
    n_samples=20_000,
    samples_per_batch=256,
    pg_weight=0.0,
):
    print(
        f"Running FID with {n_samples} samples against {target_embs.shape[0]} target embeddings"
    )
    samples = torch.zeros(n_samples, score_model.x_dim)
    idx = 0
    while idx < n_samples:
        batch_size = min([samples_per_batch, n_samples - idx])
        sample_batch = (
            ddpm_sample_routines.ddpm_sample_classifier_guidance(
                score_net=score_model,
                batch_size=batch_size,
                cg_weight=cg_weight,
                cg_due=cg_regressor,
                cg_targets=cond_set[idx : idx + batch_size],
                pg_weight=pg_weight,
            )
            .detach()
            .cpu()
        )
        samples[idx : idx + batch_size] = sample_batch
        idx = idx + samples_per_batch
    fd = calc_fd(target_embs, samples.numpy())
    return fd


def run_fd_flow(
    target_embs, flow_model, cond_set, n_samples=20_000, samples_per_batch=256
):
    print(
        f"Running FID with {n_samples} samples against {target_embs.shape[0]} target embeddings"
    )
    samples = torch.zeros(n_samples, flow_model.score_net.x_dim)
    idx = 0
    while idx < n_samples:
        batch_size = min([samples_per_batch, n_samples - idx])
        with torch.no_grad():
            x_0 = torch.randn(
                batch_size, 512, device=next(flow_model.score_net.parameters()).device
            )
            if cond_set is None:
                sample_batch = flow_model.decode(x_0, cs=None).detach().cpu()
            else:
                sample_batch = (
                    flow_model.decode(x_0, cs=cond_set[idx : idx + batch_size])
                    .detach()
                    .cpu()
                )
        samples[idx : idx + batch_size] = sample_batch
        idx = idx + samples_per_batch
    fd = calc_fd(target_embs, samples.numpy())
    return fd


from coatiLDM.models.score_models.flow_wrapper import ODEWrapper


def run_fd_dflow(
    target_embs,
    flow_net,
    cond_set,
    cond_regressor,
    ode_steps=200,
    opt_steps=2,
    n_samples=20_000,
    samples_per_batch=1000,
):
    print(
        f"Running FID with {n_samples} samples against {target_embs.shape[0]} target embeddings"
    )
    assert (
        next(flow_net.parameters()).device == next(cond_regressor.parameters()).device
    )
    assert isinstance(flow_net, ODEWrapper)
    samples = torch.zeros(n_samples, flow_net.score_net.x_dim)
    idx = 0
    while idx < n_samples:
        batch_size = min([samples_per_batch, n_samples - idx])

        x_0 = torch.randn(batch_size, 512, device=next(flow_net.parameters()).device)
        sample_batch = dflow(
            x_0,
            cond_set[idx : idx + batch_size],
            flow_net,
            cond_regressor,
            learning_rate=1.0,
            decode_steps=ode_steps,
            opt_steps=opt_steps,
            device=next(flow_net.parameters()).device,
        ).cpu()[-1]

        samples[idx : idx + batch_size] = sample_batch
        idx = idx + samples_per_batch
    fd = calc_fd(target_embs, samples.numpy())
    return fd


def run_fd_dflow_multi(
    target_embs,
    flow_net,
    cond_sets,
    cond_regressors,
    ode_steps=200,
    opt_steps=2,
    n_samples=20_000,
    samples_per_batch=1000,
):
    print(
        f"Running FID with {n_samples} samples against {target_embs.shape[0]} target embeddings"
    )
    assert (
        next(flow_net.parameters()).device
        == next(cond_regressors[0].parameters()).device
    )
    assert isinstance(flow_net, ODEWrapper)
    device = next(flow_net.parameters()).device
    samples = torch.zeros(n_samples, flow_net.score_net.x_dim)
    idx = 0
    while idx < n_samples:
        batch_size = min([samples_per_batch, n_samples - idx])

        x_0 = torch.randn(batch_size, 512, device=device)
        sample_batch = dflow_multi(
            x_0,
            [cond_set[idx : idx + batch_size] for cond_set in cond_sets],
            flow_net,
            cond_regressors,
            learning_rate=1.0,
            decode_steps=ode_steps,
            opt_steps=opt_steps,
            device=next(flow_net.parameters()).device,
        ).cpu()[-1]

        samples[idx : idx + batch_size] = sample_batch
        idx = idx + samples_per_batch
    fd = calc_fd(target_embs, samples.numpy())
    return fd


def run_fd_cg_multi(
    target_embs,
    score_model,
    cond_sets,
    cg_regressors,
    cg_weights,
    n_samples=20_000,
    samples_per_batch=256,
    pg_weight=0.0,
):
    print(
        f"Running FID with {n_samples} samples against {target_embs.shape[0]} target embeddings"
    )
    samples = torch.zeros(n_samples, score_model.x_dim)
    idx = 0
    while idx < n_samples:
        batch_size = min([samples_per_batch, n_samples - idx])
        cond_set_subsets = [x[idx : idx + batch_size] for x in cond_sets]
        sample_batch = (
            ddpm_sample_routines.ddpm_sample_multi_classifier_guidance(
                score_net=score_model,
                batch_size=batch_size,
                cg_weights=cg_weights,
                cg_dues=cg_regressors,
                cg_targets=cond_set_subsets,
                pg_weight=pg_weight,
            )
            .detach()
            .cpu()
        )
        samples[idx : idx + batch_size] = sample_batch
        idx = idx + samples_per_batch
        torch.cuda.empty_cache()
    fd = calc_fd(target_embs, samples.numpy())
    return fd


def run_fd_cfg(
    target_embs,
    uncond_score_model,
    cond_score_model,
    cond_set,
    cfg_weight,
    n_samples=20_000,
    samples_per_batch=256,
    pg_weight=0.0,
):
    print(
        f"Running FID with {n_samples} samples against {target_embs.shape[0]} target embeddings"
    )
    samples = torch.zeros(n_samples, uncond_score_model.x_dim)
    idx = 0
    while idx < n_samples:
        batch_size = min([samples_per_batch, n_samples - idx])
        sample_batch = (
            ddpm_sample_routines.ddpm_sample_classifier_free_guidance(
                uncond_score_net=uncond_score_model,
                cond_score_net=cond_score_model,
                cond=cond_set[idx : idx + batch_size],
                batch_size=batch_size,
                pg_weight=pg_weight,
                cfg_weight=cfg_weight,
            )
            .detach()
            .cpu()
        )
        samples[idx : idx + batch_size] = sample_batch
        idx = idx + samples_per_batch
    fd = calc_fd(target_embs, samples.numpy())
    return fd
