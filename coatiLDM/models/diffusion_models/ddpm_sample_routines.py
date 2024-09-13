import torch
import numpy as np
from coatiLDM.models.diffusion_models.particle_guidance import (
    similarity_guidance_gradient,
    cosine_guidance_gradient,
    cosine_guidance_updated,
    low_memory_cosine_guidance_gradient,
)
from coatiLDM.data.transforms import embed_scalar
from tqdm.auto import tqdm
import numpy as np
from torchdiffeq import odeint


def ddpm_basic_sample(
    score_net,
    cond=None,
    batch_size=4,
    pg_weight=0.0,
    embed_dim=None,
    pg_gradient_type="euclidean",
    fixed_pg_ratio=False,
    low_mem=False,
):
    with torch.no_grad():
        device = next(score_net.parameters()).device
        if not cond is None:
            if cond is None and score_net.cond_dim > 0:
                raise Exception("Give me a condition.")
            else:
                assert cond.shape[0] == batch_size
        else:
            # Conditions are normally distributed random variables.
            cond = torch.randn((batch_size, score_net.cond_dim), device=device)

        if embed_dim:
            cond = embed_scalar(cond, embedding_dim=embed_dim)

        guidance_ratio = 0.1 * pg_weight

        x_t = torch.randn((batch_size, score_net.x_dim), device=device)
        for T_ in reversed(range(score_net.scheduler.timesteps)):
            T = torch.ones(batch_size, dtype=torch.long, device=device) * T_
            if T_ > 0:
                z = torch.randn((batch_size, score_net.x_dim), device=device)
            else:
                z = torch.zeros((batch_size, score_net.x_dim), device=device)
            extracted_noise = score_net(x_t, t=T.float(), cond=cond)
            if pg_weight != 0:
                if pg_gradient_type in ("cos", "cosine"):
                    if low_mem:
                        guidance = low_memory_cosine_guidance_gradient(x_t)
                    else:
                        guidance = cosine_guidance_updated(x_t)
                else:
                    guidance = similarity_guidance_gradient(x_t)
                if fixed_pg_ratio:
                    pg_multiplier = (
                        guidance_ratio * extracted_noise.abs().mean()
                    ) / guidance.abs().mean()
                else:
                    pg_multiplier = pg_weight
                extracted_noise = extracted_noise + (guidance * pg_multiplier)
            at = score_net.scheduler.alpha(T)
            barat = score_net.scheduler.bar_alpha(T)
            noise_factor = (1.0 - at) / ((1.0 - barat).sqrt())
            x_t = (
                torch.pow(score_net.scheduler.alpha(T), -0.5)
                * (x_t - noise_factor * extracted_noise)
                + score_net.scheduler.beta(T).sqrt() * z
            )
        return x_t.detach()


def ddpm_sample_classifier_free_guidance(
    uncond_score_net,
    cond_score_net,
    cond,
    batch_size=4,
    pg_weight=0.0,
    cfg_weight=0.2,
    pg_gradient_type="euclidean",
    fixed_pg_ratio="False",
):
    try:
        assert uncond_score_net.cond_dim == 0
    except:
        ValueError("Unconditional score net is first argument, conditional second")

    try:
        assert uncond_score_net.scheduler.is_same(cond_score_net.scheduler)
    except:
        ValueError("Score nets must have the same noise schedule for sampling")

    scheduler = uncond_score_net.scheduler

    guidance_ratio = 0.1 * pg_weight

    with torch.no_grad():
        device = next(uncond_score_net.parameters()).device
        assert device == next(cond_score_net.parameters()).device

        x_t = torch.randn((batch_size, uncond_score_net.x_dim), device=device)
        for T_ in reversed(range(scheduler.timesteps)):
            T = torch.ones(batch_size, dtype=torch.long, device=device) * T_
            if T_ > 0:
                z = torch.randn((batch_size, uncond_score_net.x_dim), device=device)
            else:
                z = torch.zeros((batch_size, uncond_score_net.x_dim), device=device)
            extracted_noise_cond = cond_score_net(x_t, t=T.float(), cond=cond)
            extracted_noise_uncond = uncond_score_net(x_t, t=T.float(), cond=None)
            # linear combination
            extracted_noise = ((1 + cfg_weight) * extracted_noise_cond) - (
                cfg_weight * extracted_noise_uncond
            )
            if pg_weight != 0 and T_ > 0:
                if pg_gradient_type in ("cos", "cosine"):
                    guidance = cosine_guidance_updated(x_t)
                else:
                    guidance = similarity_guidance_gradient(x_t)
                if fixed_pg_ratio:
                    pg_multiplier = (
                        guidance_ratio * extracted_noise.abs().mean()
                    ) / guidance.abs().mean()
                else:
                    pg_multiplier = pg_weight
                extracted_noise = (guidance * pg_multiplier) + extracted_noise

            at = scheduler.alpha(T)
            barat = scheduler.bar_alpha(T)
            noise_factor = (1.0 - at) / ((1.0 - barat).sqrt())
            x_t = (
                torch.pow(scheduler.alpha(T), -0.5)
                * (x_t - noise_factor * extracted_noise)
                + scheduler.beta(T).sqrt() * z
            )
        return x_t.detach()


def ddpm_sample_classifier_guidance(
    score_net,
    batch_size,
    cg_weight,
    cg_due,
    cg_targets,
    cond=None,
    pg_weight=0.0,
    pg_gradient_type="euclidean",
    fixed_pg_ratio=False,
    start_at=1000,
    low_mem=False,
):

    try:
        score_net.scheduler.is_same(cg_due.scheduler)
    except:
        raise ValueError(
            f"classifier must share noise schedule with score net. different betas detected"
        )
    # cg_due.validate_self(self)

    device = next(score_net.parameters()).device
    if not cond is None:
        if cond is None and score_net.cond_dim > 0:
            raise Exception("Give me a condition.")
        else:
            assert cond.shape[0] == batch_size
    else:
        # Conditions are normally distributed random variables.
        cond = torch.randn((batch_size, score_net.cond_dim), device=device)

    guidance_ratio = 0.1 * pg_weight

    x_t = torch.randn((batch_size, score_net.x_dim), device=device, requires_grad=True)
    for T_ in reversed(range(score_net.scheduler.timesteps)):
        T = torch.ones(batch_size, dtype=torch.long, device=device) * T_
        if T_ > 0:
            z = torch.randn((batch_size, score_net.x_dim), device=device)
        else:
            z = torch.zeros((batch_size, score_net.x_dim), device=device)
        with torch.no_grad():
            extracted_noise = score_net(x_t, t=T.float(), cond=cond)
        cg_model_out = cg_due(x_t, T)

        if T_ > 0:
            if cg_weight > 0 and T_ < start_at:
                pred_loss = torch.pow(cg_model_out.mean - cg_targets, 2.0).sum()
                G = torch.autograd.grad(pred_loss, x_t)[0].detach()
                extracted_noise = extracted_noise + (cg_weight * G)
            if pg_weight > 0 and T_ < start_at:
                if pg_gradient_type in ("cos", "cosine"):
                    if low_mem:
                        guidance = low_memory_cosine_guidance_gradient(x_t)
                    else:
                        guidance = cosine_guidance_updated(x_t)
                else:
                    guidance = similarity_guidance_gradient(x_t)
                if fixed_pg_ratio:
                    pg_multiplier = (
                        guidance_ratio * extracted_noise.abs().mean()
                    ) / guidance.abs().mean()
                else:
                    pg_multiplier = pg_weight
                extracted_noise = (guidance * pg_multiplier) + extracted_noise

        at = score_net.scheduler.alpha(T)
        barat = score_net.scheduler.bar_alpha(T)
        noise_factor = (1.0 - at) / ((1.0 - barat).sqrt())
        x_t = (
            torch.pow(score_net.scheduler.alpha(T), -0.5)
            * (x_t - noise_factor * extracted_noise)
            + score_net.scheduler.beta(T).sqrt() * z
        )
    return x_t.detach()


def ddpm_sample_multi_classifier_guidance(
    score_net,
    batch_size,
    cg_weights,
    cg_dues,
    cg_targets,
    cond=None,
    pg_weight=0.0,
    pg_gradient_type="euclidean",
    fixed_pg_ratio=False,
    start_at=1000,
):

    try:
        for cg_due in cg_dues:
            score_net.scheduler.is_same(cg_due.scheduler)
    except:
        raise ValueError(
            f"classifier must share noise schedule with score net. different betas detected"
        )
    # cg_due.validate_self(self)

    device = next(score_net.parameters()).device
    if not cond is None:
        if cond is None and score_net.cond_dim > 0:
            raise Exception("Give me a condition.")
        else:
            assert cond.shape[0] == batch_size
    else:
        # Conditions are normally distributed random variables.
        cond = torch.randn((batch_size, score_net.cond_dim), device=device)

    x_t = torch.randn((batch_size, score_net.x_dim), device=device, requires_grad=True)
    for T_ in reversed(range(score_net.scheduler.timesteps)):
        T = torch.ones(batch_size, dtype=torch.long, device=device) * T_
        if T_ > 0:
            z = torch.randn((batch_size, score_net.x_dim), device=device)
        else:
            z = torch.zeros((batch_size, score_net.x_dim), device=device)
        with torch.no_grad():
            extracted_noise = score_net(x_t, t=T.float(), cond=cond)
        preds = [cg_due(x_t, T) for cg_due in cg_dues]

        if T_ > 0:
            cg_term = 0.0
            for cg_weight, cg_model_out, cg_target in zip(
                cg_weights, preds, cg_targets
            ):
                if cg_weight > 0 and T_ < start_at:
                    pred_loss = torch.pow(cg_model_out.mean - cg_target, 2.0).sum()
                    G = torch.autograd.grad(pred_loss, x_t)[0].detach()
                    cg_term += cg_weight * G

            extracted_noise = extracted_noise + cg_term
            if pg_weight > 0:
                if pg_gradient_type in ("cos", "cosine"):
                    guidance = cosine_guidance_updated(x_t)
                else:
                    guidance = similarity_guidance_gradient(x_t)
                guidance_ratio = 0.1 * pg_weight
                if fixed_pg_ratio:
                    pg_multiplier = (
                        guidance_ratio * extracted_noise.abs().mean()
                    ) / guidance.abs().mean()
                else:
                    pg_multiplier = pg_weight
                guidance = similarity_guidance_gradient(x_t)
                extracted_noise = (guidance * pg_multiplier) + extracted_noise

        at = score_net.scheduler.alpha(T)
        barat = score_net.scheduler.bar_alpha(T)
        noise_factor = (1.0 - at) / ((1.0 - barat).sqrt())
        x_t = (
            torch.pow(score_net.scheduler.alpha(T), -0.5)
            * (x_t - noise_factor * extracted_noise)
            + score_net.scheduler.beta(T).sqrt() * z
        )
    return x_t.detach()


def ddpm_basic_nearby(
    score_net,
    emb_batch,
    T_start,
    cond=None,
    pg_weight=0.0,
    embed_dim=None,
    pg_gradient_type="euclidean",
    fixed_pg_ratio=False,
    low_mem=False,
):

    batch_size = emb_batch.shape[0]
    with torch.no_grad():
        device = next(score_net.parameters()).device
        if not cond is None:
            if cond is None and score_net.cond_dim > 0:
                raise Exception("Give me a condition.")
            else:
                assert cond.shape[0] == batch_size
        else:
            # Conditions are normally distributed random variables.
            cond = torch.randn((batch_size, score_net.cond_dim), device=device)

        if embed_dim:
            cond = embed_scalar(cond, embedding_dim=embed_dim)
        T_init = torch.ones(batch_size, dtype=torch.long, device=device) * T_start
        noise = torch.randn((batch_size, score_net.x_dim), device=device)
        x_t = (
            score_net.scheduler.bar_alpha(T_init).sqrt() * emb_batch
            + (1.0 - score_net.scheduler.bar_alpha(T_init)).sqrt() * noise
        )
        for T_ in reversed(range(T_start)):
            T = torch.ones(batch_size, dtype=torch.long, device=device) * T_
            if T_ > 0:
                z = torch.randn((batch_size, score_net.x_dim), device=device)
            else:
                z = torch.zeros((batch_size, score_net.x_dim), device=device)
            extracted_noise = score_net(x_t, t=T.float(), cond=cond)
            if pg_weight > 0 and T_ > 0:
                if pg_gradient_type in ("cos", "cosine"):
                    if low_mem:
                        guidance = low_memory_cosine_guidance_gradient(x_t)
                    else:
                        guidance = cosine_guidance_updated(x_t)
                else:
                    guidance = similarity_guidance_gradient(x_t)
                guidance_ratio = 0.1 * pg_weight
                if fixed_pg_ratio:
                    pg_multiplier = (
                        guidance_ratio * extracted_noise.abs().mean()
                    ) / guidance.abs().mean()
                else:
                    pg_multiplier = pg_weight
                extracted_noise = extracted_noise + (guidance * pg_multiplier)
            at = score_net.scheduler.alpha(T)
            barat = score_net.scheduler.bar_alpha(T)
            noise_factor = (1.0 - at) / ((1.0 - barat).sqrt())
            x_t = (
                torch.pow(score_net.scheduler.alpha(T), -0.5)
                * (x_t - noise_factor * extracted_noise)
                + score_net.scheduler.beta(T).sqrt() * z
            )
        return x_t.detach()


def ddpm_cg_nearby(
    uncond_score_net,
    emb_batch,
    T_start,
    cg_due,
    targets,
    cg_weight=100.0,
    pg_weight=0.0,
):

    batch_size = emb_batch.shape[0]
    device = next(uncond_score_net.parameters()).device

    T_init = torch.ones(batch_size, dtype=torch.long, device=device) * T_start
    noise = torch.randn(
        (batch_size, uncond_score_net.x_dim), device=device, requires_grad=True
    )
    x_t = (
        uncond_score_net.scheduler.bar_alpha(T_init).sqrt() * emb_batch
        + (1.0 - uncond_score_net.scheduler.bar_alpha(T_init)).sqrt() * noise
    )
    for T_ in reversed(range(T_start)):
        T = torch.ones(batch_size, dtype=torch.long, device=device) * T_
        if T_ > 0:
            z = torch.randn((batch_size, uncond_score_net.x_dim), device=device)
        else:
            z = torch.zeros((batch_size, uncond_score_net.x_dim), device=device)
        with torch.no_grad():
            extracted_noise = uncond_score_net(x_t, t=T.float(), cond=None)
        cg_model_out = cg_due(x_t, T)
        if T_ > 0:
            if cg_weight > 0:
                pred_loss = torch.pow(cg_model_out.mean - targets, 2.0).sum()
                G = torch.autograd.grad(pred_loss, x_t)[0].detach()
                extracted_noise = extracted_noise + (cg_weight * G)
            if pg_weight > 0:
                with torch.no_grad():
                    guidance = similarity_guidance_gradient(x_t)
                    extracted_noise = (guidance * pg_weight) + extracted_noise
        if pg_weight > 0 and T_ > 0:
            extracted_noise = extracted_noise + (
                similarity_guidance_gradient(x_t) * pg_weight
            )
        at = uncond_score_net.scheduler.alpha(T)
        barat = uncond_score_net.scheduler.bar_alpha(T)
        noise_factor = (1.0 - at) / ((1.0 - barat).sqrt())
        x_t = (
            torch.pow(uncond_score_net.scheduler.alpha(T), -0.5)
            * (x_t - noise_factor * extracted_noise)
            + uncond_score_net.scheduler.beta(T).sqrt() * z
        )
    return x_t.detach()


def ddpm_multi_cg_nearby(
    uncond_score_net, emb_batch, T_start, cg_dues, cg_targets, cg_weights, pg_weight=0.0
):

    batch_size = emb_batch.shape[0]
    device = next(uncond_score_net.parameters()).device

    try:
        for cg_due in cg_dues:
            uncond_score_net.scheduler.is_same(cg_due.scheduler)
    except:
        raise ValueError(
            f"classifier must share noise schedule with score net. different betas detected"
        )

    T_init = torch.ones(batch_size, dtype=torch.long, device=device) * T_start
    noise = torch.randn(
        (batch_size, uncond_score_net.x_dim), device=device, requires_grad=True
    )
    x_t = (
        uncond_score_net.scheduler.bar_alpha(T_init).sqrt() * emb_batch
        + (1.0 - uncond_score_net.scheduler.bar_alpha(T_init)).sqrt() * noise
    )
    for T_ in reversed(range(T_start)):
        T = torch.ones(batch_size, dtype=torch.long, device=device) * T_
        if T_ > 0:
            z = torch.randn((batch_size, uncond_score_net.x_dim), device=device)
        else:
            z = torch.zeros((batch_size, uncond_score_net.x_dim), device=device)
        with torch.no_grad():
            extracted_noise = uncond_score_net(x_t, t=T.float(), cond=None)
        preds = [cg_due(x_t, T) for cg_due in cg_dues]
        if T_ > 0:
            cg_term = 0.0
            for cg_weight, cg_model_out, cg_target in zip(
                cg_weights, preds, cg_targets
            ):
                if cg_weight > 0:
                    pred_loss = torch.pow(cg_model_out.mean - cg_target, 2.0).sum()
                    G = torch.autograd.grad(pred_loss, x_t)[0].detach()
                    cg_term += cg_weight * G
            extracted_noise = extracted_noise + cg_term
            if pg_weight > 0:
                with torch.no_grad():
                    guidance = similarity_guidance_gradient(x_t)
                    extracted_noise = (guidance * pg_weight) + extracted_noise

        at = uncond_score_net.scheduler.alpha(T)
        barat = uncond_score_net.scheduler.bar_alpha(T)
        noise_factor = (1.0 - at) / ((1.0 - barat).sqrt())
        x_t = (
            torch.pow(uncond_score_net.scheduler.alpha(T), -0.5)
            * (x_t - noise_factor * extracted_noise)
            + uncond_score_net.scheduler.beta(T).sqrt() * z
        )
    return x_t.detach()


def ddpm_cfg_nearby(
    uncond_score_net,
    cond_score_net,
    emb_batch,
    T_start,
    cond,
    batch_size=4,
    pg_weight=0.0,
    cfg_weight=0.2,
):
    try:
        assert uncond_score_net.cond_dim == 0
    except:
        ValueError("Unconditional score net is first argument, conditional second")

    try:
        assert uncond_score_net.scheduler.is_same(cond_score_net.scheduler)
    except:
        ValueError("Score nets must have the same noise schedule for sampling")

    scheduler = uncond_score_net.scheduler

    with torch.no_grad():
        device = next(uncond_score_net.parameters()).device
        assert device == next(cond_score_net.parameters()).device

        T_init = torch.ones(batch_size, dtype=torch.long, device=device) * T_start
        noise = torch.randn(
            (batch_size, uncond_score_net.x_dim), device=device, requires_grad=True
        )
        x_t = (
            uncond_score_net.scheduler.bar_alpha(T_init).sqrt() * emb_batch
            + (1.0 - uncond_score_net.scheduler.bar_alpha(T_init)).sqrt() * noise
        )
        for T_ in reversed(range(T_start)):
            T = torch.ones(batch_size, dtype=torch.long, device=device) * T_
            if T_ > 0:
                z = torch.randn((batch_size, uncond_score_net.x_dim), device=device)
            else:
                z = torch.zeros((batch_size, uncond_score_net.x_dim), device=device)
            extracted_noise_cond = cond_score_net(x_t, t=T.float(), cond=cond)
            extracted_noise_uncond = uncond_score_net(x_t, t=T.float(), cond=None)
            # linear combination
            extracted_noise = ((1 + cfg_weight) * extracted_noise_cond) - (
                cfg_weight * extracted_noise_uncond
            )
            if pg_weight > 0 and T_ > 0:
                extracted_noise = extracted_noise + (
                    similarity_guidance_gradient(x_t) * pg_weight
                )
            at = scheduler.alpha(T)
            barat = scheduler.bar_alpha(T)
            noise_factor = (1.0 - at) / ((1.0 - barat).sqrt())
            x_t = (
                torch.pow(scheduler.alpha(T), -0.5)
                * (x_t - noise_factor * extracted_noise)
                + scheduler.beta(T).sqrt() * z
            )
    return x_t.detach()
