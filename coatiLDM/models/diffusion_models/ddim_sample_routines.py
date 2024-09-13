import torch
from coatiLDM.models.diffusion_models.particle_guidance import (
    similarity_guidance_gradient,
)
from coatiLDM.data.transforms import embed_scalar
import numpy as np


def ddim_basic_nearby(
    score_net,
    x_start,
    cond=None,
    pg_weight=0.0,
    embed_dim=None,
    eta=1.0,
    T_start=200,
    skip=1,
):

    assert T_start % skip == 0
    with torch.no_grad():
        batch_size = x_start.shape[0]

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
            score_net.scheduler.bar_alpha(T_init).sqrt() * x_start
            + (1.0 - score_net.scheduler.bar_alpha(T_init)).sqrt() * noise
        )

        # DDIM setup force this

        seq = range(0, T_start, skip)
        seq_next = [-1] + list(seq[:-1])

        for T_, T_NEXT_ in zip(reversed(seq), reversed(seq_next)):
            T = torch.ones(batch_size, dtype=torch.long, device=device) * T_
            T_NEXT = torch.ones(batch_size, dtype=torch.long, device=device) * T_NEXT_
            barat = score_net.scheduler.bar_alpha(T)
            if T_NEXT_ == -1:
                barat_next = torch.ones((batch_size, 1), device=device)
            else:
                barat_next = score_net.scheduler.bar_alpha(T_NEXT)
            extracted_noise = score_net(x_t, t=T.float(), cond=cond)
            if pg_weight > 0:
                extracted_noise = extracted_noise + (
                    similarity_guidance_gradient(x_t) * pg_weight
                )
            x_t = (x_t - extracted_noise * (1 - barat).sqrt()) / barat.sqrt()
            c1 = (
                eta * ((1 - barat / barat_next) * (1 - barat_next) / (1 - barat)).sqrt()
            )
            c2 = ((1 - barat_next) - c1**2).sqrt()
            x_t = (
                barat_next.sqrt() * x_t
                + c1 * torch.randn_like(x_t)
                + c2 * extracted_noise
            )

        return x_t.detach()


def ddim_basic_sample(
    score_net,
    cond=None,
    batch_size=4,
    pg_weight=0.0,
    embed_dim=None,
    eta=1.0,
    ddim_steps=1000,
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
        # torch.manual_seed(0)
        x_t = torch.randn((batch_size, score_net.x_dim), device=device)

        # DDIM setup
        skip = score_net.scheduler.timesteps // ddim_steps
        assert score_net.scheduler.timesteps % skip == 0
        seq = range(0, score_net.scheduler.timesteps, skip)
        seq_next = [-1] + list(seq[:-1])

        for T_, T_NEXT_ in zip(reversed(seq), reversed(seq_next)):
            T = torch.ones(batch_size, dtype=torch.long, device=device) * T_
            T_NEXT = torch.ones(batch_size, dtype=torch.long, device=device) * T_NEXT_
            barat = score_net.scheduler.bar_alpha(T)
            if T_NEXT_ == -1:
                barat_next = torch.ones((batch_size, 1), device=device)
            else:
                barat_next = score_net.scheduler.bar_alpha(T_NEXT)
            extracted_noise = score_net(x_t, t=T.float(), cond=cond)
            if pg_weight > 0:
                extracted_noise = extracted_noise + (
                    similarity_guidance_gradient(x_t) * pg_weight
                )
            x_t = (x_t - extracted_noise * (1 - barat).sqrt()) / barat.sqrt()
            c1 = (
                eta * ((1 - barat / barat_next) * (1 - barat_next) / (1 - barat)).sqrt()
            )
            c2 = ((1 - barat_next) - c1**2).sqrt()
            x_t = (
                barat_next.sqrt() * x_t
                + c1 * torch.randn_like(x_t)
                + c2 * extracted_noise
            )

        return x_t.detach()


def ddim_cfg_nearby(
    uncond_score_net,
    cond_score_net,
    x_start,
    cond,
    pg_weight=0.0,
    cfg_weight=0.2,
    eta=1.0,
    T_start=200,
    skip=1,
):

    batch_size = x_start.size(0)
    try:
        assert uncond_score_net.cond_dim == 0
    except:
        ValueError("Unconditional score net is first argument, conditional second")

    try:
        assert uncond_score_net.scheduler.is_same(cond_score_net.scheduler)
    except:
        ValueError("Score nets must have the same noise schedule for sampling")

    scheduler = uncond_score_net.scheduler

    assert T_start % skip == 0

    device = next(uncond_score_net.parameters()).device
    assert device == next(cond_score_net.parameters()).device

    # DDIM setup force this

    seq = range(0, T_start, skip)
    seq_next = [-1] + list(seq[:-1])

    with torch.no_grad():
        T_init = torch.ones(batch_size, dtype=torch.long, device=device) * T_start
        noise = torch.randn((batch_size, uncond_score_net.x_dim), device=device)

        x_t = (
            scheduler.bar_alpha(T_init).sqrt() * x_start
            + (1.0 - scheduler.bar_alpha(T_init)).sqrt() * noise
        )

        for T_, T_NEXT_ in zip(reversed(seq), reversed(seq_next)):

            T = torch.ones(batch_size, dtype=torch.long, device=device) * T_
            T_NEXT = torch.ones(batch_size, dtype=torch.long, device=device) * T_NEXT_
            barat = scheduler.bar_alpha(T)
            if T_NEXT_ == -1:
                barat_next = torch.ones((batch_size, 1), device=device)
            else:
                barat_next = scheduler.bar_alpha(T_NEXT)
            extracted_noise_cond = cond_score_net(x_t, t=T.float(), cond=cond)
            extracted_noise_uncond = uncond_score_net(x_t, t=T.float(), cond=None)
            extracted_noise = ((1 + cfg_weight) * extracted_noise_cond) - (
                cfg_weight * extracted_noise_uncond
            )
            if pg_weight > 0:
                extracted_noise = extracted_noise + (
                    similarity_guidance_gradient(x_t) * pg_weight
                )
            x_t = (x_t - extracted_noise * (1 - barat).sqrt()) / barat.sqrt()
            c1 = (
                eta * ((1 - barat / barat_next) * (1 - barat_next) / (1 - barat)).sqrt()
            )
            c2 = ((1 - barat_next) - c1**2).sqrt()
            x_t = (
                barat_next.sqrt() * x_t
                + c1 * torch.randn_like(x_t)
                + c2 * extracted_noise
            )
        return x_t.detach()


def ddim_sample_classifier_free_guidance(
    uncond_score_net,
    cond_score_net,
    cond,
    batch_size=4,
    pg_weight=0.0,
    cfg_weight=0.2,
    eta=1.0,
    ddim_steps=1000,
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

        x_t = torch.randn((batch_size, uncond_score_net.x_dim), device=device)

        # DDIM setup
        skip = scheduler.timesteps // ddim_steps
        assert scheduler.timesteps % skip == 0
        seq = range(0, scheduler.timesteps, skip)
        seq_next = [-1] + list(seq[:-1])
        for T_, T_NEXT_ in zip(reversed(seq), reversed(seq_next)):

            T = torch.ones(batch_size, dtype=torch.long, device=device) * T_
            T_NEXT = torch.ones(batch_size, dtype=torch.long, device=device) * T_NEXT_
            barat = scheduler.bar_alpha(T)
            if T_NEXT_ == -1:
                barat_next = torch.ones((batch_size, 1), device=device)
            else:
                barat_next = scheduler.bar_alpha(T_NEXT)
            extracted_noise_cond = cond_score_net(x_t, t=T.float(), cond=cond)
            extracted_noise_uncond = uncond_score_net(x_t, t=T.float(), cond=None)
            extracted_noise = ((1 + cfg_weight) * extracted_noise_cond) - (
                cfg_weight * extracted_noise_uncond
            )
            if pg_weight > 0:
                extracted_noise = extracted_noise + (
                    similarity_guidance_gradient(x_t) * pg_weight
                )
            x_t = (x_t - extracted_noise * (1 - barat).sqrt()) / barat.sqrt()
            c1 = (
                eta * ((1 - barat / barat_next) * (1 - barat_next) / (1 - barat)).sqrt()
            )
            c2 = ((1 - barat_next) - c1**2).sqrt()
            x_t = (
                barat_next.sqrt() * x_t
                + c1 * torch.randn_like(x_t)
                + c2 * extracted_noise
            )
        return x_t.detach()


def ddim_sample_classifier_guidance(
    score_net,
    batch_size,
    cg_weight,
    cg_due,
    cg_targets,
    cond=None,
    pg_weight=0.0,
    eta=1.0,
    ddim_steps=1000,
):
    try:
        score_net.scheduler.is_same(cg_due.scheduler)
    except:
        raise ValueError(
            f"classifier must share noise schedule with score net. different betas detected"
        )

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

    # DDIM setup
    skip = score_net.scheduler.timesteps // ddim_steps
    assert score_net.scheduler.timesteps % skip == 0
    seq = range(0, score_net.scheduler.timesteps, skip)
    seq_next = [-1] + list(seq[:-1])
    for T_, T_NEXT_ in zip(reversed(seq), reversed(seq_next)):

        T = torch.ones(batch_size, dtype=torch.long, device=device) * T_
        T_NEXT = torch.ones(batch_size, dtype=torch.long, device=device) * T_NEXT_
        barat = score_net.scheduler.bar_alpha(T)
        if T_NEXT_ == -1:
            barat_next = torch.ones((batch_size, 1), device=device, requires_grad=True)
        else:
            barat_next = score_net.scheduler.bar_alpha(T_NEXT)
        with torch.no_grad():
            extracted_noise = score_net(x_t, t=T.float(), cond=None)
        cg_model_out = cg_due(x_t, T)
        if cg_weight > 0:
            pred_loss = torch.pow(cg_model_out.mean - cg_targets, 2.0).sum()
            G = torch.autograd.grad(pred_loss, x_t)[0].detach()
            extracted_noise = extracted_noise + (cg_weight * G)
        if pg_weight > 0:
            with torch.no_grad():
                extracted_noise = extracted_noise + (
                    similarity_guidance_gradient(x_t) * pg_weight
                )

        x_t = (x_t - extracted_noise * (1 - barat).sqrt()) / barat.sqrt()
        c1 = eta * ((1 - barat / barat_next) * (1 - barat_next) / (1 - barat)).sqrt()
        c2 = ((1 - barat_next) - c1**2).sqrt()
        x_t = (
            barat_next.sqrt() * x_t + c1 * torch.randn_like(x_t) + c2 * extracted_noise
        )
    return x_t.detach()
