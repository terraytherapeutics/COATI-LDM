import pickle
from coatiLDM.models.diffusion_models.schedulers import DDPMScheduler
from coatiLDM.models.score_models.non_conv_unet import NonConvUNet
from coatiLDM.models.diffusion_models.flow_matching import ScoreNetCondVF
from coatiLDM.models.score_models.due_cg_model import DueCG, save_due
import gpytorch
import torch
import numpy as np
from smart_open import open

implemented_score_models = ["non_conv_unet"]


def load_score_model(model_type, params, state_dict, device="cpu"):

    if model_type == "non_conv_unet":
        score_model = NonConvUNet(**params)

    else:
        raise Exception(
            f"bad score model currently implemented: {implemented_score_models}"
        )

    score_model.load_state_dict(state_dict)
    return score_model.to(device).eval()


def load_score_model_from_model_doc(doc_url, device="cpu"):
    with open(doc_url, "rb") as f_in:
        model_doc = pickle.loads(f_in.read(), encoding="UTF-8")
    model_kwargs = model_doc["score_model_params"]
    train_args = model_doc["train_args"]
    model = load_score_model(
        train_args["score_model"], model_kwargs, model_doc["model"], device=device
    )
    return model.eval(), train_args, model_doc["norm_summary"]


def load_flow_model_from_model_doc(doc_url, device="cpu"):
    model, train_args, norm_summary = load_score_model_from_model_doc(
        doc_url, device=device
    )
    return ScoreNetCondVF(model), train_args, norm_summary


def load_due_cg(due_params, state_dict, device="cpu"):
    due = DueCG(**due_params)
    if not due.initalized:
        dummy_size = due.n_inducing_points + 10
        dummy_x = torch.zeros((dummy_size, due.x_dim))
        dummy_t = torch.zeros((dummy_size,))
        due.initalize_model(dummy_x, dummy_t)
    due.load_state_dict(state_dict)
    due = due.to(device)
    due = due.eval()

    return due


def load_due_cg_from_model_doc(
    doc_url,
    remove_spectral_norm=True,
    device="cpu",
):
    with open(doc_url, "rb") as f_in:
        model_doc = pickle.loads(f_in.read(), encoding="UTF-8")
    model_kwargs = model_doc["model_kwargs"]
    due = load_due_cg(model_kwargs, model_doc["model"], device=device)
    if remove_spectral_norm:
        due.feature_extractor.first = torch.nn.utils.remove_spectral_norm(
            due.feature_extractor.first
        )
    return due.eval()
