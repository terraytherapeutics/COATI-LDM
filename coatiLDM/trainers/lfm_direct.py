#
# Distributed Latent Flow matching.
# because why not?
#

import pickle, os, argparse, copy, datetime, gc
import numpy as np
import pandas as pd
import json
from datetime import timedelta

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
from torch.optim import Adam
import torchdata
from torch.utils.data.datapipes.iter import IterableWrapper

# Distributed stuff.
from smart_open import open


from coatiLDM.models.diffusion_models.flow_matching import (
    CondOTFlowMatching,
    ScoreNetCondVF,
)
from coatiLDM.common.utils import makedir, utc_epoch_now
from coatiLDM.common.ema import ExponentialMovingAverage

# from coatiLDM.data.transforms import xform_basic, inference_helper, embed_scalar
from coatiLDM.data.transforms import xform_basic

from coatiLDM.models.score_models.non_conv_unet import NonConvUNet


def save_model(model, output_path, norm_summary=None, train_args=None):
    torch.save(model, output_path + ".pt")
    with open(output_path + ".pkl", "wb") as f:
        pickle.dump({"norm_summary": norm_summary, "train_args": train_args}, f)


def serialize_score_model(score_model, norm_summary, train_args, score_model_params):
    model_state_dict = score_model.to("cpu").state_dict()
    model_serialized = {
        "model": model_state_dict,
        "norm_summary": norm_summary,
        "score_model_params": score_model_params,
        "train_args": train_args,
    }
    return pickle.dumps(model_serialized)


def train_flow(args):
    makedir(args.model_dir)

    data_split_name = args.data_path.split("/")[-1].split(".")[0]
    tags = {
        "data_path": f"direct_{args.data_path}",
        "run_name": f"run__{args.run_name}",
        "diff_model": args.diff_type,
        "score_model": args.score_model,
        "data_split_name": data_split_name,
    }

    train_val_meta_dict = pickle.load(open(args.data_path, "rb"))
    try:
        norm_summary = train_val_meta_dict["cond_cdfs"]
    except:
        norm_summary = None

    if isinstance(train_val_meta_dict, dict):
        try:
            coati_doc = train_val_meta_dict["metadata"]["coati_doc"]
        except:
            coati_doc = None
    else:
        train_val_meta_dict = {"train": train_val_meta_dict}
        coati_doc = None
    # load it all into memory.

    base_pipe = IterableWrapper(train_val_meta_dict["train"], deepcopy=False)

    print("loaded data from " + args.data_path)
    print(f"batch_size:{args.batch_size}")
    print("device:", args.device)

    datapipe = (
        base_pipe.shuffle()
        .batch(args.batch_size)
        .collate(
            lambda batch: xform_basic(
                batch,
                x_field=args.x_field,
                scalar_cond_fields=args.scalar_cond_fields,
                cond_emb_dim=args.dim_per_cond,
                device=args.device,
            )
        )
    )

    print("obtaining test batch... ")
    test_batch = next(iter(datapipe))

    x_dim = test_batch["samples"].shape[-1]
    if not test_batch["cond_vector"] is None:
        cond_dim = test_batch["cond_vector"].shape[-1]
    else:
        cond_dim = 0
    print("building model... ")

    if args.score_model == "non_conv_unet":
        score_model_params = {
            "x_dim": x_dim,
            "cond_dim": cond_dim,
            "time_max": 1.0,
            "time_dim": args.time_dim,
            "dropout": 0.0,
            "scheduler": None,
            "bias": args.bias,
            "use_weight_norm": args.use_weight_norm,
        }
        score_model = NonConvUNet(**score_model_params)
    else:
        raise ValueError("only score_model == non_conv_unet is supported currently")

    print(f"Cond dim {cond_dim}")

    if args.diff_type == "flow_matching":
        flow_matching = CondOTFlowMatching()
        flow_model = ScoreNetCondVF(score_model).to(args.device)  # Has no parameters.
    else:
        print(args.diff_type)
        raise Exception("bad diff_type")

    print("DEVICE: ", args.device)
    print("Flow Model: ")
    print(flow_model)

    train_args = vars(args)
    train_args["x_dim"] = x_dim
    train_args["cond_dim"] = cond_dim
    train_args["coati_doc"] = coati_doc

    optimizer = torch.optim.AdamW(
        flow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.ema:
        ema = ExponentialMovingAverage(flow_model.parameters(), decay=args.ema_const)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    num = 1
    ave_losses = []

    for epoch in range(0, args.num_epochs + 1):
        losses = []
        with tqdm(datapipe, desc="Epoch {} ".format(epoch), unit="batch") as tepoch:
            for batch in tepoch:
                optimizer.zero_grad()
                loss = flow_matching.loss(
                    flow_model, batch["samples"], batch["cond_vector"]
                )
                loss.backward()

                torch.nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
                optimizer.step()
                if args.ema:
                    ema.update()
                losses.append(loss.detach().cpu().numpy().item())
                tepoch.set_postfix(loss=loss.item())
                num += 1

        scheduler.step()
        ave = 0
        for loss in losses:
            ave += loss
        ave = ave / len(losses)
        ave_losses.append(ave)
        print("Epoch {}: Loss: {:.8f}".format(epoch, ave))

    output_path = os.path.join(args.model_dir, f"{args.exp_name}_{args.run_name}_final")
    if args.ema:
        with ema.average_parameters():
            print("serializing ema model")
            score_model_serialized = serialize_score_model(
                flow_model.score_net, norm_summary, train_args, score_model_params
            )
    else:
        print("serializing non-ema model")
        score_model_serialized = serialize_score_model(
            flow_model.score_net, norm_summary, train_args, score_model_params
        )

    with open(output_path + ".pkl", "wb") as f:
        f.write(score_model_serialized)

    return flow_model


def do_args(mock_args=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="flow")
    parser.add_argument("--run_name", type=str, default=str(int(utc_epoch_now())))
    parser.add_argument("--model_dir", type=str, default="./")

    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--diff_type", type=str, default="flow_matching")
    parser.add_argument("--score_model", type=str, default="non_conv_unet")
    parser.add_argument(
        "--scalar_cond_fields", type=list, default=["INVSIGCDF", "logp"]
    )
    parser.add_argument("--dim_per_cond", type=int, default=16)
    parser.add_argument("--use_weight_norm", type=bool, default=False)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--time_dim", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--ema", type=bool, default=True)
    parser.add_argument("--ema_const", type=float, default=0.996)

    parser.add_argument("--torch_compile", type=bool, default=False)
    parser.add_argument("--restore_chkpt", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--skip_train", type=bool, default=False)
    parser.add_argument("--save_every_n_updates", type=int, default=1e6)
    parser.add_argument("--batch_size", type=int, default=896)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=2e-4)
    if mock_args:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    return args
