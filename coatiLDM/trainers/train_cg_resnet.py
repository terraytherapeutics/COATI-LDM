import torch
from coatiLDM.models.score_models.resnet import ResAttnNetWithTime
from gpytorch.mlls import VariationalELBO
from tqdm.auto import tqdm
from coatiLDM.models.diffusion_models.schedulers import DDPMScheduler
from coatiLDM.data.datapipe import get_base_pipe
from coatiLDM.data.transforms import cg_xform_routine
from coatiLDM.common.utils import makedir, utc_epoch_now
import argparse
import os


def train_resnet(
    datapipe,
    resnet_params,
    n_samples,
    lr=1e-3,
    epochs=100,
    mode="regression",
    device="cuda:0",
):
    t_embed_dim = resnet_params["t_emb_dim"]
    resnet_params = {k: v for k, v in resnet_params.items() if k != "t_emb_dim"}
    model = ResAttnNetWithTime(resnet_params, t_emb_dim=t_embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if mode == "regression":
        loss_fn = torch.nn.functional.mse_loss
    else:
        # loss for binary classification
        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    for epoch in range(epochs):
        t = tqdm(datapipe, desc=f"Epoch {epoch}, Loss: ")
        model.train()
        avg_loss = 0
        for i, batch in enumerate(t):
            feats = batch["noised_samples"]
            targets = batch["target"]
            Ts = batch["T"]
            optimizer.zero_grad()
            preds = model(feats.to(device), Ts.to(device))
            loss = loss_fn(preds, targets.view(preds.shape).to(device))
            loss.backward()
            optimizer.step()
            t.set_description(f"Epoch {epoch}, Loss: {loss.item():.2f}")
            avg_loss += loss.item()

        avg_loss /= i + 1
    return model


def train_resnet_with_time(args):

    tags = {
        "data_path": f"norm_summary__{args.data_path}",
        "run_name": f"run__{args.run_name}",
        "target": args.scalar_field,
    }

    base_pipe = get_base_pipe(args.data_path, args.load_type)
    x_dim = next(iter(base_pipe))[args.x_field].shape[-1]

    scheduler = DDPMScheduler(
        schedule=args.schedule, timesteps=args.timesteps, beta_start=1e-4, beta_end=0.02
    )
    sched_bar_alphas = scheduler.all_bar_alphas.clone().detach().cpu()
    datapipe = (
        base_pipe.shuffle()
        .batch(args.batch_size)
        .collate(
            lambda batch: cg_xform_routine(
                batch,
                x_field=args.x_field,
                scalar_field=args.scalar_field,
                timesteps=999,
                bar_alphas=sched_bar_alphas,
            )
        )
    )

    print("obtaining test batch... ")
    test_batch = next(iter(datapipe))
    x_dim = test_batch["unnoised_samples"].shape[-1]

    resnet_params = {
        "input_dim": x_dim,
        "hidden_dim": args.hidden_dim,
        "activation": args.activation,
        "n_heads": args.n_heads,
        "specnorm": args.specnorm,
        "depth": args.depth,
        "mup": False,
        "t_emb_dim": args.time_dim,
        "output_dim": 1,
    }
    model = train_resnet(
        datapipe,
        resnet_params,
        args.n_samples,
        lr=args.lr,
        epochs=args.num_epochs,
        mode=args.mode,
        device=args.device,
    )
    params = vars(args)
    model.eval()
    model = model.to("cpu")
    output_path = os.path.join(args.model_dir, f"{args.exp_name}_{args.run_name}_final")
    torch.save(model, output_path + ".pt")
    print("saved model to: ", output_path + ".pt")
    return model


def do_args(mock_args=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="cg_model")
    parser.add_argument("--run_name", type=str, default=str(int(utc_epoch_now())))
    parser.add_argument("--model_dir", type=str, default="./")
    parser.add_argument("--load_type", type=str, default="pickle")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--x_field", type=str, default="normd_vector")
    parser.add_argument("--no_noise", type=bool, default=False)

    parser.add_argument("--scalar_field", type=list, default="normd_logp")
    parser.add_argument("--dropout_rate", type=float, default=0.03)
    parser.add_argument("--n_inducing_points", type=int, default=60)
    parser.add_argument("--soft_norm_coeff", type=float, default=0.95)
    parser.add_argument("--n_power_iterations", type=int, default=2)

    parser.add_argument("--time_dim", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=1000)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=896)
    parser.add_argument("--gamma", type=float, default=9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--specnorm", type=bool, default=False)
    parser.add_argument("--mode", type=str, default="regression")
    parser.add_argument("--depth", type=int, default=12)
    if mock_args:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = do_args()
    train_resnet_with_time(args)
