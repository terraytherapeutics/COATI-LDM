import torch
from coatiLDM.models.score_models.due_cg_model import DueCG, save_due
from gpytorch.mlls import VariationalELBO
from tqdm.auto import tqdm

from coatiLDM.models.diffusion_models.schedulers import DDPMScheduler
from coatiLDM.data.datapipe import get_base_pipe
from coatiLDM.data.transforms import cg_xform_routine
from coatiLDM.common.utils import makedir, utc_epoch_now
import argparse
import os


def train_cg_due(datapipe, due_params, n_samples, lr=1e-3, epochs=100, device="cuda:0"):
    train_samples = []
    Ts = []
    total = 0
    for i, batch in enumerate(datapipe):

        if total < n_samples:
            train_samples.append(batch["noised_samples"])
            Ts.append(batch["T"])
        total += len(batch["noised_samples"])
    train_samples = torch.cat(train_samples, dim=0)[:n_samples]
    train_Ts = torch.cat(Ts, dim=0)[:n_samples]
    model = DueCG(**due_params).to("cpu")
    model.initalize_model(X=train_samples, T=train_Ts)
    model = model.to(device)
    elbo = VariationalELBO(model.likelihood, model.gp, num_data=total)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            loss = -elbo(preds, targets.to(device))
            loss.backward()
            optimizer.step()
            t.set_description(f"Epoch {epoch}, Loss: {loss.item():.2f}")
            avg_loss += loss.item()

        avg_loss /= i + 1
    return model


def train_cg(args):

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
                timesteps=args.timesteps,
                bar_alphas=sched_bar_alphas,
            )
        )
    )

    print("obtaining test batch... ")
    test_batch = next(iter(datapipe))
    x_dim = test_batch["unnoised_samples"].shape[-1]

    due_params = {
        "scheduler": scheduler,
        "scalar_name": args.scalar_field,
        "time_embed_dim": args.time_dim,
        "train_data_sample": None,  # their implementation only uses 1k samples
        "x_dim": x_dim,
        "depth": 4,
        "num_outputs": 1,
        "spectral_normalization": True,
        "n_inducing_points": args.n_inducing_points,
        "soft_norm_coeff": args.soft_norm_coeff,
        "n_power_iterations": args.n_power_iterations,
        "dropout_rate": args.dropout_rate,
        "kernel": "RBF",
    }
    model = train_cg_due(
        datapipe,
        due_params,
        args.n_samples,
        lr=args.lr,
        epochs=args.num_epochs,
        device=args.device,
    )
    params = vars(args)

    model.eval()
    model = model.to("cpu")
    output_path = os.path.join(
        args.model_dir, f"{args.exp_name}_{args.run_name}_final.pkl"
    )
    serialized_artifact = save_due(due_params, model)

    with open(output_path, "wb") as f_out:
        f_out.write(serialized_artifact)

    print("saved model to: ", output_path)
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
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--schedule", type=str, default="linear")
    if mock_args:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = do_args()
    train_cg(args)
