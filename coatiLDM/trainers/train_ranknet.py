import torch
from tqdm import tqdm, trange
import torch.nn.functional as F
import pickle as pkl
from coatiLDM.common.utils import makedir, utc_epoch_now
import os


def get_reg_loss(
    score_i: torch.Tensor, score_j: torch.Tensor, regularization_factor: float = 1e-6
) -> torch.Tensor:
    """Returns regularization loss for the scores ||s||^2 / batch_size
    and scales it by `regularization_factor`
    """
    batch_size = score_i.size(0)
    reg_loss = (
        regularization_factor
        * (torch.norm(score_i) ** 2 + torch.norm(score_j) ** 2)
        / batch_size
    )
    return reg_loss


# target is whether or not smiles_j was chosen.
# lower score is better - logit is higher if score_j < score_i.
def get_loss(score_i, score_j, target):
    logit = (score_i - score_j).squeeze()
    return F.binary_cross_entropy_with_logits(logit, target.float())


def train_ranknet(train_pipe, model, lr=3e-4, epochs=10, device="cuda:0"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # stats = []

    for epoch in range(epochs):
        print(f"epoch {epoch}")
        t = tqdm(train_pipe)
        for i, (emb_i, emb_j, labels) in enumerate(t):
            emb_i = emb_i.to(device)
            emb_j = emb_j.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            (score_i, score_j, logit) = model(emb_i, emb_j)
            loss = get_loss(score_i, score_j, labels) + get_reg_loss(score_i, score_j)
            loss.backward()
            optimizer.step()
            t.set_description(f"Loss: {loss}")
    #         if i % LOG_EVERY:
    #             stats.append({"epoch": epoch, "batch": i, "loss": loss.cpu().detach()})

    # with open(os.path.join(prefix, "stats.pkl"), "wb") as outf:
    #     pkl.dump(stats, outf)


def infer_pipe(test_pipe, model, device="cuda:0"):
    test_entries = []
    for emb_i, emb_j, labels in test_pipe:
        emb_i = emb_i.to(device)
        emb_j = emb_j.to(device)
        labels = labels.to(device)

        with torch.inference_mode():
            score_i, score_j, logits = model(emb_i, emb_j)

        for idx in range(emb_i.size()[0]):
            test_entries.append(
                {
                    "score_i": float(score_i[idx]),
                    "score_j": float(score_j[idx]),
                    "logit": float(logits[idx]),
                    "label": int(labels[idx]),
                }
            )

    return test_entries
