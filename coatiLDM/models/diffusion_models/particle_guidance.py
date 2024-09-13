import numpy as np
import torch
from tqdm.auto import tqdm


def similarity_guidance_gradient(x):
    """
    Produces the gradient of the K(X, X') similarity kernel
    Here we use the norm for similarity/distance
    """
    with torch.no_grad():
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        distance = torch.norm(diff, p=2, dim=-1, keepdim=True)
        num_latents = x.shape[0]
        h_t = (
            distance.mean(dim=1, keepdim=True) * num_latents / (num_latents - 1)
        ) ** 2 / np.log(num_latents)
        weights = torch.exp(-(distance**2 / h_t))
        grad_phi = 2 * weights * diff / h_t * 2
        grad_phi = grad_phi.sum(dim=1)
    return -grad_phi


def low_memory_cosine_guidance_gradient(vectors):
    with torch.enable_grad():
        tore = []
        B = vectors.clone().detach().requires_grad_(False)
        CS = torch.nn.CosineSimilarity()
        for k in range(vectors.shape[0]):
            A = vectors[k].clone().detach().requires_grad_(True)
            B_ = B.clone()
            B_[k] = 0.0
            tore.append(
                torch.autograd.grad(
                    (10.0 * torch.erfinv(CS(A, B_).clamp(-0.999, 0.999))).exp().mean(),
                    [A],
                )[0].detach()
            )
            del A
    return torch.stack(tore, 0)


def cosine_guidance_gradient(input_batch):
    """
    Computes the gradient of the cosine distances between each vector in the batch.

    Args:
        input_batch (torch.Tensor): Input batch of vectors, shape (batch_size, vector_size).

    Returns:
        torch.Tensor: Gradient of cosine distances summed across dim=1, shape (batch_size, vector_size).
    """

    with torch.enable_grad():

        cloned_batch = input_batch.clone().detach().requires_grad_(True)

        # Compute cosine similarity matrix
        sim_adj = torch.nn.functional.cosine_similarity(
            cloned_batch.unsqueeze(1), cloned_batch.unsqueeze(0)
        )

        sim_adj.fill_diagonal_(0)

        sim_adj = sim_adj.abs().sum()  # .abs().sum()

        # Compute gradient of cosine similarity matrix
        cos_sim_grad = torch.autograd.grad(sim_adj, cloned_batch, create_graph=True)

        # Compute gradient of cosine distances
        cos_dist_grad = cos_sim_grad[0]

        return cos_dist_grad  # .detach().cpu()


def cosine_guidance_updated(A):
    with torch.enable_grad():
        A_ = A.clone().detach().requires_grad_(True)
        A_nrm = A_ / ((A_ * A_).sum(-1, keepdims=True).sqrt())
        no_diag = torch.einsum("ij,kj->ik", A_nrm, A_nrm) * (
            1.0 - torch.eye(A_.shape[0], device=A_.device, requires_grad=True)
        )

        return torch.autograd.grad(no_diag.clamp(-0.9999, 0.9999).mean(), [A_])[0]
