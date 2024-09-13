import torch
from rdkit import Chem
import numpy as np


def force_decode_valid_batch_efficient(
    V,
    encoder,
    tokenizer,
    max_attempts=64,
    inv_temp=1.5,
    k=2000,
    noise_scale=0.0,
    chiral=True,
    silent=False,
):

    # logger.debug(f"Running decode with max {max_attempts} attempts")
    device = V.device
    assert V.device == next(encoder.parameters()).device

    mols = ["" for _ in range(V.shape[0])]
    indices = list(range(V.shape[0]))
    vectors = V.detach().clone()

    for _ in range(max_attempts):
        with torch.no_grad():
            assert vectors.dim() == 2
            if chiral:
                regen_smiles = encoder.hcoati_to_2d_batch(
                    vectors, tokenizer, inv_temp=inv_temp, k=k, noise_scale=noise_scale
                )
            else:
                regen_smiles = encoder.hclip_to_2d_batch(
                    vectors, tokenizer, inv_temp=inv_temp, k=k, noise_scale=noise_scale
                )
            fail_flag = [1 for k in range(vectors.shape[0])]
            for j in range(vectors.shape[0]):
                smiles = regen_smiles[j]
                if smiles == "C":
                    continue
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        raise Exception
                    mols[indices[j]] = smiles
                    fail_flag[j] = 0
                except Exception as e:
                    continue
            vectors = (
                vectors[
                    torch.tensor(fail_flag, dtype=torch.bool, device=vectors.device)
                ]
                .clone()
                .detach()
            )
            indices = [indices[j] for j in range(len(indices)) if fail_flag[j]]
            if not silent:
                print(len(indices), " remaining ")
            if (len(indices)) == 0:
                break

    return mols
