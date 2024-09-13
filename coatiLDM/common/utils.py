import itertools
import os
import shutil
import datetime
from datetime import timezone
from rdkit import Chem
from rdkit.Chem.AllChem import (
    GetMorganFingerprintAsBitVect,
)
import numpy as np

import torch


def makedir(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.
    If a directory is provided, creates that directory. If a file is provided (i.e. isfile == True),
    creates the parent directory for that file.
    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != "":
        os.makedirs(path, exist_ok=True)


def rmdir(path: str):
    """
    Creates a directory given a path to either a directory or file.
    If a directory is provided, creates that directory. If a file is provided (i.e. isfile == True),
    creates the parent directory for that file.
    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    try:
        shutil.rmtree(path)
    except Exception as Ex:
        print("rmdir failure", Ex)


def utc_epoch_now():
    return datetime.datetime.now().replace(tzinfo=timezone.utc).timestamp()


def uniform_sample_in_range(sample_shape, a, b):
    return ((b - a) * torch.rand(sample_shape) + a).numpy()


def batch_iterable(iterable, n=128):
    if isinstance(iterable, list):
        iterable = iter(iterable)

    while True:
        batch = list(itertools.islice(iterable, n))
        if not batch:
            break
        yield batch


def mol_to_morgan(
    smiles: str,
    radius: int = 3,
    n_bits: int = 2048,
    chiral: bool = False,
    features: bool = False,
) -> np.ndarray:
    # if any([a.GetAtomicNum()==1 for a in mol.GetAtoms()]):
    #     print(f'WARNING: mol has hydrogens during morgan creation: "{Chem.MolToSmiles(mol)}"')
    mol = Chem.MolFromSmiles(smiles)
    return np.frombuffer(
        GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=n_bits,
            useChirality=chiral,
            useFeatures=features,
        )
        .ToBitString()
        .encode(),
        "u1",
    ) - ord("0")


def tanimoto_distance_torch(A, B):
    A = torch.tensor(A, dtype=torch.float32).to(B.device)
    dot_products = torch.mm(A, B.T)
    norm_A = torch.sum(A**2, axis=1)
    norm_B = torch.sum(B**2, axis=1)
    distances = 1 - dot_products / (norm_A[:, None] + norm_B[None, :] - dot_products)
    return distances


def colored_background(r: int, g: int, b: int, text):
    """
    r,g,b integers between 0,255
    """
    return f"\033[48;2;{r};{g};{b}m{text}\033[0m"
