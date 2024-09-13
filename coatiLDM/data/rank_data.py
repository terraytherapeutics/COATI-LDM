import pickle as pkl

import numpy as np
import torch
from smart_open import open
from torch.utils.data.datapipes.iter import IterableWrapper

rng = np.random.default_rng(42)


def collate_rank_batch(batch):
    x_i = torch.tensor(np.vstack([entry["smiles_i_enc"] for entry in batch])).float()
    x_j = torch.tensor(np.vstack([entry["smiles_j_enc"] for entry in batch])).float()
    label = torch.tensor(np.array([entry["label"] for entry in batch]))

    return x_i, x_j, label


def make_rank_pipes(datapath, train_prob: float = 0.9, bsize: int = 32):
    print(f"loading data from {datapath}")
    with open(datapath, "rb") as inf:
        data_records = pkl.load(inf)

    rng = np.random.default_rng(42)

    # split into train/test partition
    is_train = list(rng.random((len(data_records))) < train_prob)

    train_recs = [rec for train_data, rec in zip(is_train, data_records) if train_data]
    test_recs = [
        rec for train_data, rec in zip(is_train, data_records) if not train_data
    ]

    train_pipe = (
        IterableWrapper(train_recs, deepcopy=False)
        .batch(bsize)
        .collate(collate_rank_batch)
    )

    test_pipe = (
        IterableWrapper(test_recs, deepcopy=False)
        .batch(bsize)
        .collate(collate_rank_batch)
    )

    return train_pipe, test_pipe
