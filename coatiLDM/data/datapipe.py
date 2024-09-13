import os
from torch.utils.data.datapipes.iter import FileLister, Shuffler, IterableWrapper
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torchdata.datapipes.iter import FileLister, InMemoryCacheHolder
from torch.utils.data.datapipes._decorator import functional_datapipe
import pickle


@functional_datapipe("unstack_picklesv2")
class UnstackPickles(IterDataPipe):
    def __init__(self, dp) -> None:
        super().__init__()
        self.dp = dp

    def __iter__(self):
        for X in self.dp:
            # print('loading... ',X)
            with open(X, "rb") as f:
                raw_rows = pickle.load(f)
            yield raw_rows


def get_cache_pipe(
    cache_dir, masks=["*chunk*.pkl"], mem_cache=False, mem_cache_size=200000
):
    pipe = (
        FileLister(
            root=cache_dir,
            recursive=False,
            masks=masks,
        )
        .shuffle()
        # .open_files(mode="rb")
        .unstack_picklesv2()
        .unbatch()
        # .in_memory_cache(size=10_000)
        .shuffle(buffer_size=200_000)
    )
    if mem_cache:
        pipe = pipe.in_memory_cache(size=mem_cache_size)
    return pipe


def get_dist_pipe(data_path, cache_mask=["*.pkl"]):
    pipe = (
        FileLister(
            root=data_path,
            recursive=False,
            masks=cache_mask,
        )
        .shuffle()
        .unstack_picklesv2()
        .unbatch()
        .sharding_filter()
        .shuffle(buffer_size=20_000)
        .in_memory_cache(size=10_000)
    )
    return pipe


def get_base_pipe(data_path, load_type, cache_mask=["*chunk*.pkl"]):

    if load_type == "pickle":
        encoded_data = pickle.load(open(data_path, "rb"))
        base_pipe = IterableWrapper(encoded_data, deepcopy=False)
    elif load_type == "cache":
        base_pipe = get_cache_pipe(data_path, masks=cache_mask)
    elif load_type == "torch":
        encoded_data = pickle.load(data_path)
        base_pipe = IterableWrapper(encoded_data, deepcopy=False)
    # this is dumb, but I want this to work with unified train routines
    # and I'm godless. feel free to reformulate - Ben.
    elif load_type == "train_val_dict":
        split_dict = pickle.load(open(data_path, "rb"))
        encoded_data = split_dict["train"]
        base_pipe = IterableWrapper(encoded_data, deepcopy=False)
    else:
        raise ValueError("Unknown load type, choose from: [pickle, torch]")
    return base_pipe
