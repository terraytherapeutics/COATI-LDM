import os
from torch.utils.data.datapipes.iter import FileLister, Shuffler, IterableWrapper
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torchdata.datapipes.iter import FileLister, InMemoryCacheHolder
from torch.utils.data.datapipes._decorator import functional_datapipe
import pickle


@functional_datapipe("unstack_pickv2")
class UnstackPickles(IterDataPipe):
    def __init__(self, dp, keep_fields=None) -> None:
        super().__init__()
        self.dp = dp
        self.keep_fields = keep_fields

    def __iter__(self):
        for X in self.dp:
            # print('loading... ',X)
            with open(X, "rb") as f:
                raw_rows = pickle.load(f)
            if not self.keep_fields is None:
                yield [
                    {key: row[key] for key in row if key in self.keep_fields}
                    for row in raw_rows
                ]
            else:
                yield raw_rows


def get_dist_pipe(data_path, cache_mask=["*.pkl"], keep_fields=["smiles"]):
    pipe = (
        FileLister(
            root=data_path,
            recursive=False,
            masks=cache_mask,
        )
        .shuffle()
        .unstack_pickv2(keep_fields=keep_fields)
        .unbatch()
        .sharding_filter()
        .shuffle(buffer_size=100_000)
        # .in_memory_cache(size=50_000)
    )
    return pipe
