from torch.utils.data.dataset import Dataset
from collections.abc import Sequence

class ZipDataset(Dataset):
    """
    Dataset to zip up multiple datasets in parallel.
    """
    def __init__(self, *datasets, flatten=True, shortest=False):
        super(ZipDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.flatten = flatten
        self.size = min(len(d) for d in self.datasets)
        if not shortest and max(len(d) for d in self.datasets) > self.size:
            raise ValueError('datasets are not all the same size')

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not self.flatten:
            return tuple(ds[index] for ds in self.datasets)
        return sum([tuple(d) if isinstance(d, Sequence) else (d,)
                    for d in [ds[index] for ds in self.datasets]], ())
