import numpy as np
from numpy.typing import NDArray

from lane_detection.dataloaders.base import LaneDataloader
from lane_detection.datasets.base import LaneDataset


class NumPyLaneDataloader(LaneDataloader):
    """NumPy-based dataloader for lane datasets.
    
    Args:
        dataset: Lane dataset from which to load batches of data.
        batch_size: Number of samples to load from dataset at each iteration.
        shuffle: Shuffle the dataset at the beginning of every epoch.
        seed: Optional random seed to use for RNG if shuffling dataset.
    """

    def __init__(
        self,
        dataset: LaneDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int | None = None,
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, seed=seed)
        self.idx = 0

        # Get batch sample indices (for first epoch only if shuffle is True)
        if shuffle:
            self.rng = np.random.default_rng(seed=seed)
            self.batch_idxs = self._sample_batch_idxs()
        else:
            n_samples = len(dataset)
            self.batch_idxs = np.split(
                np.arange(n_samples), np.arange(batch_size, n_samples, batch_size)
            )
    
    def __next__(self) -> tuple[NDArray, ...]:
        """Return the next batch of samples."""
        if self.idx >= len(self.batch_idxs):
            if self.shuffle:
                self.batch_idxs = self._sample_batch_idxs()
            self.idx = 0
            raise StopIteration
        batch = self.dataset[self.batch_idxs[self.idx]]
        self.idx += 1
        return batch

    def _sample_batch_idxs(self) -> list[NDArray[np.int64]]:
        """Sample a new set of random batch indices for accessing samples in the dataset.
        
        Return:
            List of arrays containing the indices of samples in each batch. All arrays have equal
            lengths except the last one which depends the size of the dataset and the batch size.
        """
        n_samples = len(self.dataset)
        idxs = self.rng.permutation(n_samples)
        return np.split(idxs, np.arange(self.batch_size, n_samples, self.batch_size))