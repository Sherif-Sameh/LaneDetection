import math
from abc import ABC, abstractmethod
from typing import Self

from jax.typing import ArrayLike

from lane_detection.datasets.base import LaneDataset


class LaneDataloader(ABC):
    """Base class for dataloaders for lane datasets.
    
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
        assert len(dataset) > 0, \
            "Dataset is empty. Ensure it's loaded before initializing dataloader."
        assert batch_size > 0, f"Batch size must be > 0, got {batch_size}"
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
    
    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)
        
    def __iter__(self) -> Self:
        return self
    
    @abstractmethod
    def __next__(self) -> tuple[ArrayLike, ...]:
        """Return the next batch of samples."""