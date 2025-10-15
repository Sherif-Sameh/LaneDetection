from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class LaneDataset(ABC):
    """Base class for handling lane detection datasets.
    
    Args:
        path: Path to dataset directory.
        val: Ratio of samples allocated for validation.
        test: Ratio of samples allocated for testing.
    """

    def __init__(self, path: Path, val: float = 0.1, test: float = 0.1):
        self.paths = (path / "train", path / "val", path / "test")
        self.split = (1.0 - val - test, val, test)
        self.metadata = {}

        for p in self.paths:
            p.mkdir(parents=True, exist_ok=True)        
        assert np.isclose(sum(self.split), 1.0).item(), "Dataset split must sum to 1.0"
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples available in the loaded dataset."""
    
    @abstractmethod
    def __getitem__(self, idx: int | slice | tuple[int | slice, ...]) -> tuple[NDArray, ...]:
        """Retrieve data sample/s from the dataset and return them."""

    @abstractmethod
    def download(self) -> None:
        """Download and store the dataset if not already stored."""
    
    @abstractmethod
    def load(self, val: bool = False, test: bool = False) -> None:
        """Load the stored dataset to prepare for data reading.
        
        Args:
            val: Load the data allocated for validation instead of training.
            test: Load the data allocated for testing instead of training. If both val and test
                flags are set, then validation takes higher precedence over testing.
        """
    
    @abstractmethod
    def exists(self) -> bool:
        """Check whether the dataset already exists at its specified path or not.
        
        Returns:
            True if the dataset already exists and False otherwise.
        """