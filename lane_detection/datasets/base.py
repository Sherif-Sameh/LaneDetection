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
    def download(self) -> None:
        """Download and store the dataset if not already stored."""
    
    @abstractmethod
    def load(self, val: bool = False, test: bool = False) -> tuple[NDArray, NDArray]:
        """Load the stored dataset and return its data and labels.
        
        Args:
            val: Load the data allocated for validation instead of training.
            test: Load the data allocated for testing instead of training. If both val and test
                flags are set, then validation takes higher precedence over testing.
        
        Returns:
            tuple
            - data: array containing all raw data in the dataset.
            - labels: array containing all labels in the dataset.
        """
    
    @abstractmethod
    def exists(self) -> bool:
        """Check whether the dataset already exists at its specified path or not.
        
        Returns:
            True if the dataset already exists and False otherwise.
        """