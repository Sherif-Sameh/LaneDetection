from abc import ABC, abstractmethod
from pathlib import Path

from numpy.typing import NDArray


class LaneDataset(ABC):
    """Base class for handling lane detection datasets.
    
    Args:
        path: Path to dataset directory.
    """

    def __init__(self, path: Path):
        self.path = path
        self.metadata = {}
    
    @abstractmethod
    def download(self) -> None:
        """Download and store the dataset if not already stored."""
    
    @abstractmethod
    def load(self, n_samples: int | None = None) -> tuple[NDArray, NDArray]:
        """Load the stored dataset and return its data and labels.
        
        Args:
            n_samples: Number of samples to load. If None, then all dataset is loaded.
        
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