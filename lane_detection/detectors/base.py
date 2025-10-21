from abc import ABC, abstractmethod

from jax.typing import ArrayLike

class LaneDetector(ABC):
    """Base class for lane detectors."""

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def detect_lanes(self, images: ArrayLike) -> ArrayLike:
        """Extract all lanes in the input images and return their correponding lane images.

        
        Note: The detector can only classify pixels as lane/background or perform full lane
        instance segmentation, identifying each unique lane separately.

        Args:
            images: (N, H, W, C) batch of input images.
        
        Returns:
            (N, H, W, 1) batch of lane images, where each pixel is labeled as either background (0)
            or belonging to a lane (>0).
        """