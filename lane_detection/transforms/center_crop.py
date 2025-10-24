from typing import Type

import jax
from flax.struct import dataclass, field
from jax import Array

from lane_detection.transforms.base import Transform


@dataclass
class CenterCrop(Transform):
    """Center crop image transform."""
    size: tuple[int] = field(pytree_node=False)

    @classmethod
    def create(
        cls: Type["CenterCrop"],
        size: tuple[int, int] | int,
    ) -> "CenterCrop":
        """Create a new CenterCrop transform.
        
        Args:
            size: Tuple (H, W) of desired output size or single int if square image.
        """
        size = (size, size) if isinstance(size, int) else tuple(size)
        return cls(size)
    
    @jax.jit
    def __call__(self, x: Array) -> Array:
        """Crop the given image at the center.
        
        Args:
            x: (B, H, W, C) array of input images.
        
        Returns:
            (B, H_size, W_size, C) array of output cropped images.
        """
        assert x.ndim == 4
        h_in, w_in = x.shape[1:3]
        h_out, w_out = self.size
        assert h_in >= h_out and w_in >= w_out
        
        top = int(round((h_in - h_out) / 2.0))
        left = int(round((w_in - w_out) / 2.0))
        x = x[:, top: top + h_out, left: left + w_out, :]
        return x