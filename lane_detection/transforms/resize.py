from typing import Type

import jax
import jax.image as image
from flax.struct import dataclass, field
from jax import Array
from jax.image import ResizeMethod

from lane_detection.transforms.base import Transform


@dataclass
class Resize(Transform):
    """Resize image transform."""
    size: tuple[int] = field(pytree_node=False)
    method: ResizeMethod = field(pytree_node=False)

    @classmethod
    def create(
        cls: Type["Resize"],
        size: tuple[int, int] | int,
        method: str | ResizeMethod = "bilinear",
    ) -> "Resize":
        """Create a new Resize transform.
        
        Args:
            size: Tuple (H, W) of desired output size or single int if square image.
            method: Method for resizing image. Defaults to binlinear interpolation.
        """
        size = (size, size) if isinstance(size, int) else size
        method = ResizeMethod.from_string(method) if isinstance(method, str) else method
        return cls(size, method)
    
    @jax.jit
    def __call__(self, x: Array) -> Array:
        """Resize an input image's spatial dimensions.
        
        Args:
            x: (B, H, W, C) array of input images.
        
        Returns:
            (B, H_size, W_size, C) array of output resized images.
        """
        assert x.ndim == 4

        h_out, w_out = self.size
        shape = (x.shape[0], h_out, w_out, x.shape[3])
        x = image.resize(x, shape, self.method)
        return x