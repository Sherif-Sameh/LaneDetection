from collections.abc import Sequence
from typing import Type

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jax import Array

from lane_detection.transforms.base import Transform


@dataclass
class Normalize(Transform):
    """Z-score normalization image transform."""
    mean: Array
    std: Array

    @classmethod
    def create(
        cls: Type["Array"],
        mean: Sequence[float],
        std: Sequence[float],
    ) -> "Array":
        """Create a new Normalize transform.
        
        Args:
            mean: Sequence of means for each channel.
            std: Sequence of standard deviations for each channel.
        """
        mean = jnp.array(mean, dtype=jnp.float32)
        std = jnp.array(std, dtype=jnp.float32)
        return cls(mean, std)
    
    @jax.jit
    def __call__(self, x: Array) -> Array:
        """Apply Z-score normalization to the input images.
        
        Args:
            x: (B, H, W, C) array of input images.
        
        Returns:
            (B, H, W, C) array of output normalized images.
        """
        assert x.ndim == 4
        
        x = (x - self.mean) / self.std
        return x