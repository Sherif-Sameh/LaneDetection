from typing import Type

import jax
from flax.struct import dataclass
from jax import Array

from lane_detection.transforms.base import Transform


@dataclass
class Identity(Transform):
    """Identity image transform."""

    @classmethod
    def create(
        cls: Type["Identity"],
    ) -> "Identity":
        """Create a new Identity transform."""
        return cls()
    
    @jax.jit
    def __call__(self, x: Array) -> Array:
        """Return the input image unchanged."""
        return x