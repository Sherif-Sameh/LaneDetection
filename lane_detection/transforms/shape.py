from collections.abc import Sequence
from typing import Type

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field
from jax import Array

from lane_detection.transforms.base import Transform


@dataclass
class Squeeze(Transform):
    """Squeeze transform."""
    axis: tuple[int, ...] = field(pytree_node=False)

    @classmethod
    def create(cls: Type["Squeeze"], axis: int | Sequence[int]) -> "Squeeze":
        """Create a new Squeeze transform.
        
        Args:
            axis: Int or sequence of ints specifying axes to remove.
        """
        axis = (axis,) if isinstance(axis, int) else tuple(axis)
        return cls(axis)
    
    @jax.jit
    def __call__(self, x: Array) -> Array:
        """Squeeze specified axes from input array.
        
        Args:
            x: Input array to be transformed.
        
        Returns:
            Transformed output array after specified axes have been removed.
        """
        return x.squeeze(axis=self.axis)


@dataclass
class Unsqueeze(Transform):
    """Unsqueeze transform."""
    axis: tuple[int, ...] = field(pytree_node=False)

    @classmethod
    def create(cls: Type["Unsqueeze"], axis: int | Sequence[int]) -> "Unsqueeze":
        """Create a new Unsqueeze transform.
        
        Args:
            axis: Int or sequence of ints specifying axes to add.
        """
        axis = (axis,) if isinstance(axis, int) else tuple(axis)
        return cls(axis)
    
    @jax.jit
    def __call__(self, x: Array) -> Array:
        """Squeeze specified axes into input array.
        
        Args:
            x: Input array to be transformed.
        
        Returns:
            Transformed output array after specified axes have been added.
        """
        return jnp.expand_dims(x, axis=self.axis)
