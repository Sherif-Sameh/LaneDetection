from functools import singledispatchmethod
from typing import Any, Type

import jax
from flax.struct import dataclass
from jax import Array
from numpy import ndarray
from PIL.Image import Image

from lane_detection.transforms.base import Transform
from lane_detection.transforms.to_array import ToArray, ToArrayMask


@dataclass
class Compose(Transform):
    """Compose transform."""
    transforms: list[Transform]

    @classmethod
    def create(cls: Type["Compose"], transforms: list[Transform]) -> "Compose":
        """Create a new Compose transform made up of several transforms chained together.
        
        Args:
            transforms: List of transforms to apply to inputs sequentially.
        """
        return cls(transforms)
    
    @singledispatchmethod
    def __call__(self, x: Any) -> Array:
        """Apply all transforms to input sequentially.
        
        If the input is a PIL Image, the first transform is expected to be a `to_array` transform.

        Args:
            x: Input PIL Image/NumPy array/JAX array to be transformed.
        
        Returns:
            Transformed output image array after all transforms have been applied.
        """
        raise NotImplementedError(f"Invalid input type {type(x)}.")
    
    @__call__.register
    def _(self, x: Image) -> Array:
        assert isinstance(self.transforms[0], [ToArray, ToArrayMask]), \
            "If input is a PIL Image, first transform must be ToArray() or ToArrayMask()."
        
        @jax.jit
        def apply_transforms(x: Array) -> Array:
            for tf in self.transforms[1:]:
                x = tf(x)
            return x
        
        x = apply_transforms(x)
        return x
    
    @jax.jit
    @__call__.register
    def _(self, x: Array | ndarray) -> Array:
        for tf in self.transforms:
            x = tf(x)        
        return x
