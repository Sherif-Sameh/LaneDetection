from functools import singledispatchmethod
from typing import Any, Type

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from jax import Array
from jax.core import Tracer
from PIL.Image import Image

from lane_detection.transforms.base import Transform


@dataclass
class ToArray(Transform):
    """Convert to JAX Array image transform."""

    @classmethod
    def create(
        cls: Type["ToArray"],
    ) -> "ToArray":
        """Create a new ToArray transform."""
        return cls()
    
    @singledispatchmethod
    def __call__(self, x: Any) -> Array:
        """Convert input image to a JAX float Array from PIL Image or NumPy Array."""
        raise NotImplementedError(f"Invalid input type {type(x)}.")
    
    @__call__.register
    def _(self, x: Image) -> Array:
        """Convert input image to a JAX float Array from PIL Image.
        
        Args:
            x: PIL Image to convert to an array.
        
        Returns:
            (H, W, C) float array image with intensity [0.0, 1.0].
        """
        x = jnp.asarray(x, dtype=jnp.float32) / 255
        return x
    
    @jax.jit
    @__call__.register
    def _(self, x: np.ndarray | Tracer) -> Array:
        """Convert input image to a JAX float Array from NumPy Array.
        
        Args:
            x: (H, W, C) or (B, H, W, C) NumPy array of input image/s.
        
        Returns:
            (H, W, C) or (B, H, W, C) float array image/s with intensity [0.0, 1.0].
        """
        dtype = x.dtype
        x = jnp.array(x, dtype=jnp.float32)
        x = x / 255 if dtype == np.uint8 else x
        return x


@dataclass
class ToArrayMask(Transform):
    """Convert to JAX Array Mask transform.
    
    Unlike `ToArray`, this transform does not scale the input to [0.0, 1.0]. The input remains
    an integer array to allow for use as a segmentation mask.
    """

    @classmethod
    def create(
        cls: Type["ToArrayMask"],
    ) -> "ToArrayMask":
        """Create a new ToArray transform."""
        return cls()
    
    @singledispatchmethod
    def __call__(self, x: Any) -> Array:
        """Convert input segmentation mask to a JAX int Array from PIL Image or NumPy Array."""
        raise NotImplementedError(f"Invalid input type {type(x)}.")
    
    @__call__.register
    def _(self, x: Image) -> Array:
        """Convert input segmentation mask to a JAX int Array from PIL Image.
        
        Args:
            x: PIL Image of segmentation mask to convert to an array.
        
        Returns:
            (H, W) int array segmentation mask.
        """
        x = jnp.asarray(x, dtype=jnp.int32)
        return x
    
    @jax.jit
    @__call__.register
    def _(self, x: np.ndarray | Tracer) -> Array:
        """Convert input segmentation mask to a JAX int Array from NumPy Array.
        
        Args:
            x: (H, W) or (B, H, W) NumPy array of segmentation mask/s.
        
        Returns:
            (H, W) or (B, H, W) int array of segmentation mask/s.
        """
        x = jnp.array(x, dtype=jnp.int32)
        return x
