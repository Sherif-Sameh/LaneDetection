from abc import abstractmethod
from typing import Any, Type, Protocol

from jax import Array


class Transform(Protocol):
    """Base abstract transform class."""
    
    @classmethod
    @abstractmethod
    def create(cls: Type["Transform"], *args, **kwargs) -> "Transform":
        """Factory method for initializing transforms."""
    
    @abstractmethod
    def __call__(self, x: Any) -> Array:
        """Apply transform to input."""
