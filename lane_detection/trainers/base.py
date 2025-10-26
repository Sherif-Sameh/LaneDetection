from abc import ABC, abstractmethod

from flax import nnx
from jax import Array

class Trainer(ABC):
    """Base abstract class for model trainers."""

    def __init__(self, model: nnx.Module):
        self.model = model

    @abstractmethod
    def train(self, dataloader, test_dataloader, n_epochs, learning_rate) -> dict[str, Array]:
        """Train and evaluate model for a number of epochs.
        
        Args:
            dataloader: Dataloader for training dataset.
            test_dataloader: Dataloader for testing dataset.
            n_epochs: Number of epochs to train for.
            learning_rate: Learning rate used by optimizer.
        
        Returns:
            Dict containing the history of all tracked metrics for training and testing. 
        """
    
    @staticmethod
    @abstractmethod
    def _train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        metrics: dict[str, nnx.Metric],
        batch: tuple[Array, ...],
    ) -> None:
        """Perform a single training step using given batch.
        
        Args:
            model: NNX module to train.
            optimizer: NNX Optax optimizer.
            metrics: Dict of NNX metrics to track during training step.
            batch: A tuple of arbitrary length of model input/s and target/s.
        """
    
    @staticmethod
    @abstractmethod
    def _eval_step(
        model: nnx.Module,
        metrics: dict[str, nnx.Metric],
        batch: tuple[Array, ...],
    ) -> None:
        """Perform a single evaluation step using given batch.
        
        Args:
            model: NNX module to train.
            metrics: Dict of NNX metrics to track during training step.
            batch: A tuple of arbitrary length of model input/s and target/s.
        """
    
    @staticmethod
    @abstractmethod
    def _loss_fn(model: nnx.Module, batch: tuple[Array, ...]) -> tuple[Array, ...]:
        """Compute loss function for model using given batch and return loss + auxiliary values.
        
        Args:
            model: NNX module to evaluate loss for.
            batch: A tuple of arbitrary length of model input/s and target/s.
        
        Returns:
            Tuple of arbitrary length where the first element **must** be the loss function,
                followed by auxiliary arrays used for tracking metrics.
        """