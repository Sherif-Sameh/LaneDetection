from typing import Callable
from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array
from tqdm import tqdm

from lane_detection.dataloaders.base import LaneDataloader
from lane_detection.models.scnn import SCNN
from lane_detection.trainers.base import Trainer
from lane_detection.transforms.base import Transform
from lane_detection.transforms.identity import Identity

BatchType = tuple[Array, Array, Array]              # (input, target_seg, target_ext)
LossReturnType = tuple[Array, tuple[Array, Array]]  # (loss, (logits_seg, logits_ext))


class SCNNTrainer(Trainer):
    """Trainer for SCNN model for lane detection.
    
    Args:
        model: SCNN model to train.
        n_lanes: Maximum number of unique lanes that can exist in an input image.
        loss_seg_weight: Weight for lane instance segmentation cross-entropy loss.
        loss_ext_weight: Weight for lane existence prediction binary cross-entropy loss.
        transform: Image transform applied to input RGB images.
        target_transforms: Image transform applied to lane segmentation masks.
    """

    def __init__(
        self,
        model: SCNN,
        n_lanes: int,
        loss_seg_weight: float = 1.0,
        loss_ext_weight: float = 0.1,
        transform: Transform = Identity.create(),
        target_transform: Transform = Identity.create(),
    ):
        super().__init__(model)
        self.n_lanes = n_lanes
        self.transform = transform
        self.target_transform = target_transform
        self.metrics = {
            "loss": nnx.metrics.Average(),
            "acc_seg": nnx.metrics.Accuracy(),
            "acc_ext": nnx.metrics.Accuracy(threshold=0.5),
        }
        SCNNTrainer._loss_fn = staticmethod(self._make_loss_fn(loss_seg_weight, loss_ext_weight))
    
    def train(
        self,
        dataloader: LaneDataloader,
        test_dataloader: LaneDataloader,
        n_epochs: int,
        learning_rate: float,
        weight_decay: float = 1e-4,
    ) -> dict[str, Array]:
        """Train and evaluate SCNN for a number of epochs using lane detection dataset.
        
        Args:
            dataloader: Dataloader for lane training dataset.
            test_dataloader: Dataloader for lane testing dataset.
            n_epochs: Number of epochs to train for.
            learning_rate: Learning rate used by AdamW optimizer.
            weight_decay: Weight decay used by AdamW optimizer.
        
        Returns:
            dict
            - "train_loss": Average training loss over each epoch.
            - "train_acc_seg": Average training lane segmentation accuracy over each epoch.
            - "train_acc_ext": Average training lane existence accuracy over each epoch.
            - "test_loss": Average testing loss over each epoch.
            - "test_acc_seg": Average testing lane segmentation accuracy over each epoch.
            - "train_acc_ext": Average testing lane existence accuracy over each epoch.
        """
        optimizer = nnx.Optimizer(
            self.model, optax.adamw(learning_rate, weight_decay=weight_decay), wrt=nnx.Param
        )
        metrics_history = {"train_loss": [], "train_acc_seg": [], "train_acc_ext": [], 
                           "test_loss": [], "test_acc_seg": [], "test_acc_ext": []}
        for metric in self.metrics.values():
            metric.reset()
         
        for _ in tqdm(range(n_epochs), desc="Epochs"):
            # Train and record training metrics histories
            self.model.train()
            for input, target_seg in dataloader:
                input, target_seg = self.transform(input), self.target_transform(target_seg)
                target_ext = self._get_target_ext(target_seg, self.n_lanes)
                self._train_step(self.model, optimizer, self.metrics, (input, target_seg, target_ext))
            self._update_and_report_metrics(metrics_history, train=True)
            
            # Test and record testing metrics histories
            self.model.eval()
            for input, target_seg in test_dataloader:
                input, target_seg = self.transform(input), self.target_transform(target_seg)
                target_ext = self._get_target_ext(target_seg, self.n_lanes)
                self._eval_step(self.model, self.metrics, (input, target_seg, target_ext))
            self._update_and_report_metrics(metrics_history, train=False)
            self.model.save()
        
        for key, metric_history in metrics_history.items():
            metrics_history[key] = jnp.stack(metric_history)
        return metrics_history
    
    def _update_and_report_metrics(
        self,
        metrics_history: dict[str, list[Array]],
        train: bool = True,
    ) -> None:
        """Update metrics history and report their latest values then reset all metrics."""
        prefix = "train" if train else "test"
        for key, metric in self.metrics.items():
            metrics_history[f"{prefix}_{key}"].append(metric.compute())
            metric.reset()
        epoch = len(metrics_history[f"{prefix}_{key}"])
        if train: 
            print(f"Epoch {epoch}\n")
        print(f"\t{prefix.title()} Metrics:") 
        for key, metrics in metrics_history.items():
            if prefix in key:
                print(f"\t\t{key.replace('_', ' ').title()}: {metrics[-1].item():.4f}")
        print("")

    @staticmethod
    @nnx.jit
    def _train_step(
        model: SCNN,
        optimizer: nnx.Optimizer,
        metrics: dict[str, nnx.Metric],
        batch: BatchType,
    ) -> None:
        """Perform a single training step using given batch.
        
        Args:
            model: SCNN lane prediction model.
            optimizer: NNX Optax optimizer.
            metrics: Dict of NNX metrics for loss, segmentation accuracy and existence accuracy.
            batch: Tuple of input rgb images, segmentation target (int labels) and existence
                target (binary labels {0, 1}).
        """
        grad_fn = nnx.value_and_grad(SCNNTrainer._loss_fn, has_aux=True)
        (loss, (logits_seg, logits_ext)), grads = grad_fn(model, batch)
        metrics["loss"].update(values=loss)
        metrics["acc_seg"].update(logits=logits_seg, labels=batch[1])
        metrics["acc_ext"].update(logits=logits_ext, labels=batch[2])
        optimizer.update(model, grads)
        
    @staticmethod
    @nnx.jit
    def _eval_step(
        model: SCNN,
        metrics: dict[str, nnx.Metric],
        batch: BatchType,
    ) -> None:
        """Perform a single evaluation step using given batch.
        
        Args:
            model: SCNN lane prediction model.
            metrics: Dict of NNX metrics for loss, segmentation accuracy and existence accuracy.
            batch: Tuple of input rgb images, segmentation target (int labels) and existence
                target (binary labels {0, 1}).
        """
        loss, (logits_seg, logits_ext) = SCNNTrainer._loss_fn(model, batch)
        metrics["loss"].update(values=loss)
        metrics["acc_seg"].update(logits=logits_seg, labels=batch[1])
        metrics["acc_ext"].update(logits=logits_ext, labels=batch[2])

    @staticmethod
    def _make_loss_fn(
        loss_seg_weight: float,
        loss_ext_weight: float,
    ) -> Callable[[SCNN, BatchType], LossReturnType]:
        """Contructs and returns a combined loss function for lane segmentation and existence
        prediction that is parameterized by the given weights for combining the two losses.
        
        Args:
            loss_seg_weight: Weight for lane instance segmentation cross-entropy loss.
            loss_ext_weight: Weight for lane existence prediction binary cross-entropy loss.
        
        Returns:
            Closure over loss function parameterized with given loss weights.
        """
        def _loss_fn(model: SCNN, batch: BatchType) -> LossReturnType:
            """Compute the combined loss function for SCNN lane predicition. The combined loss is made
            up of a weighted sum of two losses: CE (segmentation) + BCE (existence).
            
            Args:
                model: SCNN lane prediction model.
                batch: Tuple of input rgb images, segmentation target (int labels) and existence
                    target (binary labels {0, 1}).
            Returns:
                Tuple of combined loss function, segmentation logits and existence binary logits.
            """
            input, target_seg, target_ext = batch
            logits_seg, logits_ext = model(input)
            loss_seg = optax.losses.softmax_cross_entropy_with_integer_labels(
                logits_seg, target_seg
            ).mean()
            loss_ext = optax.losses.sigmoid_binary_cross_entropy(logits_ext, target_ext).mean()           
            loss = loss_seg_weight * loss_seg + loss_ext_weight * loss_ext
            return loss, (logits_seg, logits_ext)
        
        return _loss_fn
    
    @staticmethod
    @partial(jax.jit, static_argnames=["n_lanes"])
    def _get_target_ext(target_seg: Array, n_lanes: int) -> Array:
        """Compute lane existence targets from segmentation targets.
        
        Args:
            target_seg: (B, H, W) target int array for lane instance segmentation task.
            n_lanes: Maximum number of unique lanes that can exist in an input image.
        
        Returns:
            (B, N_lanes) target binary {0, 1} array for lane existence prediction task.
        """
        def set_lane_ext(i: int, targets: tuple[Array, Array]) -> tuple[Array, Array]:
            target_seg, target_ext = targets
            target_ext = target_ext.at[:, i].set(jnp.any(target_seg == (i + 1), axis=(1, 2)))
            return target_seg, target_ext
        
        target_ext = jnp.zeros((target_seg.shape[0], n_lanes), dtype=jnp.int32)
        _, target_ext = lax.fori_loop(0, n_lanes, set_lane_ext, (target_seg, target_ext))
        return target_ext
