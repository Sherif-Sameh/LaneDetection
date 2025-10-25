import jax
import jax.lax as lax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array
from numpy.typing import NDArray
from tqdm import tqdm

from lane_detection.dataloaders.base import LaneDataloader
from lane_detection.detectors.base import LaneDetector
from lane_detection.models import SCNN
from lane_detection.transforms import Identity
from lane_detection.transforms.base import Transform


class SCNNLaneDetector(LaneDetector):
    """SCNN-based lane detector using Flax.
    
    Args:
        scnn: SCNN model to use for lane predicition.
        n_lanes: Maximum number of unique lanes that can exist in an input image.
        pretrained: Load pre-trained SCNN weights if they exist.
        transform: Image transform applied to input RGB images.
        target_transforms: Image transform applied to lane segmentation masks.
    """

    def __init__(
        self,
        scnn: SCNN,
        n_lanes: int,
        pretrained: bool = False,
        transform: Transform = Identity.create(),
        target_transform: Transform = Identity.create(),
    ):
        self.scnn = scnn
        self.n_lanes = n_lanes
        self.transform = transform
        self.target_transform = target_transform
        if pretrained:
            self.scnn.load()
    
    def detect_lanes(self, images: NDArray) -> Array:
        """Extract all lanes in the input images and return their correponding lane images.

        Args:
            images: (N, H, W, C) batch of input images.
        
        Returns:
            (N, H, W) batch of lane instance segmentation masks. 
        """
        @jax.jit
        def mask_out_lanes(i: int, inputs: tuple[Array, Array]) -> tuple[Array, Array]:
            seg_mask, lane_not_ext = inputs
            seg_mask = jnp.where(
                jnp.logical_and(lane_not_ext[..., i], seg_mask == (i + 1)), 0, seg_mask
            )
            return seg_mask, lane_not_ext
        
        @nnx.jit
        def predict_lanes(scnn: SCNN, images: Array) -> Array:
            logits_seg, logits_ext = scnn(images)
            seg_mask = jnp.argmax(logits_seg, axis=-1)
            lane_not_ext = nnx.sigmoid(logits_ext)[:, None, None, :] < 0.5
            seg_mask = lax.fori_loop(0, self.n_lanes, mask_out_lanes, (seg_mask, lane_not_ext))
            return seg_mask
        
        images = self.transform(images)
        out = predict_lanes(self.scnn, images)
        return out

    def train(
        self,
        dataloader: LaneDataloader,
        test_dataloader: LaneDataloader,
        n_epochs: int,
        learning_rate: float,
        weight_decay: float = 1e-4,
    ) -> dict[str, Array]:
        """Train SCNN for a number of epochs using lane detection dataset.
        
        Args:
            dataloader: Lane dataset training dataloader.
            test_dataloader: Lane dataset testing dataloader.
            n_epochs: Number of epochs to train for.
            learning_rate: Learning rate used by AdamW optimizer.
            weight_decay: Weight decay used by AdamW optimizer.
        
        Returns:
            dict
            - "train_loss": (N_epochs) Average training loss over each epoch.
            - "train_acc_seg": (N_epochs) Average training lane segmentation accuracy over each epoch.
            - "train_acc_ext": (N_epochs) Average training lane existence accuracy over each epoch.
            - "test_loss": (N_epochs) Average testing loss over each epoch.
            - "test_acc_seg": (N_epochs) Average testing lane segmentation accuracy over each epoch.
            - "train_acc_ext": (N_epochs) Average testing lane existence accuracy over each epoch.
        """
        optimizer = nnx.Optimizer(
            self.scnn, optax.adamw(learning_rate, weight_decay=weight_decay), wrt=nnx.Param
        )
        metrics_history = {
            "train_loss": [],
            "train_acc_seg": [],
            "train_acc_ext": [],
            "test_loss": [],
            "test_acc_seg": [],
            "test_acc_ext": [],
        }
        metrics = {
            "loss": nnx.metrics.Average(argname="loss"),
            "acc_seg": nnx.metrics.Accuracy(),
            "acc_ext": nnx.metrics.Accuracy(),
        }
        for _ in tqdm(range(n_epochs), desc="Epochs"):
            # Train and record training metrics histories
            self.scnn.train()
            for input, target_seg in tqdm(dataloader, desc="Training"):
                input, target_seg = self.transform(input), self.target_transform(target_seg)
                target_ext = self._get_target_ext(target_seg, self.n_lanes)
                self._train_step(self.scnn, optimizer, metrics, input, target_seg, target_ext)
            for key, metric in metrics.items():
                metrics_history[f"train_{key}"].append(metric.compute())
                metric.reset()
            
            # Test and record testing metrics histories
            self.scnn.eval()
            for input, target_seg in tqdm(test_dataloader, desc="Testing"):
                input, target_seg = self.transform(input), self.target_transform(target_seg)
                target_ext = self._get_target_ext(target_seg, self.n_lanes)
                self._eval_step(self.scnn, metrics, input, target_seg, target_ext)
            for key, metric in metrics.items():
                metrics_history[f"test_{key}"].append(metric.compute())
                metric.reset()
        return metrics_history

    @jax.jit
    @staticmethod
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
        target_ext = lax.fori_loop(0, n_lanes, set_lane_ext, (target_seg, target_ext))
        return target_ext

    @nnx.jit
    @staticmethod
    def _train_step(
        scnn: SCNN,
        optimizer: nnx.Optimizer,
        metrics: dict[str, nnx.Metric],
        input: Array,
        target_seg: Array,
        target_ext: Array,
    ) -> None:
        """Perform a single training step using given batch.
        
        Args:
            scnn: SCNN lane prediction model.
            optimizer: NNX Optax optimizer.
            metrics: Dict of NNX metrics for loss, segmentation accuracy and existence accuracy.
            input: (B, H, W, 3) input array of RGB images.
            target_seg: (B, H, W) target int array for lane instance segmentation task.
            target_ext: (B, N_lanes) target binary {0, 1} array for lane existence prediction task.
        """
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits_seg, logits_ext), grads = grad_fn(scnn, input, target_seg, target_ext)
        metrics["loss"].update(loss=loss)
        metrics["acc_seg"].update(logits=logits_seg, labels=target_seg)
        metrics["acc_ext"].update(logits=logits_ext, labels=target_ext)
        optimizer.update(grads)
    
    @nnx.jit
    @staticmethod
    def _eval_step(
        scnn: SCNN,
        metrics: dict[str, nnx.Metric],
        input: Array,
        target_seg: Array,
        target_ext: Array,
    ) -> None:
        """Perform a single evaluation step using given batch.
        
            scnn: SCNN lane prediction model.
            metrics: Dict of NNX metrics for loss, segmentation accuracy and existence accuracy.
            input: (B, H, W, 3) input array of RGB images.
            target_seg: (B, H, W) target int array for lane instance segmentation task.
            target_ext: (B, N_lanes) target binary {0, 1} array for lane existence prediction task.
        """
        loss, logits_seg, logits_ext = loss_fn(scnn, input, target_seg, target_ext)
        metrics["loss"].update(loss=loss)
        metrics["acc_seg"].update(logits=logits_seg, labels=target_seg)
        metrics["acc_ext"].update(logits=logits_ext, labels=target_ext)


def loss_fn(
    scnn: SCNN,
    input: Array,
    target_seg: Array,
    target_ext: Array,
) -> tuple[Array, Array, Array]:
    """Compute the combined loss function for SCNN lane predicition. The combined loss is made
    up of a weighted sum of two losses: CE (segmentation) + BCE (existence).
    
    Args:
        scnn: SCNN lane prediction model.
        input: (B, H, W, 3) input array of RGB images.
        target_seg: (B, H, W) target int array for lane instance segmentation task.
        target_ext: (B, N_lanes) target binary {0, 1} array for lane existence prediction task.
        
    Returns:
        tuple:
        - loss: Combined loss function.
        - logits_seg: (B, H, W, N_lanes + 1) array of model's lane segmentation logits.  
        - logits_ext: (B, N_lanes) array of model's lane existence binary logits.  
        
    """
    logits_seg, logits_ext = scnn(input)
    loss_seg = optax.losses.softmax_cross_entropy_with_integer_labels(
        logits_seg, target_seg
    ).mean()
    loss_ext = optax.losses.sigmoid_binary_cross_entropy(logits_ext, target_ext).mean()
    
    LOSS_SEG_W, LOSS_EXT_W = 1.0, 0.1 
    loss = LOSS_SEG_W * loss_seg + LOSS_EXT_W * loss_ext
    return loss, logits_seg, logits_ext