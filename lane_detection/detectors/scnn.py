import jax
import jax.lax as lax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from numpy.typing import NDArray

from lane_detection.detectors.base import LaneDetector
from lane_detection.models.scnn import SCNN
from lane_detection.transforms.base import Transform
from lane_detection.transforms.identity import Identity


class SCNNLaneDetector(LaneDetector):
    """SCNN-based lane detector using Flax.
    
    Args:
        scnn: SCNN model to use for lane predicition.
        n_lanes: Maximum number of unique lanes that can exist in an input image.
        transform: Image transform applied to input RGB images.
    """

    def __init__(
        self,
        scnn: SCNN,
        n_lanes: int,
        transform: Transform = Identity.create(),
    ):
        self.scnn = scnn
        self.n_lanes = n_lanes
        self.transform = transform
        
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
            seg_mask, _ = lax.fori_loop(0, self.n_lanes, mask_out_lanes, (seg_mask, lane_not_ext))
            return seg_mask
        
        images = self.transform(images)
        out = predict_lanes(self.scnn, images)
        return out
