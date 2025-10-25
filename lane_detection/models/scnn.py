import warnings
from functools import partial
from pathlib import Path

import jax.image as image
import jax.lax as lax
import orbax.checkpoint as ocp
from flax import nnx
from jax import Array
from jax.image import ResizeMethod

from lane_detection.models import VGG16


class MessagePassing(nnx.Module):
    """Message passing layer from SCNN.
    
    Args:
        in_features: Number of features (channels) of inputs.
        kernel_size: Kernel size for 1D convolution used in message passing.
        vertical: Apply message passing along vertical direction. Horizontal if false.
        reverse: Apply message passing in reverse direction. Forward if false.
        rngs: RNG state to use for layer weight initialization.
    """

    def __init__(
        self,
        in_features: int,
        kernel_size: int,
        *,
        vertical: bool,
        reverse: bool,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.vertical = vertical
        self.reverse = reverse
        self.conv = nnx.Conv(in_features, in_features, kernel_size, use_bias=False, rngs=rngs)
    
    def __call__(self, x: Array) -> Array:
        """Apply message passing algorithm to input feature maps.
        
        Args:
            x: (B, H, W, C) input feature maps.
        
        Returns:
            (B, H, W, C) output feature maps after message passing specified direction.
        """
        assert x.ndim == 4, f"Expected batched 4-dim inputs, got {x.ndim}-dim."
        assert x.shape[3] == self.in_features, \
            f"Expected {self.in_features} features, got {x.shape[3]}"
        x = lax.cond(
            self.vertical,
            self._message_passing_vertical,
            self._message_passing_horizontal,
            x,
        )
        return x
    
    @nnx.jit
    def _message_passing_vertical(self, x: Array) -> Array:
        def forward(i: int, input: tuple[nnx.Conv, Array]) -> Array:
            conv, x = input
            y = x.at[:, i].set(x[:, i] + nnx.relu(conv(x[:, i - 1])))
            return conv, y
        
        h_in = x.shape[1]
        x = x[:, ::-1] if self.reverse else x
        _, x = nnx.fori_loop(1, h_in, forward, (self.conv, x))
        x = x[:, ::-1] if self.reverse else x
        return x
    
    @nnx.jit
    def _message_passing_horizontal(self, x: Array) -> Array:
        def forward(j: int, input: tuple[nnx.Conv, Array]) -> Array:
            conv, x = input
            y = x.at[:, :, j].set(x[:, :, j] + nnx.relu(conv(x[:, :, j - 1])))
            return conv, y
        
        w_in = x.shape[2]
        x = x[:, :, ::-1] if self.reverse else x
        _, x = nnx.fori_loop(1, w_in, forward, (self.conv, x))
        x = x[:, :, ::-1] if self.reverse else x
        return x


class SCNN(nnx.Module):
    """Spatial CNN from `Spatial As Deep: Spatial CNN for Traffic Scene Understanding`.
    
    Paper: <https://arxiv.org/abs/1712.06080>
    
    Args:
        input_size: int tuple (H, W) of input image spatial dimensions.
        out_features: Number of output features from passed-in backbone.
        ms_kernel_size: Kernel size for 1D convolution used in message passing layers.
        n_lanes: Maximum number of unique lanes that can exist in an input image.
        backbone: Feature extractor backbone module for processing input image.
        rngs: RNG state to use for layer weight initialization.        
    """
    
    def __init__(
        self,
        input_size: tuple[int, int],
        out_features: int,
        ms_kernel_size: int,
        n_lanes: int,
        backbone: nnx.Module | None = None,
        *
        rngs: nnx.Rngs,
    ):
        h_in, w_in = input_size
        self.in_shape = (h_in, w_in, 3)
        # Set backbone feature extractor
        if backbone is None:
            out_features = 512
            self.backbone = self._get_default_backbone(rngs=rngs)
        else:
            self.backbone = backbone

        # Add additional convolutional block after backbone
        self.conv = nnx.Sequential(
            nnx.Conv(
                out_features,
                1024,
                kernel_size=(3, 3),
                padding=(4, 4),
                kernel_dilation=(4, 4),
                use_bias=False,
                rngs=rngs,
            ),
            nnx.BatchNorm(1024, rngs=rngs),
            nnx.relu,
            nnx.Conv(1024, 128, kernel_size=(1, 1), use_bias=False, rngs=rngs),
            nnx.BatchNorm(128, rngs=rngs),
            nnx.relu,
        )

        # Add SCNN message passing layers in all 4 directions
        self.message_passing = nnx.Sequential(
            MessagePassing(128, ms_kernel_size, vertical=True, reverse=False, rngs=rngs),
            MessagePassing(128, ms_kernel_size, vertical=True, reverse=True, rngs=rngs),
            MessagePassing(128, ms_kernel_size, vertical=False, reverse=False, rngs=rngs),
            MessagePassing(128, ms_kernel_size, vertical=False, reverse=True, rngs=rngs),
        )

        # Add lane instance segmentation head
        self.lane_seg_head = nnx.Sequential(
            nnx.Dropout(0.1, rngs=rngs),
            nnx.Conv(128, n_lanes + 1, kernel_size=(1, 1), rngs=rngs),
        )

        # Add lane existence prediction head
        fc_in_features = int(h_in / 16) * int(w_in / 16) * 5 
        self.lane_ext_head1 = nnx.Sequential(
            nnx.softmax,
            partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2)),
        )
        self.lane_ext_head2 = nnx.Sequential(
            nnx.Linear(fc_in_features, 128, rngs=rngs),
            nnx.relu,
            nnx.Linear(128, n_lanes, rngs=rngs),
        )
    
    def __call__(self, x: Array) -> tuple[Array, Array]:
        """Forward propagate input through SCNN model to predict lane instance segmentation logits
        and existence binary logits.

        Args:
            x: (B, H, W, 3) input array of RGB images.
        
        Returns:
            tuple
            - logits_seg: (B, H, W, N_lanes + 1) array of lane instance segmentation maps with
                pixel-wise lane logits (i.e. pre-softmax).
            - logits_ext: (B, N_lanes) array of image-wide lane existence binary logits
                (i.e. pre-sigmoid).
        """
        assert x.ndim == 4, f"Expected batched 4-dim inputs, got {x.ndim}-dim."
        assert x.shape[1:] == self.in_shape, \
            f"Expected image shape {self.in_shape}, got{x.shape[1:]}"
        # Extract image features
        x = self.backbone(x)
        x = self.conv(x)
        x = self.message_passing(x)

        # Predict pixel-wise instance segmentation logits (i.e. pre-softmax)
        x = self.lane_seg_head(x)
        b_x, h_x, w_x, f_x = x.shape
        logits_seg = image.resize(x, (b_x, h_x * 8, w_x * 8, f_x), method=ResizeMethod.LINEAR)

        # Predict image-wise lane existence binary logits (i.e. pre-sigmoid)
        x = self.lane_ext_head1(x)
        x = x.reshape(x.shape[0], -1)  # Flatten for FC layers
        logits_ext = self.lane_ext_head2(x)
        return logits_seg, logits_ext

    def save(self) -> None:
        """Save the current model's weights using Orbax."""
        ckpt_path = Path(__file__).parent / "weights/scnn"
        if ckpt_path.exists():
            ckpt_path.unlink()
        _, state = nnx.split(self)
        pure_dict_state = nnx.to_pure_dict(state)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(ckpt_path, pure_dict_state)
    
    def load(self) -> None:
        """Load saved model weights using Orbax if they exist."""
        ckpt_path = Path(__file__).parent / "weights/scnn"
        if not ckpt_path.exists():
            warnings.warn(f"No checkpoints found for SCNN at {ckpt_path.parent}.")
            return
        checkpointer = ocp.StandardCheckpointer()
        pure_dict_state = checkpointer.restore(ckpt_path)
        abstract_model = nnx.eval_shape(self)
        graphdef, abstract_state = nnx.split(abstract_model)
        nnx.replace_by_pure_dict(abstract_state, pure_dict_state)
        self = nnx.merge(graphdef, abstract_state)
    
    def _get_default_backbone(self, *, rngs: nnx.Rngs) -> nnx.Module:
        """Get default SCNN VGG16-BN backbone with pre-trained ImageNet weights.
        
        Returns:
            VGG16-BN feature extractor backbone module after processing.
        """
        # Load pre-trained VGG16-BN backbone
        backbone = VGG16(
            batch_norm=True, pretrained=True, rngs=nnx.Rngs(0)
        ).features

        # Replace convolutions in last block with dilated convolutions
        for i, layer in enumerate(backbone[-1].layers):
            if not isinstance(layer, nnx.Conv):
                continue
            backbone[-1].layers[i] = nnx.Conv(
                layer.in_features,
                layer.out_features,
                kernel_size=(3, 3),
                padding=(2, 2),
                kernel_dilation=(2, 2),
                use_bias=layer.use_bias,
                rngs=rngs,
            )
        
        # Remove the max pool layers at the end of blocks 4 and 5
        backbone[-2].layers.pop(-1)
        backbone[-1].layers.pop(-1)
        return backbone
