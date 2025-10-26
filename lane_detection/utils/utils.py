import urllib.request
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from flax import nnx
from torch import Tensor

import lane_detection.transforms as tf

def download_weights_from_url(url: str, file: Path) -> None:
    """Download the PyTorch model weights from URL."""
    out_dir = file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading weights...")
    urllib.request.urlretrieve(url, file)
    print(f"Downloaded weights into: {file}.")


def load_bn_from_torch(
    module: nnx.BatchNorm,
    mean: Tensor,
    var: Tensor,
    scale: Tensor | None,
    bias: Tensor | None,
) -> None:
    """Load a batch norm's weights/stats from PyTorch into NNX module.
    
    Args:
        module: NNX batch norm module to load weights/stats into.
        mean: (C) tensor of running mean stats.
        var: (C) tensor of running var stats.
        scale: Optional (C) tensor of bn scale weights.
        bias: Optional (C) tensor of bn bias weights.
    """
    assert tuple(mean.shape) == module.mean.value.shape
    assert tuple(var.shape) == module.var.value.shape
    if module.use_scale:
        assert scale is not None, "Scale mismatch between module and weights."
        assert tuple(scale.shape) == module.scale.value.shape
    if module.use_bias:
        assert bias is not None, "Bias mismatch between module and weights."
        assert tuple(bias.shape) == module.bias.value.shape
    
    dtype = module.mean.value.dtype
    device = module.mean.value.device
    module.mean.value = jnp.array(mean.cpu().numpy(), dtype=dtype, device=device)
    module.var.value = jnp.array(var.cpu().numpy(), dtype=dtype, device=device)
    if module.use_scale:
        module.scale.value = jnp.array(scale.cpu().numpy(), dtype=dtype, device=device)
    if module.use_bias:
        module.bias.value = jnp.array(bias.cpu().numpy(), dtype=dtype, device=device)


def load_conv_from_torch(module: nnx.Conv, kernel: Tensor, bias: Tensor | None) -> None:
    """Load a convolutional layer's weights from PyTorch into NNX module.
    
    Args:
        module: NNX convolutional module to load weights into.
        kernel: (C_out, C_in, K_h, K_w) tensor of conv kernel weights.
        bias: Optional (C_out) tensor of conv biases if supported.
    """
    kernel = kernel.permute(2, 3, 1, 0)  # Convert to NNX convention
    assert tuple(kernel.shape) == module.kernel.value.shape
    if module.use_bias:
        assert bias is not None, "Bias use mismatch between module and weights."
        assert tuple(bias.shape) == module.bias.value.shape
    
    dtype = module.kernel.value.dtype
    device = module.kernel.value.device
    module.kernel.value = jnp.array(kernel.cpu().numpy(), dtype=dtype, device=device)
    if bias is not None:
        module.bias.value = jnp.array(bias.cpu().numpy(), dtype=dtype, device=device)


def load_linear_from_torch(module: nnx.Linear, weight: Tensor, bias: Tensor | None) -> None:
    """Load a linear layer's weights from PyTorch into NNX module.
    
    Args:
        module: NNX linear module to load weights into.
        weight: (F_out, F_in) tensor of layer weights.
        bias: Optional (F_out) tensor of layer biases if supported.
    """
    weight = weight.transpose(0, 1)  # Convert to NNX convention
    assert tuple(weight.shape) == module.kernel.value.shape
    if module.use_bias:
        assert bias is not None, "Bias use mismatch between module and weights."
        assert tuple(bias.shape) == module.bias.value.shape
    
    dtype = module.kernel.value.dtype
    device = module.kernel.value.device
    module.kernel.value = jnp.array(weight.cpu().numpy(), dtype=dtype, device=device)
    if bias is not None:
        module.bias.value = jnp.array(bias.cpu().numpy(), dtype=dtype, device=device)


def convert_transforms(transforms: list[dict[str, Any]]) -> tf.Compose:
    """Convert transforms from list configuration to a Compose tranform."""
    transforms_list = []
    for transform in transforms:
        assert "type" in transform, \
            "Each transform dict must have a 'type' entry specifying its class."
        assert hasattr(tf, transform["type"]), \
            f"Found no valid transform with type {transform['type']} in transforms module."
        type = transform["type"]
        kwargs = transform.get("kwargs", {})
        cls = getattr(tf, type)
        transforms_list.append(cls(**kwargs))
    return tf.Compose.create(transforms_list)
