from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path

import torch
from flax import nnx
from jax import Array

from lane_detection.utils.utils import (
    download_weights_from_url,
    load_bn_from_torch,
    load_conv_from_torch,
    load_linear_from_torch,
)


class VGGBlock(nnx.Module):
    """VGG convolutional block.
    
    Each block is made up of one or multiple 3x3 conv layers, followed by an optional batch norm,
    with ReLU activations. Each block ends with a 2x2 max pooling layer with a stride of 2.

    Args:
        n_convs: Number of convolutional layers in the block.
        in_features: Number of features (channels) of inputs.
        out_features: Number of features (channels) of outputs after convolutions.
        batch_norm: Use batch normalization after each convolutional layer.
        rngs: RNG state to use for layer weight initialization.
    """

    def __init__(
        self,
        n_convs: int,
        in_features: int,
        out_features: int,
        batch_norm: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        layers = {}
        for i in range(n_convs):
            layers[f"conv{i+1}"] = nnx.Conv(in_features, out_features, kernel_size=(3, 3), rngs=rngs)
            if batch_norm:
                layers[f"bn{i+1}"] = nnx.BatchNorm(out_features, rngs=rngs)
            layers[f"act{i+1}"] = nnx.relu
            in_features = out_features
        layers["pool"] = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))
        self.layers = nnx.Dict(layers)
    
    def __call__(self, x: Array) -> Array:
        """Forward propagate input through conv block.
        
        Args:
            x: (B, H, W, C) input array.
        
        Returns:
            (B, H/2, W/2, C') output array.
        """
        assert x.ndim == 4, f"Expected batched 4-dim inputs, got {x.ndim}-dim."
        assert x.shape[3] == self.in_features, \
            f"Expected {self.in_features} features, got {x.shape[3]}"
        for layer in self.layers.values():
            x = layer(x)
        return x


class VGG(nnx.Module, ABC):
    """VGG from `Very Deep Convolutional Networks for Large-Scale Image Recognition`.
    
    Paper: <https://arxiv.org/abs/1409.1556>.

    Note: Unlike the PyTorch equivalent, this implementation is only compatible with (224 x 224)
    inputs since Flax NNX lacks adaptive pooling to ensure a fixed input size to the FC layers. 
    
    Args:
        n_classes: Number of classes to predict output label from.
        dropout: Dropout probability (same as PyTorch).
        batch_norm: Use batch normalization after each convolutional layer.
        pretrained: Pre-load ImageNet-trained weights from PyTorch (downloaded if needed).
        rngs: RNG state to use for layer weight initialization.
    """

    def __init__(
        self,
        n_classes: int = 1000,
        dropout: float = 0.5,
        batch_norm: bool = False,
        pretrained: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_shape = (224, 224, 3)  # Expected input shape
        self.batch_norm = batch_norm

        # Define architecture of feature extractor backbone
        self.features = self._make_feature_extractor(rngs=rngs)

        # Define architecture of FC classfier
        self.classifier = nnx.Dict({
            "fc1": nnx.Linear(7 * 7 * 512, 4096, rngs=rngs),
            "act1": nnx.relu,
            "dp1": nnx.Dropout(dropout, rngs=rngs),
            "fc2": nnx.Linear(4096, 4096, rngs=rngs),
            "act2": nnx.relu,
            "dp2": nnx.Dropout(dropout, rngs=rngs),
            "out": nnx.Linear(4096, n_classes, rngs=rngs),
        })

        # Download and load parameters if required
        if pretrained:
            path = self._download_weights_from_torch()
            self._load_weights_from_torch(path)
    
    def __call__(self, x: Array) -> Array:
        """Forward propagate input through VGG model to predict output labels.
        
        Args:
            x: (B, 224, 224, 3) input array.
        
        Returns:
            (B, n_classes) output array.
        """
        assert x.ndim == 4, f"Expected batched 4-dim inputs, got {x.ndim}-dim."
        assert x.shape[1:] == self.in_shape, \
            f"Expected image shape {self.in_shape}, got{x.shape[1:]}"
        for layer in self.features.values():
            x = layer(x)
        x = x.reshape(x.shape[0], -1)
        for layer in self.classifier.values():
            x = layer(x)
        return x 

    def _load_weights_from_torch(self, path: Path) -> None:
        """Load the ImageNet-trained weights from PyTorch."""
        state_dict = torch.load(path, map_location="cpu")
        n_layers = 0 
        for block in self.features.values():
            for module in block.layers.values():
                match module:
                    case nnx.BatchNorm():
                        load_bn_from_torch(
                            module,
                            state_dict[f"features.{n_layers}.running_mean"],
                            state_dict[f"features.{n_layers}.running_var"],
                            scale=state_dict.get(f"features.{n_layers}.weight"),
                            bias=state_dict.get(f"features.{n_layers}.bias"),
                        )
                    case nnx.Conv():
                        load_conv_from_torch(
                            module,
                            state_dict[f"features.{n_layers}.weight"],
                            bias=state_dict.get(f"features.{n_layers}.bias"),
                        )
                    case _:
                        pass
                n_layers += 1
        
        n_layers = 0
        for module in self.classifier.values():
            match module:
                case nnx.Linear():
                    load_linear_from_torch(
                        module,
                        state_dict[f"classifier.{n_layers}.weight"],
                        bias=state_dict.get(f"classifier.{n_layers}.bias"),
                    )
                case _:
                    pass
            n_layers += 1
    
    @abstractmethod
    def _make_feature_extractor(self, *, rngs: nnx.Rngs) -> nnx.Dict:
        """Create the feature extractor according to the chosen model from the VGG models."""
    
    @abstractmethod
    def _download_weights_from_torch(self) -> Path:
        """Download the ImageNet-trained weights from PyTorch.
        
        Returns:
            Path to the downloaded weights file.
        """


class VGG11(VGG):
    """VGG-11 from `Very Deep Convolutional Networks for Large-Scale Image Recognition`.
    
    Paper: <https://arxiv.org/abs/1409.1556>.

    Note: Unlike the PyTorch equivalent, this implementation is only compatible with (224 x 224)
    inputs since Flax NNX lacks adaptive pooling to ensure a fixed input size to the FC layers. 
    
    Args:
        n_classes: Number of classes to predict output label from.
        dropout: Dropout probability (same as PyTorch).
        batch_norm: Use batch normalization after each convolutional layer.
        rngs: RNG state to use for layer weight initialization.
    """

    def _make_feature_extractor(self, *, rngs: nnx.Rngs) -> nnx.Dict:
        """Create the feature extractor according to the chosen model from the VGG models."""
        features = nnx.Dict({
            "block1": VGGBlock(1, 3, 64, batch_norm=self.batch_norm, rngs=rngs),
            "block2": VGGBlock(1, 64, 128, batch_norm=self.batch_norm, rngs=rngs),
            "block3": VGGBlock(2, 128, 256, batch_norm=self.batch_norm, rngs=rngs),
            "block4": VGGBlock(2, 256, 512, batch_norm=self.batch_norm, rngs=rngs),
            "block5": VGGBlock(2, 512, 512, batch_norm=self.batch_norm, rngs=rngs),
        })
        return features

    def _download_weights_from_torch(self) -> Path:
        """Download the ImageNet-trained weights from PyTorch.
        
        Returns:
            Path to the downloaded weights file.
        """
        dir = Path(__file__).parent / "weights"
        if not self.batch_norm:
            url = "https://download.pytorch.org/models/vgg11-8a719046.pth"
            file = dir / "vgg11.pth"
        else:
            url = "https://download.pytorch.org/models/vgg11_bn-6002323d.pth"
            file = dir / "vgg11_bn.pth"
        if not file.exists():
            download_weights_from_url(url, file)
        return file


class VGG13(VGG):
    """VGG-13 from `Very Deep Convolutional Networks for Large-Scale Image Recognition`.
    
    Paper: <https://arxiv.org/abs/1409.1556>.

    Note: Unlike the PyTorch equivalent, this implementation is only compatible with (224 x 224)
    inputs since Flax NNX lacks adaptive pooling to ensure a fixed input size to the FC layers. 
    
    Args:
        n_classes: Number of classes to predict output label from.
        dropout: Dropout probability (same as PyTorch).
        batch_norm: Use batch normalization after each convolutional layer.
        rngs: RNG state to use for layer weight initialization.
    """

    def _make_feature_extractor(self, *, rngs: nnx.Rngs) -> nnx.Dict:
        """Create the feature extractor according to the chosen model from the VGG models."""
        features = nnx.Dict({
            "block1": VGGBlock(2, 3, 64, batch_norm=self.batch_norm, rngs=rngs),
            "block2": VGGBlock(2, 64, 128, batch_norm=self.batch_norm, rngs=rngs),
            "block3": VGGBlock(2, 128, 256, batch_norm=self.batch_norm, rngs=rngs),
            "block4": VGGBlock(2, 256, 512, batch_norm=self.batch_norm, rngs=rngs),
            "block5": VGGBlock(2, 512, 512, batch_norm=self.batch_norm, rngs=rngs),
        })
        return features

    def _download_weights_from_torch(self) -> Path:
        """Download the ImageNet-trained weights from PyTorch.
        
        Returns:
            Path to the downloaded weights file.
        """
        dir = Path(__file__).parent / "weights"
        if not self.batch_norm:
            url = "https://download.pytorch.org/models/vgg13-19584684.pth"
            file = dir / "vgg13.pth"
        else:
            url = "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth"
            file = dir / "vgg13_bn.pth"
        if not file.exists():
            download_weights_from_url(url, file)
        return file


class VGG16(VGG):
    """VGG-16 from `Very Deep Convolutional Networks for Large-Scale Image Recognition`.
    
    Paper: <https://arxiv.org/abs/1409.1556>.

    Note: Unlike the PyTorch equivalent, this implementation is only compatible with (224 x 224)
    inputs since Flax NNX lacks adaptive pooling to ensure a fixed input size to the FC layers. 
    
    Args:
        n_classes: Number of classes to predict output label from.
        dropout: Dropout probability (same as PyTorch).
        batch_norm: Use batch normalization after each convolutional layer.
        rngs: RNG state to use for layer weight initialization.
    """

    def _make_feature_extractor(self, *, rngs: nnx.Rngs) -> nnx.Dict:
        """Create the feature extractor according to the chosen model from the VGG models."""
        features = nnx.Dict({
            "block1": VGGBlock(2, 3, 64, batch_norm=self.batch_norm, rngs=rngs),
            "block2": VGGBlock(2, 64, 128, batch_norm=self.batch_norm, rngs=rngs),
            "block3": VGGBlock(3, 128, 256, batch_norm=self.batch_norm, rngs=rngs),
            "block4": VGGBlock(3, 256, 512, batch_norm=self.batch_norm, rngs=rngs),
            "block5": VGGBlock(3, 512, 512, batch_norm=self.batch_norm, rngs=rngs),
        })
        return features

    def _download_weights_from_torch(self) -> Path:
        """Download the ImageNet-trained weights from PyTorch.
        
        Returns:
            Path to the downloaded weights file.
        """
        dir = Path(__file__).parent / "weights"
        if not self.batch_norm:
            url = "https://download.pytorch.org/models/vgg16-397923af.pth"
            file = dir / "vgg16.pth"
        else:
            url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
            file = dir / "vgg16_bn.pth"
        if not file.exists():
            download_weights_from_url(url, file)
        return file


class VGG19(VGG):
    """VGG-19 from `Very Deep Convolutional Networks for Large-Scale Image Recognition`.
    
    Paper: <https://arxiv.org/abs/1409.1556>.

    Note: Unlike the PyTorch equivalent, this implementation is only compatible with (224 x 224)
    inputs since Flax NNX lacks adaptive pooling to ensure a fixed input size to the FC layers. 
    
    Args:
        n_classes: Number of classes to predict output label from.
        dropout: Dropout probability (same as PyTorch).
        batch_norm: Use batch normalization after each convolutional layer.
        rngs: RNG state to use for layer weight initialization.
    """

    def _make_feature_extractor(self, *, rngs: nnx.Rngs) -> nnx.Dict:
        """Create the feature extractor according to the chosen model from the VGG models."""
        features = nnx.Dict({
            "block1": VGGBlock(2, 3, 64, batch_norm=self.batch_norm, rngs=rngs),
            "block2": VGGBlock(2, 64, 128, batch_norm=self.batch_norm, rngs=rngs),
            "block3": VGGBlock(4, 128, 256, batch_norm=self.batch_norm, rngs=rngs),
            "block4": VGGBlock(4, 256, 512, batch_norm=self.batch_norm, rngs=rngs),
            "block5": VGGBlock(4, 512, 512, batch_norm=self.batch_norm, rngs=rngs),
        })
        return features
    
    def _download_weights_from_torch(self) -> Path:
        """Download the ImageNet-trained weights from PyTorch.
        
        Returns:
            Path to the downloaded weights file.
        """
        dir = Path(__file__).parent / "weights"
        if not self.batch_norm:
            url = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
            file = dir / "vgg19.pth"
        else:
            url = "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth"
            file = dir / "vgg19_bn.pth"
        if not file.exists():
            download_weights_from_url(url, file)
        return file
