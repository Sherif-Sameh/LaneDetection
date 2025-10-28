import os
from pathlib import Path
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax.numpy as jnp
import matplotlib.pyplot as plt
import toml
from flax import nnx
from jax import Array

from lane_detection.dataloaders import NumPyLaneDataloader
from lane_detection.datasets import CULaneDataset
from lane_detection.models import SCNN, VGG11
from lane_detection.trainers import SCNNTrainer
from lane_detection.utils import convert_transforms


def get_dataloader(batch_size: int, use_mmap: bool, test: bool = False) -> NumPyLaneDataloader:
    path = Path(__file__).parents[2] / "data/CULane"
    dataset = CULaneDataset(path, val=0.0, test=0.1)
    if not any(path.iterdir()):
        dataset.download()
    dataset.load(test=test, use_mmap=use_mmap)
    dataloader = NumPyLaneDataloader(dataset, batch_size=batch_size, shuffle=(not test))
    return dataloader


def get_VGG11_backbone(*, rngs: nnx.Rngs) -> nnx.List:
    # Load pre-trained VGG16-BN backbone
    backbone = VGG11(
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
    backbone = nnx.Sequential(*backbone)
    return backbone


def plot_metrics(metrics_history: dict[str, Array], prefix: str = "train") -> None:
    n_epochs = metrics_history[f"{prefix}_loss"].shape[0]
    epochs = jnp.arange(1, n_epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    fig.suptitle(f"{prefix.title()}ing Metrics")
    axes[0].plot(epochs, metrics_history[f"{prefix}_loss"])
    axes[1].plot(epochs, metrics_history[f"{prefix}_acc_seg"], label="Lane Segmentation")
    axes[1].plot(epochs, metrics_history[f"{prefix}_acc_ext"], label="Lane Existence")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].grid(visible=True)
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(visible=True)
    plt.tight_layout()
    plt.show()


def main():
    # Load configurations and convert transforms
    path = Path(__file__).parent / "configs/scnn.toml"
    config = toml.load(path)
    config["trainer"]["transform"] = convert_transforms(config["trainer"]["transform"])
    config["trainer"]["target_transform"] = convert_transforms(
        config["trainer"]["target_transform"]
    )

    # Prepare CULane datasets and get dataloaders
    dataloader = get_dataloader(**config["data"], test=False)
    test_dataloader = get_dataloader(**config["data"], test=True)
    
    # Initialize SCNN model and trainer
    model = SCNN(
        **config["scnn"], backbone=get_VGG11_backbone(rngs=nnx.Rngs(0)), rngs=nnx.Rngs(0)
    )
    trainer = SCNNTrainer(model, **config["trainer"])

    # Train model for set number of epochs
    metrics_history = trainer.train(dataloader, test_dataloader, **config["train"])
    trainer.model.save()

    # Plot training and testing results
    plot_metrics(metrics_history, prefix="train")
    plot_metrics(metrics_history, prefix="test")


if __name__ == "__main__":
    main()
