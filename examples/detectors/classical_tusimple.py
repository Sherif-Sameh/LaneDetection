from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import toml
from numpy.typing import NDArray

from lane_detection.datasets import TuSimpleDataset
from lane_detection.detectors import ClassicalLaneDetector


def visualize_detections(
    image: NDArray,
    binary: NDArray,
    edge: NDArray,
    lane: NDArray,
) -> None:
    _, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    label_color = np.array([0, 255, 0])
    axes[0].imshow(image[0])
    axes[0].set_title("Input Image")

    axes[1].imshow(binary[0], cmap='gray')
    axes[1].set_title("Binary Image")
    
    axes[2].imshow(edge[0], cmap='gray')
    axes[2].set_title("Edge Image")
    
    lane_image = np.copy(image)
    lane_mask = lane != 0
    lane_image[lane_mask] = label_color
    axes[3].imshow(image[0])
    axes[3].imshow(lane_image[0], alpha=0.6)
    axes[3].set_title("Lane Image")
    
    plt.tight_layout()
    plt.show()


def main():
    # Initialize TuSimple dataset
    path = Path(__file__).parents[2] / "data/TuSimple"
    dataset = TuSimpleDataset(path, val=0.0)
    dataset.download()
    dataset.load()
    
    # Load detector config and initialize it
    path = Path(__file__).parent / "configs/classical.toml"
    config = toml.load(path)
    detector = ClassicalLaneDetector(**config)

    # Apply detector to 4 input images from dataset
    n_samples = 4
    for i in range(n_samples):
        image, _ = dataset[i:i+1]
        lane = detector.detect_lanes(image)
        binary, edge = detector.intermed_outs
        visualize_detections(image, binary, edge, lane)


if __name__ == "__main__":
    main()