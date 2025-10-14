from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from lane_detection.datasets import CULaneDataset


def visualize_lanes(images: NDArray, labels: NDArray) -> None:
    assert len(images) == 9
    assert len(labels) == 9
    
    # Visualize images with labels overlayed on top
    _, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    label_colors = {
        1: np.array([255, 0, 0]),
        3: np.array([0, 0, 255]),
        2: np.array([0, 255, 0]),
        4: np.array([0, 128, 128]),
    }
    for i, (ax, img, lbl) in enumerate(zip(axes, images, labels)):
        ax.imshow(img)
        ax.set_title(f"Image {i + 1}")
        for j in label_colors.keys():
            label_img = np.copy(img)
            lane_mask = lbl == j
            label_img[lane_mask] = label_colors[j]
            ax.imshow(label_img, alpha=0.8)
    
    plt.tight_layout()
    plt.show()


def main():
    # Download and load the CULane dataset
    path = Path(__file__).parents[1] / "data/CULane"
    dataset = CULaneDataset(path, val=0.0, test=0.1)
    dataset.download()
    
    # Extract subset of 9 images from training dataset
    images, labels = dataset.load(use_mmap=True)
    idxs = np.random.choice(images.shape[0], size=9, replace=False)
    images = images[idxs]
    labels = labels[idxs]

    # Visualize images with labels overlayed on top
    visualize_lanes(images, labels)

    # Extract subset of 9 images from test dataset
    images, labels = dataset.load(test=True, use_mmap=True)
    idxs = np.random.choice(images.shape[0], size=9, replace=False)
    images = images[idxs]
    labels = labels[idxs]

    # Visualize images with labels overlayed on top
    visualize_lanes(images, labels)


if __name__ == "__main__":
    main()