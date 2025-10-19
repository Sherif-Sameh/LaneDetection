from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from lane_detection.datasets import TuSimpleDataset


def visualize_lanes(images: NDArray, labels: NDArray) -> None:
    assert len(images) == 9
    assert len(labels) == 9
    
    # Visualize images with labels overlayed on top
    _, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    label_color = np.array([0, 255, 0])
    for i, (ax, img, lbl) in enumerate(zip(axes, images, labels)):
        label_img = np.copy(img)
        lane_mask = lbl != 0
        label_img[lane_mask] = label_color
        ax.imshow(img)
        ax.imshow(label_img, alpha=0.6)
        ax.set_title(f"Image {i + 1}")

    plt.tight_layout()
    plt.show()


def main():
    # Download and load the TuSimple dataset
    path = Path(__file__).parents[2] / "data/TuSimple"
    dataset = TuSimpleDataset(path, val=0.0)
    dataset.download()
    
    # Extract subset of 9 images from training dataset
    dataset.load(use_mmap=True)
    idxs = np.random.choice(len(dataset), size=9, replace=False)
    images, labels = dataset[idxs]

    # Visualize images with labels overlayed on top
    visualize_lanes(images, labels)

    # Extract subset of 9 images from test dataset
    dataset.load(test=True, use_mmap=True)
    idxs = np.random.choice(len(dataset), size=9, replace=False)
    images, labels = dataset[idxs]

    # Visualize images with labels overlayed on top
    visualize_lanes(images, labels)


if __name__ == "__main__":
    main()