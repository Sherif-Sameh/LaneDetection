import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from lane_detection.dataloaders.numpy import NumPyLaneDataloader
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
    dataset.load()

    # Initialize dataloder and get sample batch from it
    dataloader = NumPyLaneDataloader(dataset, batch_size=9, shuffle=True)
    images, labels = next(iter(dataloader))
    visualize_lanes(images, labels)
    
    # Measure time to iterate through full dataset
    dataloader = NumPyLaneDataloader(dataset, batch_size=32, shuffle=True)
    time_st = time.time() 
    for images, labels in dataloader:
        continue
    time_dt = time.time() - time_st
    print(f"Took {time_dt:.2f} seconds to iterate through dataset")


if __name__ == "__main__":
    main()