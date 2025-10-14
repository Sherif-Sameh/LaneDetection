from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from lane_detection.datasets import download_culane, load_culane


def main():
    # Download and load CULane dataset
    data_dir = Path(__file__).parents[1] / "data/CULane"
    download_culane(data_dir)
    images, labels, _ = load_culane(data_dir, n_samples=9)

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


if __name__ == "__main__":
    main()