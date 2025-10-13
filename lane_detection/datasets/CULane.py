import os
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

import fire
import kagglehub
from tqdm import tqdm


def download_culane(data_dir: Path) -> None:
    """Download the CULane dataset with segmentation-based annotations.
    
    Warning: this downloaded dataset is not the full original CULane dataset.
    The dataset is downloaded from the following link:
    https://www.kaggle.com/datasets/greatgamedota/culane.
    
    Args:
        data_dir: Path to directory where the dataset should be stored.
    """
    # Get credentials either from kaggle.json or by propmting user
    kaggle_dir = Path().home() / ".kaggle" if "KAGGLE_CONFIG_DIR" not in os.environ \
        else Path(os.environ["KAGGLE_CONFIG_DIR"])
    kaggle_path = kaggle_dir / "kaggle.json"
    if not kaggle_path.exists():
        kagglehub.login()

    # Download dataset
    raw_dir = kagglehub.dataset_download("greatgamedota/culane")
    raw_dir = Path(raw_dir)

    # Convert dataset to NumPy arrays and store it into desired directory
    save_culane(raw_dir, data_dir)


def save_culane(raw_dir: Path, data_dir: Path) -> None:
    """Iterates through CULane dataset to extract all images and labels and saves them as
    NumPy arrays in .npy files.
    
    Args:
        raw_dir: Path to raw CULane dataset.
        data_dir: Path to directory for storing extracted dataset.
    """
    if (data_dir / "images.npy").exists() and (data_dir / "labels.npy").exists():
        return
    
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = raw_dir / "driver_161_90frame"
    labels_dir = raw_dir / "driver_161_90frame_labels"
    
    # Allocate array for storing images
    n_videos = len(list(images_dir.iterdir()))
    n_imgs_per_video = 60
    height, width = 288, 800
    images = np.zeros((n_videos * n_imgs_per_video, height, width, 3), dtype=np.uint8)

    # Store all images
    for i, video_dir in enumerate(tqdm(sorted(images_dir.iterdir()), desc="Processing Images")):
        j = 0
        for image_file in sorted(video_dir.iterdir()):
            if image_file.suffix != ".jpg":
                continue
            label_file = labels_dir / f"{video_dir.name}/{image_file.stem}.png"
            if not label_file.exists():
                continue

            # Load image and corresponding label add to list
            image = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            images[i * n_imgs_per_video + j] = cv2.resize(image, (width, height))
            j += 1  
    np.save(data_dir / "images.npy", images)
    del images

    # Store all labels
    labels = np.zeros((n_videos * n_imgs_per_video, height, width), dtype=np.uint8)
    for i, video_dir in enumerate(tqdm(sorted(images_dir.iterdir()), desc="Processing Labels")):
        j = 0
        for image_file in sorted(video_dir.iterdir()):
            if image_file.suffix != ".jpg":
                continue
            label_file = labels_dir / f"{video_dir.name}/{image_file.stem}.png"
            if not label_file.exists():
                continue

            # Load image and corresponding label add to list
            label = cv2.cvtColor(cv2.imread(label_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            labels[i * n_imgs_per_video + j] = cv2.resize(label, (width, height))[..., 0]
            j += 1      
    np.save(data_dir / "labels.npy", labels)
    del labels


def load_culane(data_dir: Path, n_samples: int = -1) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Load both raw images and labels from the stored CULane dataset. The dataset is not loaded
    fully into RAM. Instead, the mmap_mode flag is used to keep it on disk and load data slices
    as needed.
    
    Args:
        data_dir: Path to directory where the dataset was stored.
        n_samples: Number of samples to load. If negative, then all dataset is loaded. 
    
    Returns:
        tuple
        - images: (N, H, W, 3) array containing all images in the dataset.
        - labels: (N, H, W) array containing all labels in the dataset.
    """
    assert data_dir.exists()
    assert data_dir.is_dir()
    
    images_path = data_dir / "images.npy"
    labels_path = data_dir / "labels.npy"
    assert images_path.exists() and labels_path.exists()

    # Load dataset from .npz files
    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    
    # Determine the number of samples
    n_samples_total = images.shape[0]
    n_samples = n_samples_total if n_samples < 0 else n_samples
    n_samples = n_samples_total if n_samples > n_samples_total else n_samples
    return images[:n_samples], labels[:n_samples]


def main(data_dir: str):
    download_culane(Path(data_dir))


if __name__ == "__main__":
    fire.Fire(main)