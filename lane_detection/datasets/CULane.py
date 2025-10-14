import os
import json
from pathlib import Path
from typing import Any

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
    if exists_culane(data_dir):
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

    # Store dataset's metadata
    metadata = {
        "size": n_videos * n_imgs_per_video,
        "n_videos": n_videos,
        "n_imgs_per_video": n_imgs_per_video,
        "img_shape": (height, width, 3),
        "lbl_shape": (height, width),
    }
    with open(data_dir / "metadata.json", "w") as metafile: 
        json.dump(metadata, metafile, indent=4)


def exists_culane(data_dir: Path) -> bool:
    """Checks whether the dataset already exists or not.
    
    Args:
        data_dir: Path to directory where the dataset was stored.
    
    Returns:
        True if the dataset already exists and False otherwise.
    """
    images_path = data_dir / "images.npy"
    labels_path = data_dir / "labels.npy"
    meta_path = data_dir / "metadata.json"
    return all([f.exists() for f in [images_path, labels_path, meta_path]])


def load_culane(
    data_dir: Path,
    n_samples: int = -1,
    use_mmap: bool = True,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], dict[str, Any]]:
    """Load both raw images and labels from the stored CULane dataset.
    
    Args:
        data_dir: Path to directory where the dataset was stored.
        n_samples: Number of samples to load. If negative, then all dataset is loaded.
        use_mmap: Keep dataset on disk and load slices dynamically in RAM as needed.
    
    Returns:
        tuple
        - images: (N, H, W, 3) array containing all images in the dataset.
        - labels: (N, H, W) array containing all labels in the dataset.
        - metadata: Dict containing the dataset's metadata.
    """
    assert data_dir.exists()
    assert data_dir.is_dir()
    
    images_path = data_dir / "images.npy"
    labels_path = data_dir / "labels.npy"
    metadata_path = data_dir / "metadata.json"
    assert exists_culane(data_dir), f"Dataset does not exist at {data_dir}."

    # Load dataset
    mmap_mode = "r" if use_mmap else None
    images = np.load(images_path, mmap_mode=mmap_mode)
    labels = np.load(labels_path, mmap_mode=mmap_mode)
    with open(metadata_path, "r") as metafile:
        metadata = json.load(metafile)
    
    # Determine the number of samples
    n_samples = metadata["size"] if n_samples < 0 else n_samples
    n_samples = metadata["size"] if n_samples > metadata["size"] else n_samples
    return images[:n_samples], labels[:n_samples], metadata


def main(data_dir: str):
    download_culane(Path(data_dir))


if __name__ == "__main__":
    fire.Fire(main)