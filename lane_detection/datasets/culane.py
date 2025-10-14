import os
import json
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

import fire
import kagglehub
from tqdm import tqdm

from lane_detection.datasets.base import LaneDataset


class CULaneDataset(LaneDataset):
    """Class for handling the CULane lane detection dataset.
    
    Args:
        path: Path to dataset directory.
    """
    N_IMGS_PER_VIDEO = 60
    
    def __init__(self, path, height: int = 288, width: int = 800):
        super().__init__(path)
        self.height = height
        self.width = width
        
        self.images_path = self.path / "images.npy"
        self.labels_path = self.path / "labels.npy"
        self.metadata_path = self.path / "metadata.json"
    
    def download(self) -> None:
        """Download the CULane dataset with segmentation-based annotations.
    
        Warning: this downloaded dataset is not the full original CULane dataset.
        The dataset is downloaded from the following link:
        https://www.kaggle.com/datasets/greatgamedota/culane.
        """
        if self.exists():
            return
        
        # Get credentials either from kaggle.json or by propmting user
        kaggle_dir = Path().home() / ".kaggle" if "KAGGLE_CONFIG_DIR" not in os.environ \
            else Path(os.environ["KAGGLE_CONFIG_DIR"])
        kaggle_path = kaggle_dir / "kaggle.json"
        if not kaggle_path.exists():
            kagglehub.login()

        # Download dataset
        download_path = kagglehub.dataset_download("greatgamedota/culane")
        download_path = Path(download_path)

        # Convert dataset to NumPy arrays and store it into desired directory
        self._convert(download_path)
    
    def load(
        self,
        n_samples: int | None = None,
        use_mmap: bool = True,
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Load the stored dataset and return its data and labels.
        
        Args:
            n_samples: Number of samples to load. If None, then all dataset is loaded.
            use_mmap: Keep dataset on disk and load slices dynamically in RAM as needed.
        
        Returns:
            tuple
            - data: (N, H, W, 3) array containing all raw data in the dataset.
            - labels: (N, H, W) array containing all labels in the dataset.
        """
        assert self.exists(), f"Dataset does not exist at {self.path}."

        # Load dataset
        mmap_mode = "r" if use_mmap else None
        images = np.load(self.images_path, mmap_mode=mmap_mode)
        labels = np.load(self.labels_path, mmap_mode=mmap_mode)
        with open(self.metadata_path, "r") as metafile:
            self.metadata = json.load(metafile)
        
        # Determine the number of samples
        size = self.metadata["size"]
        n_samples = size if n_samples is None else min(n_samples, size)
        return images[:n_samples], labels[:n_samples]
    
    def exists(self) -> bool:
        """Check whether the dataset already exists at its specified path or not.
        
        Returns:
            True if the dataset already exists and False otherwise.
        """
        paths = [self.images_path, self.labels_path, self.metadata_path]
        return all([f.exists() for f in paths])
    
    def _convert(self, download_path: Path) -> None:
        """Iterates through CULane dataset to extract all images and labels, resize them as needed
        and store them into .npy files.
    
        Args:
            download_path: Path to raw CULane dataset.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        images_path = download_path / "driver_161_90frame"
        labels_path = download_path / "driver_161_90frame_labels"
        
        # Process and store images
        n_videos = len(list(images_path.iterdir()))
        images_shape = (n_videos * self.N_IMGS_PER_VIDEO, self.height, self.width, 3)
        images = np.zeros(images_shape, dtype=np.uint8)
        for video_idx, video_dir in enumerate(
            tqdm(sorted(images_path.iterdir()), desc="Processing Images")
        ):
            image_idx = 0
            for image_file in sorted(video_dir.iterdir()):
                if image_file.suffix != ".jpg":
                    continue
                label_file = labels_path / f"{video_dir.name}/{image_file.stem}.png"
                if not label_file.exists():
                    continue

                image = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                array_idx = video_idx * self.N_IMGS_PER_VIDEO + image_idx
                images[array_idx] = cv2.resize(image, (self.width, self.height))
                image_idx += 1  
        np.save(self.images_path, images)
        del images

        # Process and store labels
        labels_shape = images_shape[:-1]  # labels don't need a channel dimension
        labels = np.zeros(labels_shape, dtype=np.uint8)
        for video_idx, video_dir in enumerate(
            tqdm(sorted(images_path.iterdir()), desc="Processing Labels")
        ):
            image_idx = 0
            for image_file in sorted(video_dir.iterdir()):
                if image_file.suffix != ".jpg":
                    continue
                label_file = labels_path / f"{video_dir.name}/{image_file.stem}.png"
                if not label_file.exists():
                    continue

                label = cv2.cvtColor(cv2.imread(label_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                array_idx = video_idx * self.N_IMGS_PER_VIDEO + image_idx
                labels[array_idx] = cv2.resize(label, (self.width, self.height))[..., 0]
                image_idx += 1      
        np.save(self.labels_path, labels)
        del labels

        # Store dataset's metadata
        self.metadata = {
            "size": n_videos * self.N_IMGS_PER_VIDEO,
            "n_videos": n_videos,
            "n_imgs_per_video": self.N_IMGS_PER_VIDEO,
            "image_shape": images_shape[1:],
            "label_shape": labels_shape[1:],
        }
        with open(self.metadata_path, "w") as metafile: 
            json.dump(self.metadata, metafile, indent=4)


def main(data_dir: str):
    dataset = CULaneDataset(Path(data_dir))
    dataset.download()


if __name__ == "__main__":
    fire.Fire(main)