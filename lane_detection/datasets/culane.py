import json
import os
import warnings
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
        val: Ratio of samples allocated for validation.
        test: Ratio of samples allocated for testing.
        height: Height for resizing images and labels.
        width: Width for resizing images and labels.
    """
    N_IMGS_PER_VIDEO = 60
    
    def __init__(
        self,path: Path,
        val: float = 0.1,
        test: float = 0.1,
        height: int = 288,
        width: int = 800,
    ):
        super().__init__(path, val=val, test=test)
        self.height = height
        self.width = width
        self.files = ("images.npy", "labels.npy", "metadata.json")
    
    def download(self) -> None:
        """Download the CULane dataset with segmentation-based annotations.
    
        Warning: this downloaded dataset is not the full original CULane dataset.
        The dataset is downloaded from the following link:
        https://www.kaggle.com/datasets/greatgamedota/culane.
        """
        if self.exists():
            warnings.warn(
                "Dataset already exists. To re-download/convert, delete it and try again."
            )
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
        val: bool = False,
        test: bool = False,
        use_mmap: bool = True,
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Load the stored dataset and return its data and labels.
        
        Args:
            val: Load the data allocated for validation instead of training.
            test: Load the data allocated for testing instead of training. If both val and test
                flags are set, then validation takes higher precedence over testing.
            use_mmap: Keep dataset on disk and load slices dynamically in RAM as needed.
        
        Returns:
            tuple
            - data: (N, H, W, 3) array containing raw data in the dataset.
            - labels: (N, H, W) array containing labels in the dataset.
        """
        assert self.exists(), f"Dataset does not exist at {self.paths[0].parent}."
        path = self.paths[2] if test else self.paths[0]
        path = self.paths[1] if val else path

        # Load dataset
        mmap_mode = "r" if use_mmap else None
        images = np.load(path / self.files[0], mmap_mode=mmap_mode)
        labels = np.load(path / self.files[1], mmap_mode=mmap_mode)
        with open(path / self.files[2], "r") as metafile:
            self.metadata = json.load(metafile)
        return images, labels
    
    def exists(self) -> bool:
        """Check whether the dataset already exists at its specified path or not.
        
        Returns:
            True if the dataset already exists and False otherwise.
        """
        dir_flags = []
        for p in self.paths:
            dir_flag = all([(p / f).exists() for f in self.files])
            dir_flags.append(dir_flag)
        return all(dir_flags)
    
    def _convert(self, download_path: Path) -> None:
        """Iterates through CULane dataset to extract all images and labels, resize them as needed
        and store them into .npy files.
    
        Args:
            download_path: Path to raw CULane dataset.
        """
        images_path = download_path / "driver_161_90frame"
        labels_path = download_path / "driver_161_90frame_labels"
        
        # Generate permutation for splitting dataset
        n_videos = len(list(images_path.iterdir()))
        split_idxs = self._get_permutations(n_videos * self.N_IMGS_PER_VIDEO)
        
        # Process and store images
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
        np.save(self.paths[0].parent / self.files[0], images)
        del images
        images = np.load(self.paths[0].parent / self.files[0], mmap_mode="r")
        for p, idxs in zip(self.paths, split_idxs):
            images_split = images[idxs]
            np.save(p / self.files[0], images_split)
            del images_split
        del images
        (self.paths[0].parent / self.files[0]).unlink()

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
        np.save(self.paths[0].parent / self.files[1], labels)
        del labels
        labels = np.load(self.paths[0].parent / self.files[1], mmap_mode="r")
        for p, idxs in zip(self.paths, split_idxs):
            labels_split = labels[idxs]
            np.save(p / self.files[1], labels_split)
            del labels_split
        del labels
        (self.paths[0].parent / self.files[1]).unlink()
        
        # Store dataset's metadata
        for p, idxs in zip(self.paths, split_idxs):
            metadata = {
                "size": len(idxs),
                "image_shape": images_shape[1:],
                "label_shape": labels_shape[1:],
            }
            with open(p / self.files[2], "w") as metafile: 
                json.dump(metadata, metafile, indent=4)
    
    def _get_permutations(self, size: int) -> tuple[NDArray, NDArray, NDArray]:
        """Get permutations for shuffling and splitting dataset into train, val and test.
        
        Returns:
            tuple
            - train_idxs: array containing indices for training samples.
            - val_idxs: array containing indices for validation samples.
            - test_idxs: array containing indices for testing samples.
        """
        train_size = int(size * self.split[0])
        val_size = int(size * self.split[1])
        test_size = size - train_size - val_size
        split_cumsum = np.cumsum([train_size, val_size, test_size], dtype=np.long)

        perm = np.random.permutation(size)
        train_idxs = perm[:split_cumsum[0]]
        val_idxs = perm[split_cumsum[0]: split_cumsum[1]]
        test_idxs = perm[split_cumsum[1]: split_cumsum[2]]
        return train_idxs, val_idxs, test_idxs
    

def main(data_dir: str, val: float = 0.1, test: float = 0.1):
    dataset = CULaneDataset(Path(data_dir), val=val, test=test)
    dataset.download()


if __name__ == "__main__":
    fire.Fire(main)