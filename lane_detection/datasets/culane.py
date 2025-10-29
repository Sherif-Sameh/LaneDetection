import json
import warnings
from pathlib import Path
from random import shuffle

import cv2
import h5py
import kagglehub
import numpy as np
from numpy.typing import NDArray
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
        self.files = ("dataset.hdf5", "metadata.json")

        self.data = None
        self.images = np.zeros((0, self.height, self.width, 3), dtype=np.uint8)
        self.labels = np.zeros((0, self.height, self.width), dtype=np.uint8)
    
    def __len__(self) -> int:
        """Return the number of samples available in the loaded dataset."""
        return self.images.shape[0]
    
    def __getitem__(self, idx: int | slice | tuple[int | slice, ...]) -> tuple[NDArray, NDArray]:
        """Retrieve data sample/s from the dataset and return them."""
        return self.images[idx], self.labels[idx]

    def download(self, batch_size: int = 1024) -> None:
        """Download the CULane dataset with segmentation-based annotations.
    
        Warning: this downloaded dataset is not the full original CULane dataset.
        The dataset is downloaded from the following link:
        https://www.kaggle.com/datasets/greatgamedota/culane.

        Args:
            batch_size: Batch size for converting dataset.
        """
        if self.exists():
            warnings.warn(
                "Dataset already exists. To re-download/convert, delete it and try again."
            )
            return
        
        # Download dataset
        download_path = kagglehub.dataset_download("greatgamedota/culane")
        download_path = Path(download_path)

        # Convert dataset to NumPy arrays and store it into desired directory
        self._convert(download_path, batch_size)
    
    def load(self, val: bool = False, test: bool = False) -> None:
        """Load the stored dataset to prepare for data reading.
        
        Args:
            val: Load the data allocated for validation instead of training.
            test: Load the data allocated for testing instead of training. If both val and test
                flags are set, then validation takes higher precedence over testing.
        """
        assert self.exists(), f"Dataset does not exist at {self.paths[0].parent}."
        path = self.paths[2] if test else self.paths[0]
        path = self.paths[1] if val else path

        # Load dataset
        self.data = h5py.File(path / self.files[0], "r")
        self.images = self.data["images"]
        self.labels = self.data["labels"]
        with open(path / self.files[1], "r") as metafile:
            self.metadata = json.load(metafile)
    
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
    
    def _convert(self, download_path: Path, batch_size: int) -> None:
        """Iterates through CULane dataset to extract all images and labels, resize them as needed
        and store them into .npy files.
    
        Args:
            download_path: Path to raw CULane dataset.
            batch_size: Batch size for converting dataset.
        """
        images_path = download_path / "driver_161_90frame"
        labels_path = download_path / "driver_161_90frame_labels"
        
        # Prepare dataset files for storing samples
        img_shape = (self.height, self.width, 3)
        video_dirs = list(images_path.iterdir())
        shuffle(video_dirs)
        sizes = self._get_sizes(download_path)
        files = [self._create_file(path, size, img_shape, dtype="uint8") \
                 for (path, size) in zip(self.paths, sizes)]
        
        # Process and store images and labels
        n_samples, data_idx = 0, 0
        img_batch = np.zeros((batch_size,) + img_shape, dtype=np.uint8)
        lbl_batch = np.zeros((batch_size,) + img_shape[:-1], dtype=np.uint8)
        for video_dir in tqdm(video_dirs, desc="Processing Videos"):
            for image_file in sorted(video_dir.iterdir()):
                if image_file.suffix != ".jpg":
                    continue
                label_file = labels_path / f"{video_dir.name}/{image_file.stem}.png"
                if not label_file.exists():
                    continue

                sample_idx = n_samples % batch_size
                image = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                label = cv2.cvtColor(cv2.imread(label_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                img_batch[sample_idx] = cv2.resize(image, (self.width, self.height)) 
                lbl_batch[sample_idx] = cv2.resize(label, (self.width, self.height))[..., 0]
                n_samples += 1

                is_complete = n_samples == sizes[data_idx]
                if n_samples % batch_size == 0 or is_complete:
                    self._write_batch_to_file(files[data_idx], n_samples, img_batch, lbl_batch)
                    if is_complete:
                        n_samples = 0
                        data_idx += 1
                    while data_idx < 3 and sizes[data_idx] == 0:
                        data_idx += 1
                    
        for file in files:
            file.close()
        
        # Store dataset's metadata
        for data_idx, path in enumerate(self.paths):
            metadata = {
                "size": sizes[data_idx],
                "image_shape": img_shape,
                "label_shape": img_shape[:-1],
            }
            with open(path / self.files[1], "w") as metafile: 
                json.dump(metadata, metafile, indent=4)
    
    def _get_sizes(self, download_path: Path) -> tuple[int, int, int]:
        """Calculate the sizes of each dataset based on total size and data split."""
        total_size = 0
        images_path = download_path / "driver_161_90frame"
        labels_path = download_path / "driver_161_90frame_labels"
        for video_dir in images_path.iterdir():
            for image_file in video_dir.iterdir():
                if image_file.suffix != ".jpg":
                    continue
                label_file = labels_path / f"{video_dir.name}/{image_file.stem}.png"
                if not label_file.exists():
                    continue
                total_size += 1
        train_size = int(total_size * self.split[0])
        val_size = int(total_size * self.split[1])
        test_size = total_size - train_size - val_size
        return train_size, val_size, test_size
    
    def _create_file(
        self,
        path: Path,
        size: int,
        shape: tuple[int, ...],
        dtype: str = "int",
    ) -> h5py.File:
        """Create a hdf5 file for storing dataset.
        
        Args:
            path: Path to directory to store dataset within.
            size: Total number of samples to be stored in dataset.
            shape: Shape of images to be stored in dataset of the format (H, W, 3).
            dtype: Data type of images and labels.
        
        Returns:
            H5Py File object for created dataset opened in "write" mode.
        """
        images_shape = (size,) + shape
        labels_shape = (size,) + shape[:-1]
        f = h5py.File(str(path / self.files[0]), mode="w")
        f.create_dataset("images", shape=images_shape, dtype=dtype)
        f.create_dataset("labels", shape=labels_shape, dtype=dtype)
        return f
    
    def _write_batch_to_file(
        self,
        file: h5py.File,
        n_samples: int,
        imgs: NDArray,
        lbls: NDArray,
    ) -> None:
        """Write a batch of images and labels into H5Py dataset file."""
        batch_size = imgs.shape[0]
        n_valid = (n_samples - 1) % batch_size
        start = ((n_samples - 1) // batch_size) * batch_size
        end = start + n_valid
        file["images"][start: end + 1] = imgs[:n_valid + 1]
        file["labels"][start: end + 1] = lbls[:n_valid + 1]
