import json
import random
import warnings
from pathlib import Path

import cv2
import h5py
import kagglehub
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from lane_detection.datasets.base import LaneDataset


class TuSimpleDataset(LaneDataset):
    """Class for handling the TuSimple lane detection dataset.

    The TuSimple dataset consists of 3626 samples that are split between training and
    validation data only and another 2782 samples that are completely reserved for testing.
    All images in the TuSimple dataset have a resolution of 1280 x 720 pixels.
    
    Args:
        path: Path to dataset directory.
        val: Ratio of training samples allocated for validation.
        height: Height for resizing images and labels.
        width: Width for resizing images and labels.
    """
    
    def __init__(
        self,path: Path,
        val: float = 0.1,
        height: int = 360,
        width: int = 640,
    ):
        super().__init__(path, val=val, test=0.0)
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
        """Download the TuSimple dataset with segmentation-based annotations.
    
        Note: The downloaded dataset comes from the pre-processed entry on Kaggle available at the
        following link:
        https://www.kaggle.com/datasets/hikmatullahmohammadi/tusimple-preprocessed.
        
        Args:
            batch_size: Batch size for converting dataset.
        """
        if self.exists():
            warnings.warn(
                "Dataset already exists. To re-download/convert, delete it and try again."
            )
            return

        # Download dataset
        download_path = kagglehub.dataset_download("hikmatullahmohammadi/tusimple-preprocessed")
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
        """Iterates through TuSimple dataset to extract all images and labels, resize them as needed
        and store them into .hdf5 files.
    
        Args:
            download_path: Path to raw TuSimple dataset.
            batch_size: Batch size for converting dataset.
        """
        # Prepare dataset files for storing samples
        img_shape = (self.height, self.width, 3)
        sizes = self._get_sizes(download_path)
        files = [self._create_file(path, size, img_shape, dtype="uint8") \
                 for (path, size) in zip(self.paths, sizes)]
        
        # Process and store images and labels for training/validation dataset
        images_path = download_path / "tusimple_preprocessed/training/frames"
        labels_path = download_path / "tusimple_preprocessed/training/lane-masks"
        self._store_images_labels(
            images_path,
            labels_path,
            files[:2],
            sizes[:2],
            batch_size,
            "Train/Val",
            shuffle=True,
        )

        # Process and store images and labels for testing dataset
        images_path = download_path / "tusimple_preprocessed/test/frames"
        labels_path = download_path / "tusimple_preprocessed/test/lane-masks"
        self._store_images_labels(
            images_path,
            labels_path,
            files[2:],
            sizes[2:],
            batch_size,
            "Test",
            shuffle=False,
        )

        for file in files:
            file.close()
        
        # Store dataset's metadata
        for p, size in zip(self.paths, sizes):
            metadata = {
                "size": size,
                "image_shape": (self.height, self.width, 3),
                "label_shape": (self.height, self.width),
            }
            with open(p / self.files[1], "w") as metafile: 
                json.dump(metadata, metafile, indent=4)
    
    def _get_sizes(self, download_path: Path) -> tuple[int, int, int]:
        """Calculate the sizes of each dataset based on total size and data split."""
        # Count samples in training and validation dataset
        images_path = download_path / "tusimple_preprocessed/training/frames"
        labels_path = download_path / "tusimple_preprocessed/training/lane-masks"
        total_train_val_size = 0
        for image_file in images_path.iterdir():
            if image_file.suffix != ".jpg":
                continue
            label_file = labels_path / image_file.name
            if not label_file.exists():
                continue
            total_train_val_size += 1
        train_size = int(total_train_val_size * self.split[0])
        val_size = total_train_val_size - train_size
        
        # Count samples in testing dataset
        images_path = download_path / "tusimple_preprocessed/test/frames"
        labels_path = download_path / "tusimple_preprocessed/test/lane-masks"
        test_size = 0
        for image_file in images_path.iterdir():
            if image_file.suffix != ".jpg":
                continue
            label_file = labels_path / image_file.name
            if not label_file.exists():
                continue
            test_size += 1
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
    
    def _store_images_labels(
        self,
        images_path: Path,
        labels_path: Path,
        files: tuple[h5py.File, ...],
        sizes: tuple[int, ...],
        batch_size: int,
        data_title: str,
        shuffle: bool = False,
    ) -> None:
        """Loop image and label directories, get all entries, pre-process them and store them into
        .hdf5 files.
        
        Args:
            images_path: Path to the directory containing the dataset's images.
            labels_path: Path to the directory containing the dataset's labels.
            files: H5Py files for storing dataset/s.
            sizes: Number of expected image and label pairs in dataset/s.
            batch_size: Batch size for converting dataset/s.
            data_title: Title to display with tqdm while processing the dataset/s.
            shuffle: Iterate over the dataset in random order.
        """
        images_shape = (batch_size, self.height, self.width, 3)
        labels_shape = images_shape[:-1]  # labels don't need a channel dimension
        images_paths = sorted(images_path.iterdir())
        if shuffle:
            random.shuffle(images_paths)
        
        img_batch = np.zeros(images_shape, dtype=np.uint8)
        lbl_batch = np.zeros(labels_shape, dtype=np.uint8)
        n_samples, data_idx = 0, 0
        for image_file in tqdm(images_paths, desc=f"Processing {data_title} Data"):
            if image_file.suffix != ".jpg":
                continue
            label_file = labels_path / image_file.name
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
                while data_idx < len(sizes) and sizes[data_idx] == 0:
                    data_idx += 1
        
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
