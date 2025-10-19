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
    N_TRAIN_VAL_IMGS = 3626
    N_TEST_IMGS = 2782
    
    def __init__(
        self,path: Path,
        val: float = 0.1,
        height: int = 360,
        width: int = 640,
    ):
        super().__init__(path, val=val, test=0.0)
        self.height = height
        self.width = width
        self.files = ("images.npy", "labels.npy", "metadata.json")

        self.images = np.zeros((0, self.height, self.width, 3), dtype=np.uint8)
        self.labels = np.zeros((0, self.height, self.width), dtype=np.uint8)
    
    def __len__(self) -> int:
        """Return the number of samples available in the loaded dataset."""
        return self.images.shape[0]
    
    def __getitem__(self, idx: int | slice | tuple[int | slice, ...]) -> tuple[NDArray, NDArray]:
        """Retrieve data sample/s from the dataset and return them."""
        return self.images[idx], self.labels[idx]

    def download(self) -> None:
        """Download the TuSimple dataset with segmentation-based annotations.
    
        Note: The downloaded dataset comes from the pre-processed entry on Kaggle available at the
        following link:
        https://www.kaggle.com/datasets/hikmatullahmohammadi/tusimple-preprocessed.
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
        download_path = kagglehub.dataset_download("hikmatullahmohammadi/tusimple-preprocessed")
        download_path = Path(download_path)

        # Convert dataset to NumPy arrays and store it into desired directory
        self._convert(download_path)
    
    def load(self, val: bool = False, test: bool = False, use_mmap: bool = True) -> None:
        """Load the stored dataset to prepare for data reading.
        
        Args:
            val: Load the data allocated for validation instead of training.
            test: Load the data allocated for testing instead of training. If both val and test
                flags are set, then validation takes higher precedence over testing.
            use_mmap: Keep dataset on disk and load slices dynamically in RAM as needed.
        """
        assert self.exists(), f"Dataset does not exist at {self.paths[0].parent}."
        path = self.paths[2] if test else self.paths[0]
        path = self.paths[1] if val else path

        # Load dataset
        mmap_mode = "r" if use_mmap else None
        self.images = np.load(path / self.files[0], mmap_mode=mmap_mode)
        self.labels = np.load(path / self.files[1], mmap_mode=mmap_mode)
        with open(path / self.files[2], "r") as metafile:
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
    
    def _convert(self, download_path: Path) -> None:
        """Iterates through TuSimple dataset to extract all images and labels, resize them as needed
        and store them into .npy files.
    
        Args:
            download_path: Path to raw TuSimple dataset.
        """
        images_path = download_path / "tusimple_preprocessed/training/frames"
        labels_path = download_path / "tusimple_preprocessed/training/lane-masks"
        split_idxs = self._get_permutations(self.N_TRAIN_VAL_IMGS)
        
        # Process and store images and labels for training/validation dataset
        images, labels = self._get_images_labels(
            images_path, labels_path, self.N_TRAIN_VAL_IMGS, "Train/Val"
        )
        for p, idxs in zip(self.paths[:2], split_idxs):
            images_split = images[idxs]
            np.save(p / self.files[0], images_split)
            del images_split
            labels_split = labels[idxs]
            np.save(p / self.files[1], labels_split)
            del labels_split
        del images, labels

        # Process and store images and labels for testing dataset
        images_path = download_path / "tusimple_preprocessed/test/frames"
        labels_path = download_path / "tusimple_preprocessed/test/lane-masks"
        images, labels = self._get_images_labels(
            images_path, labels_path, self.N_TEST_IMGS, "Test"
        )
        np.save(self.paths[2] / self.files[0], images)
        np.save(self.paths[2] / self.files[1], labels)
        del images, labels

        # Store dataset's metadata
        sizes = [len(idxs) for idxs in split_idxs] + [self.N_TEST_IMGS]
        for p, size in zip(self.paths, sizes):
            metadata = {
                "size": size,
                "image_shape": (self.height, self.width, 3),
                "label_shape": (self.height, self.width),
            }
            with open(p / self.files[2], "w") as metafile: 
                json.dump(metadata, metafile, indent=4)
    
    def _get_permutations(self, size: int) -> tuple[NDArray, NDArray, NDArray]:
        """Get permutations for shuffling and splitting the training dataset into train and val.
        
        Returns:
            tuple
            - train_idxs: array containing indices for training samples.
            - val_idxs: array containing indices for validation samples.
        """
        train_size = int(size * self.split[0])
        val_size = size - train_size
        split_cumsum = np.cumsum([train_size, val_size], dtype=np.long)

        perm = np.random.permutation(size)
        train_idxs = perm[:split_cumsum[0]]
        val_idxs = perm[split_cumsum[0]: split_cumsum[1]]
        return train_idxs, val_idxs
    
    def _get_images_labels(
        self,
        images_path: Path,
        labels_path: Path,
        size: int,
        data_title: str,
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Loop image and label directories, get all entries, pre-process them and return them as
        NumPy arrays.
        
        Args:
            images_path: Path to the directory containing the dataset's images.
            labels_path: Path to the directory containing the dataset's labels.
            size: Number of expected image and label pairs in the dataset.
            data_title: Title to display with tqdm while processing the dataset.
        
        Returns:
            tuple
            images: (N, H, W, 3) uint8 array containing the processed dataset RGB images.
            labels: (N, H, W) uint8 array containing the processed dataset lane labels.
        """
        images_shape = (size, self.height, self.width, 3)
        labels_shape = images_shape[:-1]  # labels don't need a channel dimension
        images = np.zeros(images_shape, dtype=np.uint8)
        labels = np.zeros(labels_shape, dtype=np.uint8)
        image_idx = 0
        for image_file in tqdm(
            sorted(images_path.iterdir()), desc=f"Processing {data_title} Data"
        ):
            if image_file.suffix != ".jpg":
                continue
            label_file = labels_path / image_file.name
            assert label_file.exists(), f"Label missing for {image_file.name}"

            image = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            label = cv2.cvtColor(cv2.imread(label_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            images[image_idx] = cv2.resize(image, (self.width, self.height))
            labels[image_idx] = cv2.resize(label, (self.width, self.height))[..., 0]
            image_idx += 1
        assert image_idx == size, f"Missing images from dataset. Found {image_idx}/{size}."
        return images, labels

def main(data_dir: str, val: float = 0.1):
    dataset = TuSimpleDataset(Path(data_dir), val=val)
    dataset.download()


if __name__ == "__main__":
    fire.Fire(main)