import os
from pathlib import Path
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import cv2
import imageio
import numpy as np
import toml
from flax import nnx
from jax import Array
from numpy.typing import NDArray

from lane_detection.dataloaders import NumPyLaneDataloader
from lane_detection.datasets import CULaneDataset
from lane_detection.models import SCNN, VGG11
from lane_detection.detectors import SCNNLaneDetector
from lane_detection.utils import convert_transforms


def get_dataloader(batch_size: int, test: bool = False) -> NumPyLaneDataloader:
    path = Path(__file__).parents[2] / "data/CULane"
    dataset = CULaneDataset(path, val=0.0, test=0.1)
    if not any(path.iterdir()):
        dataset.download()
    dataset.load(test=test)
    dataloader = NumPyLaneDataloader(dataset, batch_size=batch_size, shuffle=(not test))
    return dataloader


def get_VGG11_backbone(*, rngs: nnx.Rngs) -> nnx.List:
    # Load pre-trained VGG16-BN backbone
    backbone = VGG11(
        batch_norm=True, pretrained=True, rngs=nnx.Rngs(0)
    ).features

    # Replace convolutions in last block with dilated convolutions
    for i, layer in enumerate(backbone[-1].layers):
        if not isinstance(layer, nnx.Conv):
            continue
        backbone[-1].layers[i] = nnx.Conv(
            layer.in_features,
            layer.out_features,
            kernel_size=(3, 3),
            padding=(2, 2),
            kernel_dilation=(2, 2),
            use_bias=layer.use_bias,
            rngs=rngs,
        )
    
    # Remove the max pool layers at the end of blocks 4 and 5
    backbone[-2].layers.pop(-1)
    backbone[-1].layers.pop(-1)
    backbone = nnx.Sequential(*backbone)
    return backbone


def overlay_lane_masks(img: Array, pred_mask: Array, gt_mask: Array) -> NDArray:
    """Overaly lane masks onto RGB image, add text labels to each of them and then combine them
    into a single RGB image. 

    Args:
        img: (H, W, 3) uint8 array of input RGB image.
        pred_mask: (H, W) int array of predicted lane segmentation mask.
        gt_mask: (H, W) int array of ground truth lane segmentation mask.
        
    Returns:
        (H, 2 * W, 3) Concatenated RGB image (predicted | ground truth) as uint8 NumPy array.
    """
    alpha = 0.6
    label_colors = {
        1: np.array([0, 0, 255]),
        2: np.array([0, 255, 0]),
        3: np.array([255, 0, 0]),
        4: np.array([128, 128, 0]),
    }

    # Convert from JAX to NumPy
    img = np.asarray(img)
    pred_mask = np.asarray(pred_mask).astype(np.uint8)
    gt_mask = np.asarray(gt_mask).astype(np.uint8)
    
    # Convert image to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def make_overlay(mask):
        """Overlay lane masks on top of input image."""
        overlay = np.zeros_like(img)
        for lane_id, color in label_colors.items():
            if lane_id == 0:
                continue  # skip background
            overlay[mask == lane_id] = color
        blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
        return blended

    # Create overlay images
    pred_img = make_overlay(pred_mask)
    gt_img = make_overlay(gt_mask)

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    cv2.putText(pred_img, "SCNN", (30, 50), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(gt_img, "Ground Truth", (30, 50), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Concatenate prediction and GT images vertically
    combined = np.concatenate((pred_img, gt_img), axis=0)
    combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    return combined


def main():
    # Recover training configurations and convert transforms
    path = Path(__file__).parent / "configs/scnn.toml"
    config = toml.load(path)
    transform = convert_transforms(config["transform"])
    
    # Get CULane dataloaders for testing dataset
    dataloader = get_dataloader(batch_size=1, test=True)

    # Create video writer with imageio
    path = Path(__file__).parent / "videos"
    path.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path / "scnn.mp4", fps=10)
    
    # Initialize SCNN model with pre-trained weights
    model = SCNN(
        **config["scnn"], backbone=get_VGG11_backbone(rngs=nnx.Rngs(0)), rngs=nnx.Rngs(0)
    )
    model = model.load()

    # Initialize SCNN lane detector
    detector = SCNNLaneDetector(model, n_lanes=config["scnn"]["n_lanes"], transform=transform)

    # Iterate over testing dataset and store combined images to video
    for input, target in dataloader:
        pred = detector.detect_lanes(input)
        writer.append_data(overlay_lane_masks(input[0], pred[0], target[0]))
    
    # Write out video
    writer.close()
    print(f"Video saved to {str(path / 'scnn.mp4')}")


if __name__ == "__main__":
    main()
