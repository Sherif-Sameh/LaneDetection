from pathlib import Path

import cv2
from flax import nnx
import jax.numpy as jnp
import jax.nn as nn

from lane_detection.models import VGG16
from lane_detection.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToArray,
)


def main():
    # Load the sample images from ImageNet
    image_dir = Path(__file__).parent / "images"
    maltese_dog = cv2.imread(str(image_dir / "Maltese_dog.JPEG"))
    tiger_shark = cv2.imread(str(image_dir / "tiger_shark.JPEG"))
    maltese_dog = cv2.cvtColor(maltese_dog, cv2.COLOR_BGR2RGB)
    tiger_shark = cv2.cvtColor(tiger_shark, cv2.COLOR_BGR2RGB)

    # Create and apply image transforms
    transforms = Compose.create([
        ToArray.create(),
        Resize.create(256),
        CenterCrop.create(224),
        Normalize.create(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    maltese_dog = transforms(maltese_dog)
    tiger_shark = transforms(tiger_shark)
    images = jnp.concat([maltese_dog, tiger_shark])

    # Create and apply VGG16-BN model with pre-trained weights
    model = VGG16(batch_norm=True, pretrained=True, rngs=nnx.Rngs(0))
    model.eval()
    preds = model(images)
    probs = nn.softmax(preds, axis=-1)
    probs = jnp.max(probs, axis=-1)
    preds = jnp.argmax(preds, axis=-1)

    # Verify the expected labels of each model
    print("Maltese Dog:")
    print("\tExpected Label 153")
    print(f"\tPredicted Label {preds[0]} with probability {probs[0]:.3f}\n")

    print("Tiger Shark:")
    print("\tExpected Label 3")
    print(f"\tPredicted Label {preds[1]} with probability {probs[1]:.3f}\n")

if __name__ == "__main__":
    main()
