"""
Dummy Breast Ultrasound Dataset Generator Script.

Creates synthetic sample ultrasound images in:
- dataset/train/Normal, dataset/train/Benign, dataset/train/Malignant
- dataset/val/Normal, dataset/val/Benign, dataset/val/Malignant
- dataset/test/Normal, dataset/test/Benign, dataset/test/Malignant

Use this script to populate the dataset directory with synthetic sample images
for testing data pipelines, verification scripts, and model initializations.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from src.config import TRAIN_DIR, VAL_DIR, TEST_DIR, CLASS_NAMES



def create_synthetic_ultrasound_image(class_name, seed=0):
    """
    Generates a synthetic 224x224 grayscale ultrasound-style image with noise
    and simulated focal lesions based on class type.
    """
    np.random.seed(seed)
    # Base acoustic speckle noise pattern (typical in ultrasound)
    img_arr = np.random.gamma(shape=2.0, scale=30.0, size=(224, 224)).astype(np.float32)
    
    # Add tissue layer gradient (dark top skin line, acoustic attenuation downwards)
    y_grid, x_grid = np.ogrid[:224, :224]
    gradient = np.exp(-y_grid / 150.0)
    img_arr = img_arr * gradient

    # Add class-specific lesion characteristics
    center_y, center_x = 112, 112
    dist_sq = (y_grid - center_y)**2 + (x_grid - center_x)**2

    if class_name.lower() == "benign":
        # Oval, well-circumscribed hypoechoic mass (dark center)
        mask = dist_sq < 30**2
        img_arr[mask] *= 0.4
    elif class_name.lower() == "malignant":
        # Irregular, microlobulated/spiculated hypoechoic lesion with posterior acoustic shadowing
        mask = dist_sq < 35**2
        # Spiculation noise
        spicules = (np.sin(x_grid / 5.0) * np.cos(y_grid / 5.0)) > 0
        img_arr[mask & spicules] *= 0.25
        # Posterior shadow
        shadow_mask = (y_grid > center_y + 20) & (np.abs(x_grid - center_x) < 30)
        img_arr[shadow_mask] *= 0.5
    # Normal ultrasound has uniform glandular speckle without discrete mass

    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(img_arr, mode="L")


def generate_dummy_dataset(
    train_dir=TRAIN_DIR,
    val_dir=VAL_DIR,
    test_dir=TEST_DIR,
    samples_per_class=(12, 4, 4)
):
    """
    Generates synthetic dummy dataset split into train, val, and test sets.
    """
    splits = [
        (train_dir, samples_per_class[0], "train"),
        (val_dir, samples_per_class[1], "val"),
        (test_dir, samples_per_class[2], "test"),
    ]

    total_created = 0

    for split_dir, num_samples, split_name in splits:
        split_dir = Path(split_dir)
        for class_name in CLASS_NAMES:
            class_folder = split_dir / class_name
            class_folder.mkdir(parents=True, exist_ok=True)

            for i in range(num_samples):
                seed_val = hash(f"{split_name}_{class_name}_{i}") % (2**31 - 1)
                img = create_synthetic_ultrasound_image(class_name, seed=seed_val)
                file_path = class_folder / f"{class_name.lower()}_{split_name}_{i+1:03d}.png"
                img.save(file_path)
                total_created += 1

    print(f"Successfully generated {total_created} synthetic ultrasound images across train/val/test splits.")
    print(f"Target directory: {train_dir.parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic breast ultrasound dataset for testing.")
    parser.add_argument("--train_count", type=int, default=12, help="Number of train samples per class.")
    parser.add_argument("--val_count", type=int, default=4, help="Number of val samples per class.")
    parser.add_argument("--test_count", type=int, default=4, help="Number of test samples per class.")

    args = parser.parse_args()
    generate_dummy_dataset(
        samples_per_class=(args.train_count, args.val_count, args.test_count)
    )
