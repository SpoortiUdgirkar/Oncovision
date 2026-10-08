"""
Dataset Preparation Script for BUSI (Dataset of Breast Ultrasound Images).

Extracts C:\\Users\\spoor\\Downloads\\archive.zip, filters out mask files,
performs a stratified split (70% train, 15% validation, 15% test) with SEED=42,
and populates dataset/train, dataset/val, and dataset/test directories.

Usage:
    python scripts/prepare_busi_dataset.py --zip_path C:\\Users\\spoor\\Downloads\\archive.zip
"""

import argparse
import os
import sys
import shutil
import zipfile
import random
from pathlib import Path

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATASET_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    CLASS_NAMES,
    VALID_IMAGE_EXTENSIONS,
    SEED
)
from scripts.verify_dataset import verify_dataset
from scripts.test_dataloader import test_dataloader_shapes


def extract_and_split_dataset(
    zip_path=r"C:\Users\spoor\Downloads\archive.zip",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=SEED
):
    """
    Extracts zip archive, filters out mask files, performs stratified split across classes,
    and populates train/val/test directories.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive file not found at: '{zip_path}'")

    print("=" * 70)
    print("      ONCOVISION - EXTRACTING AND PREPARING REAL BUSI DATASET      ")
    print("=" * 70)

    # 1. Clear existing contents in train, val, test
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        for cls_name in CLASS_NAMES:
            (split_dir / cls_name).mkdir(parents=True, exist_ok=True)

    # Temporary extract folder
    temp_extract_dir = DATASET_DIR / "_temp_extract"
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting zip archive: {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)

    # 2. Find root directory inside extracted files
    busi_root = None
    for root, dirs, files in os.walk(temp_extract_dir):
        # Check if subfolders benign, malignant, normal exist in root
        lower_dirs = [d.lower() for d in dirs]
        if "benign" in lower_dirs or "malignant" in lower_dirs:
            busi_root = Path(root)
            break

    if busi_root is None:
        busi_root = temp_extract_dir

    print(f"Dataset root identified at: {busi_root}")

    # Set random seed for reproducible stratified split
    random.seed(seed)

    total_copied = 0
    split_counts = {"train": 0, "val": 0, "test": 0}

    # 3. Process each target class (Normal, Benign, Malignant)
    for class_name in CLASS_NAMES:
        # Find corresponding source class folder
        source_folder = None
        for child in busi_root.iterdir():
            if child.is_dir() and child.name.lower() == class_name.lower():
                source_folder = child
                break

        if source_folder is None or not source_folder.exists():
            print(f"[WARNING] Could not find folder for class '{class_name}' in zip archive.")
            continue

        # Collect valid image files (excluding masks)
        valid_images = []
        for file_path in source_folder.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                continue
            if "_mask" in file_path.stem.lower():
                continue
            valid_images.append(file_path)

        # Shuffle deterministically
        random.shuffle(valid_images)
        n_total = len(valid_images)

        # Calculate split indices
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # Remaining goes to test to account for rounding
        n_test = n_total - n_train - n_val

        train_imgs = valid_images[:n_train]
        val_imgs = valid_images[n_train:n_train + n_val]
        test_imgs = valid_images[n_train + n_val:]

        print(f"Class '{class_name:<9}': {n_total} images -> Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

        # Copy files to respective split directories
        for img_path in train_imgs:
            shutil.copy2(img_path, TRAIN_DIR / class_name / img_path.name)
            split_counts["train"] += 1
            total_copied += 1

        for img_path in val_imgs:
            shutil.copy2(img_path, VAL_DIR / class_name / img_path.name)
            split_counts["val"] += 1
            total_copied += 1

        for img_path in test_imgs:
            shutil.copy2(img_path, TEST_DIR / class_name / img_path.name)
            split_counts["test"] += 1
            total_copied += 1

    # Cleanup temporary extraction directory
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)

    print(f"\nSuccessfully extracted and split {total_copied} real ultrasound images into dataset/")
    print(f"Train: {split_counts['train']} | Val: {split_counts['val']} | Test: {split_counts['test']}\n")

    # 4. Run automated verification & dataloader shape tests on extracted dataset
    verify_dataset(TRAIN_DIR, VAL_DIR, TEST_DIR)
    test_dataloader_shapes(batch_size=32)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare real BUSI dataset from zip file.")
    parser.add_argument(
        "--zip_path",
        type=str,
        default=r"C:\Users\spoor\Downloads\archive.zip",
        help="Path to downloaded archive.zip"
    )
    args = parser.parse_args()

    extract_and_split_dataset(zip_path=args.zip_path)
