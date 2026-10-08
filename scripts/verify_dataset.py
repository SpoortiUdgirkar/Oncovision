"""
Dataset Verification Script for OncoVision (Breast Ultrasound Classification).

This script performs comprehensive dataset integrity, split verification, and visualization checks:
1. Counts total images, class distribution (Normal, Benign, Malignant), and train/val/test split counts.
2. Checks for corrupted images and unsupported file formats.
3. Evaluates class imbalance ratio and outputs warnings if severe imbalance exists.
4. Prevents data leakage by detecting exact duplicate images across train, validation, and test splits using MD5 hashing.
5. Displays and saves sample images with their class labels to outputs/sample_verification.png.

Usage:
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --dataset_dir dataset/
"""

import argparse
import hashlib
import sys
from pathlib import Path

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import torch
from PIL import Image, UnidentifiedImageError

from src.config import (
    DATASET_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    OUTPUTS_DIR,
    CLASS_NAMES,
    VALID_IMAGE_EXTENSIONS
)

from src.preprocessing.dataset import BreastUltrasoundDataset, create_dataloaders
from src.preprocessing.transforms import denormalize


def compute_file_hash(file_path):
    """Computes MD5 hash of a file to identify duplicate image content across dataset splits."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def check_directory_integrity(data_dir, class_names=CLASS_NAMES):
    """
    Scans a split directory for:
    - Supported vs unsupported file extensions
    - Corrupted or unreadable image files
    - Mask files (logged for information)
    """
    data_dir = Path(data_dir)
    valid_files = []
    unsupported_files = []
    corrupted_files = []
    mask_files = []
    class_counts = {cls_name: 0 for cls_name in class_names}
    file_hashes = {}  # hash -> file_path

    if not data_dir.exists():
        return {
            "valid_count": 0,
            "unsupported": [],
            "corrupted": [],
            "masks": [],
            "class_counts": class_counts,
            "hashes": file_hashes,
            "files": []
        }

    for class_name in class_names:
        # Case-insensitive folder lookup
        class_folder = None
        for child in data_dir.iterdir():
            if child.is_dir() and child.name.lower() == class_name.lower():
                class_folder = child
                break

        if class_folder is None or not class_folder.exists():
            continue

        for img_path in class_folder.iterdir():
            if not img_path.is_file():
                continue

            ext = img_path.suffix.lower()
            if ext not in VALID_IMAGE_EXTENSIONS:
                unsupported_files.append(img_path)
                continue

            if "_mask" in img_path.stem.lower():
                mask_files.append(img_path)
                continue

            # Verify image file integrity
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_files.append((img_path, class_name))
                class_counts[class_name] += 1
                file_hashes[compute_file_hash(img_path)] = img_path
            except (UnidentifiedImageError, OSError, Exception) as e:
                corrupted_files.append((img_path, str(e)))

    return {
        "valid_count": len(valid_files),
        "unsupported": unsupported_files,
        "corrupted": corrupted_files,
        "masks": mask_files,
        "class_counts": class_counts,
        "hashes": file_hashes,
        "files": valid_files
    }


def verify_dataset(
    train_dir=TRAIN_DIR,
    val_dir=VAL_DIR,
    test_dir=TEST_DIR,
    output_dir=OUTPUTS_DIR
):
    """
    Executes full dataset verification pipeline and reports status.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("        ONCOVISION - DATASET VERIFICATION & PREPROCESSING REPORT        ")
    print("=" * 70)

    # 1. Directory Integrity Scans
    train_info = check_directory_integrity(train_dir)
    val_info = check_directory_integrity(val_dir)
    test_info = check_directory_integrity(test_dir)

    total_images = train_info["valid_count"] + val_info["valid_count"] + test_info["valid_count"]

    if total_images == 0:
        print("\n[WARNING] No valid images found in the dataset directories.")
        print(f"Expected structure:\n  {train_dir}/[Normal|Benign|Malignant]\n  {val_dir}/[Normal|Benign|Malignant]\n  {test_dir}/[Normal|Benign|Malignant]")
        print("\nTip: Run 'python scripts/create_dummy_dataset.py' to generate synthetic samples for testing.")
        print("=" * 70)
        return False

    # 2. Split Counts Printout
    print("\n--- 1. DATASET SPLIT SUMMARY ---")
    print(f"Total Valid Images    : {total_images}")
    print(f"Training Images       : {train_info['valid_count']} ({train_info['valid_count']/total_images*100:.1f}%)")
    print(f"Validation Images     : {val_info['valid_count']} ({val_info['valid_count']/total_images*100:.1f}%)")
    print(f"Testing Images        : {test_info['valid_count']} ({test_info['valid_count']/total_images*100:.1f}%)")

    # 3. Class Counts Printout
    overall_class_counts = {cls_name: 0 for cls_name in CLASS_NAMES}
    for cls_name in CLASS_NAMES:
        overall_class_counts[cls_name] = (
            train_info["class_counts"][cls_name] +
            val_info["class_counts"][cls_name] +
            test_info["class_counts"][cls_name]
        )

    print("\n--- 2. CLASS DISTRIBUTION SUMMARY ---")
    print(f"{'Class Name':<15} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8} | {'Ratio (%)':<10}")
    print("-" * 68)
    for cls_name in CLASS_NAMES:
        t_cnt = train_info["class_counts"][cls_name]
        v_cnt = val_info["class_counts"][cls_name]
        ts_cnt = test_info["class_counts"][cls_name]
        tot = overall_class_counts[cls_name]
        pct = (tot / total_images * 100) if total_images > 0 else 0
        print(f"{cls_name:<15} | {t_cnt:<8} | {v_cnt:<8} | {ts_cnt:<8} | {tot:<8} | {pct:<10.2f}%")

    # 4. Integrity Checks: Unsupported & Corrupted Files
    total_unsupported = len(train_info["unsupported"]) + len(val_info["unsupported"]) + len(test_info["unsupported"])
    total_corrupted = len(train_info["corrupted"]) + len(val_info["corrupted"]) + len(test_info["corrupted"])

    print("\n--- 3. DATASET INTEGRITY & FILE FORMAT CHECK ---")
    print(f"Unsupported File Formats Found : {total_unsupported}")
    if total_unsupported > 0:
        print("  Unsupported files:")
        for uf in train_info["unsupported"] + val_info["unsupported"] + test_info["unsupported"]:
            print(f"    - {uf}")

    print(f"Corrupted Images Found         : {total_corrupted}")
    if total_corrupted > 0:
        print("  Corrupted files details:")
        for cf, err in train_info["corrupted"] + val_info["corrupted"] + test_info["corrupted"]:
            print(f"    - {cf}: {err}")

    # 5. Class Imbalance Check
    print("\n--- 4. CLASS IMBALANCE EVALUATION ---")
    counts_list = list(overall_class_counts.values())
    if min(counts_list) == 0:
        print("[ALERT] Class imbalance severe: At least one class has 0 images!")
    else:
        imbalance_ratio = max(counts_list) / min(counts_list)
        print(f"Max-to-Min Class Count Ratio   : {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 2.5:
            print("[NOTICE] Moderate to high class imbalance detected. Consider using weighted loss (e.g. CrossEntropyLoss weights) or class-balanced sampling in Phase 4.")
        else:
            print("[INFO] Class distribution is relatively balanced.")

    # 6. Data Leakage Check across splits
    print("\n--- 5. DATA LEAKAGE PREVENTION CHECK ---")
    train_hashes = set(train_info["hashes"].keys())
    val_hashes = set(val_info["hashes"].keys())
    test_hashes = set(test_info["hashes"].keys())

    tv_overlap = train_hashes.intersection(val_hashes)
    tt_overlap = train_hashes.intersection(test_hashes)
    vt_overlap = val_hashes.intersection(test_hashes)

    total_leakage = len(tv_overlap) + len(tt_overlap) + len(vt_overlap)
    if total_leakage > 0:
        print(f"[CRITICAL WARNING] DATA LEAKAGE DETECTED! Found {total_leakage} duplicate image files across splits.")
        if tv_overlap:
            print(f"  - Train & Val overlap: {len(tv_overlap)} images")
        if tt_overlap:
            print(f"  - Train & Test overlap: {len(tt_overlap)} images")
        if vt_overlap:
            print(f"  - Val & Test overlap: {len(vt_overlap)} images")
    else:
        print("[PASSED] Data leakage check passed! Zero duplicate image hashes found between train, validation, and test splits.")

    # 7. Visualization of Sample Images
    print("\n--- 6. VISUALIZING SAMPLE IMAGES WITH LABELS ---")
    try:
        train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_dataloaders(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            batch_size=9,
            num_workers=0
        )

        if len(train_ds) > 0:
            images, labels = next(iter(train_loader))
            denorm_images = denormalize(images)

            fig, axes = plt.subplots(3, 3, figsize=(10, 10))
            fig.suptitle("OncoVision - Preprocessed & Augmented Sample Ultrasound Images", fontsize=14, fontweight="bold")

            num_to_plot = min(len(images), 9)
            for i in range(9):
                ax = axes[i // 3, i % 3]
                if i < num_to_plot:
                    img_np = denorm_images[i].permute(1, 2, 0).numpy()
                    label_idx = labels[i].item()
                    label_name = CLASS_NAMES[label_idx]

                    ax.imshow(img_np)
                    ax.set_title(f"Label: {label_name} ({label_idx})", fontsize=11, color="darkblue")
                ax.axis("off")

            plt.tight_layout()
            sample_plot_path = output_dir / "sample_verification.png"
            plt.savefig(sample_plot_path, dpi=150)
            plt.close()
            print(f"[SUCCESS] Sample image grid saved to: {sample_plot_path}")

    except Exception as e:
        print(f"[WARNING] Visual sampling encountered error: {str(e)}")

    print("\n" + "=" * 70)
    print("                    DATASET VERIFICATION COMPLETE                       ")
    print("=" * 70 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify dataset structure, formats, leakage, and sample visualization.")
    parser.add_argument("--train_dir", type=str, default=str(TRAIN_DIR), help="Path to training set directory.")
    parser.add_argument("--val_dir", type=str, default=str(VAL_DIR), help="Path to validation set directory.")
    parser.add_argument("--test_dir", type=str, default=str(TEST_DIR), help="Path to test set directory.")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR), help="Path to outputs directory.")

    args = parser.parse_args()
    verify_dataset(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir
    )
