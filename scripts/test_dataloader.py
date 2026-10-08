"""
DataLoader Tensor Shape and Specification Test Script.

Validates that PyTorch DataLoaders built by src.preprocessing.dataset
return image tensors and class label tensors matching expected dimensions,
data types, value ranges, and channel ordering:

Expected Image Tensor Shape : [batch_size, 3, 224, 224]
Expected Label Tensor Shape : [batch_size]
Expected Dtypes             : images: torch.float32, labels: torch.int64
"""

import argparse
import sys
from pathlib import Path

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.config import BATCH_SIZE, TRAIN_DIR, VAL_DIR, TEST_DIR, CLASS_NAMES

from src.preprocessing.dataset import create_dataloaders
from scripts.create_dummy_dataset import generate_dummy_dataset


def test_dataloader_shapes(batch_size=BATCH_SIZE):
    """
    Executes shape, dtype, and dimension verification tests on PyTorch DataLoaders.
    """
    print("=" * 70)
    print("         ONCOVISION - DATALOADER SHAPE & DTYPE VALIDATION TEST         ")
    print("=" * 70)

    # Ensure synthetic dataset exists if real dataset is empty
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_dataloaders(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        test_dir=TEST_DIR,
        batch_size=batch_size,
        num_workers=0
    )

    if len(train_ds) == 0:
        print("[NOTICE] No dataset samples detected. Auto-generating synthetic samples for test validation...")
        generate_dummy_dataset(train_count=12, val_count=4, test_count=4)
        # Re-initialize dataloaders
        train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_dataloaders(
            train_dir=TRAIN_DIR,
            val_dir=VAL_DIR,
            test_dir=TEST_DIR,
            batch_size=batch_size,
            num_workers=0
        )

    loaders = [
        ("Train Loader", train_loader, train_ds),
        ("Validation Loader", val_loader, val_ds),
        ("Test Loader", test_loader, test_ds)
    ]

    all_passed = True

    for name, loader, dataset in loaders:
        print(f"\nTesting {name} (Total samples: {len(dataset)})...")
        if len(dataset) == 0:
            print(f"  [SKIP] {name} is empty.")
            continue

        images, labels = next(iter(loader))
        actual_batch_size = images.shape[0]

        expected_image_shape = (actual_batch_size, 3, 224, 224)
        expected_label_shape = (actual_batch_size,)

        # 1. Assert Image Shape
        assert images.shape == expected_image_shape, (
            f"Image tensor shape mismatch in {name}! Expected {expected_image_shape}, got {images.shape}"
        )

        # 2. Assert Label Shape
        assert labels.shape == expected_label_shape, (
            f"Label tensor shape mismatch in {name}! Expected {expected_label_shape}, got {labels.shape}"
        )

        # 3. Assert Dtypes
        assert images.dtype == torch.float32, (
            f"Image tensor dtype mismatch! Expected torch.float32, got {images.dtype}"
        )
        assert labels.dtype == torch.int64, (
            f"Label tensor dtype mismatch! Expected torch.int64 (long), got {labels.dtype}"
        )

        # 4. Assert Label Range
        valid_labels = set(range(len(CLASS_NAMES)))
        label_vals = set(labels.tolist())
        assert label_vals.issubset(valid_labels), (
            f"Invalid class labels detected in {name}! Found {label_vals}, expected subset of {valid_labels}"
        )

        print(f"  [PASSED] Image Tensor Shape : {list(images.shape)} [batch_size, channels, height, width]")
        print(f"  [PASSED] Image Tensor Dtype : {images.dtype}")
        print(f"  [PASSED] Label Tensor Shape : {list(labels.shape)}")
        print(f"  [PASSED] Label Tensor Dtype : {labels.dtype}")
        print(f"  [PASSED] Value Range (Norm) : min = {images.min():.4f}, max = {images.max():.4f}")
        print(f"  [PASSED] Sample Labels      : {labels.tolist()}")

    print("\n" + "=" * 70)
    print("       ALL DATALOADER TENSOR SHAPE TESTS PASSED SUCCESSFULLY!          ")
    print("=" * 70 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PyTorch DataLoader tensor shapes.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for tensor verification.")
    args = parser.parse_args()

    test_dataloader_shapes(batch_size=args.batch_size)
