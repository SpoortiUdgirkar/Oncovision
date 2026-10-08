"""
PyTorch Dataset and DataLoader Implementation for Breast Ultrasound Classification.

This module provides:
1. BreastUltrasoundDataset: A custom PyTorch Dataset that loads ultrasound scans,
   handles RGB conversion, excludes segmentation masks (e.g., *_mask.png),
   verifies file integrity, and applies torchvision preprocessing/augmentation transforms.
2. get_dataloaders: Factory function to create training, validation, and test PyTorch DataLoaders.
"""

from pathlib import Path
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    VALID_IMAGE_EXTENSIONS,
    BATCH_SIZE,
    NUM_WORKERS,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
)
from src.preprocessing.transforms import get_train_transforms, get_val_test_transforms


class BreastUltrasoundDataset(Dataset):
    """
    Custom PyTorch Dataset for Breast Ultrasound Images.

    Supports three classes:
    - Normal (0)
    - Benign (1)
    - Malignant (2)
    """

    def __init__(
        self,
        data_dir,
        transform=None,
        class_names=CLASS_NAMES,
        exclude_masks=True
    ):
        """
        Args:
            data_dir (str or Path): Path to split directory (containing Normal, Benign, Malignant subfolders).
            transform (callable, optional): Torchvision transform pipeline to apply to images.
            class_names (list): List of class strings in target order.
            exclude_masks (bool): Whether to ignore segmentation mask files (e.g. *_mask.png).
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_names = class_names
        self.exclude_masks = exclude_masks
        self.samples = []
        self.corrupted_files = []

        self._load_samples()

    def _load_samples(self):
        """
        Scans the directory structure, maps subfolders to labels,
        filters invalid/corrupted files, and populates self.samples.
        """
        if not self.data_dir.exists():
            print(f"Warning: Directory '{self.data_dir}' does not exist.")
            return

        for idx, class_name in enumerate(self.class_names):
            # Check for folder matching class name (case-insensitive search)
            class_folder = None
            for child in self.data_dir.iterdir():
                if child.is_dir() and child.name.lower() == class_name.lower():
                    class_folder = child
                    break

            if class_folder is None or not class_folder.exists():
                continue

            # Iterate over files in class folder
            for img_path in class_folder.iterdir():
                if not img_path.is_file():
                    continue

                # Check supported file extension
                if img_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                    continue

                # Exclude segmentation mask files (common in datasets like BUSI)
                if self.exclude_masks and "_mask" in img_path.stem.lower():
                    continue

                # Verify image readable & not corrupted
                try:
                    with Image.open(img_path) as img:
                        img.verify()  # Quick verify header integrity
                    self.samples.append((img_path, idx))
                except (UnidentifiedImageError, OSError, Exception) as e:
                    self.corrupted_files.append((img_path, str(e)))

    def __len__(self):
        """Returns total number of valid images in dataset split."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves the item at index `idx`.

        Returns:
            tuple: (image_tensor, label_tensor)
        """
        img_path, label = self.samples[idx]

        try:
            # Always open and convert image to RGB 3-channel
            with Image.open(img_path) as img:
                img = img.convert("RGB")

                if self.transform:
                    image_tensor = self.transform(img)
                else:
                    image_tensor = img

                return image_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            raise RuntimeError(f"Error loading image '{img_path}': {str(e)}")

    def get_class_counts(self):
        """Returns dictionary of class name to sample count in this dataset split."""
        counts = {cls_name: 0 for cls_name in self.class_names}
        for _, label_idx in self.samples:
            cls_name = self.class_names[label_idx]
            counts[cls_name] += 1
        return counts

    def get_image_paths(self):
        """Returns list of all image paths in this dataset split."""
        return [path for path, _ in self.samples]


def create_dataloaders(
    train_dir=TRAIN_DIR,
    val_dir=VAL_DIR,
    test_dir=TEST_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_transform=None,
    val_test_transform=None
):
    """
    Factory function to build train, validation, and test PyTorch DataLoaders.

    Args:
        train_dir (str or Path): Path to training set directory.
        val_dir (str or Path): Path to validation set directory.
        test_dir (str or Path): Path to test set directory.
        batch_size (int): Number of samples per batch.
        num_workers (int): Multi-process data loading worker count.
        train_transform (callable): Transforms for training. Defaults to get_train_transforms().
        val_test_transform (callable): Transforms for val/test. Defaults to get_val_test_transforms().

    Returns:
        tuple: (train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)
    """
    if train_transform is None:
        train_transform = get_train_transforms()
    if val_test_transform is None:
        val_test_transform = get_val_test_transforms()

    train_dataset = BreastUltrasoundDataset(train_dir, transform=train_transform)
    val_dataset = BreastUltrasoundDataset(val_dir, transform=val_test_transform)
    test_dataset = BreastUltrasoundDataset(test_dir, transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
