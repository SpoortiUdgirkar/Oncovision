"""
Data Preprocessing and Augmentation Pipeline for Breast Ultrasound Images.

This module provides torchvision transformation pipelines tailored for breast ultrasound imaging:
1. Training Transforms: Includes mild, anatomically valid data augmentations (horizontal flip, small rotation, mild gain adjustment).
2. Validation & Test Transforms: Applies deterministic resizing and normalization without random augmentations.
3. Denormalization Utility: Converts normalized PyTorch tensors back to RGB displayable numpy arrays/tensors for visualization.
"""

import torch
from torchvision import transforms
from src.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


def get_train_transforms(
    image_size=IMAGE_SIZE,
    mean=NORMALIZE_MEAN,
    std=NORMALIZE_STD
):
    """
    Returns PyTorch torchvision transforms for the training dataset.

    Augmentations applied:
    - Resize: Standardizes all ultrasound scans to (224, 224).
    - Random Horizontal Flip: Safe for ultrasound scans as lateral orientation 
      does not change acoustic shadows or margin boundaries.
    - Random Rotation (+/- 15 degrees): Simulates realistic transducer positioning angles.
    - Mild ColorJitter (brightness 0.1, contrast 0.1): Simulates minor ultrasound machine 
      gain/contrast settings variation.
    - ToTensor: Scales pixel values from [0, 255] to [0.0, 1.0].
    - Normalize: Normalizes image channels using ImageNet pretrained stats.

    Unrealistic transformations excluded:
    - Vertical flip: Violates anatomical acoustic depth (skin line vs chest wall).
    - Extreme distortion/warping: Alters tumor boundary geometry and calcifications.
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_val_test_transforms(
    image_size=IMAGE_SIZE,
    mean=NORMALIZE_MEAN,
    std=NORMALIZE_STD
):
    """
    Returns PyTorch torchvision transforms for validation and testing datasets.

    Deterministic pipeline:
    - Resize to target size (224, 224).
    - Convert PIL Image to PyTorch Float Tensor [3, 224, 224].
    - Standard ImageNet Normalization.
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def denormalize(tensor, mean=NORMALIZE_MEAN, std=NORMALIZE_STD):
    """
    Denormalizes a PyTorch image tensor [3, H, W] or batch tensor [B, 3, H, W]
    back to the range [0.0, 1.0] for plotting and visual verification.

    Args:
        tensor (torch.Tensor): Image tensor of shape (3, H, W) or (B, 3, H, W).
        mean (list): Normalization mean used during preprocessing.
        std (list): Normalization standard deviation used during preprocessing.

    Returns:
        torch.Tensor: Denormalized image tensor clamped to [0.0, 1.0].
    """
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)

    if tensor.ndim == 4:
        mean_tensor = mean_tensor.unsqueeze(0)
        std_tensor = std_tensor.unsqueeze(0)

    # Denormalize: original = (normalized * std) + mean
    denorm_tensor = tensor * std_tensor + mean_tensor
    return torch.clamp(denorm_tensor, 0.0, 1.0)
