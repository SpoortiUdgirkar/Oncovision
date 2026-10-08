"""
EfficientNet-B0 Transfer Learning Architecture for Breast Ultrasound Classification.

This module provides:
1. build_efficientnet_b0: Factory function to instantiate EfficientNet-B0 with custom classification head,
   pretrained ImageNet weights, and configurable backbone freezing.
2. unfreeze_efficientnet_layers: Utility to unfreeze specific deeper blocks for fine-tuning.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from src.config import NUM_CLASSES


def build_efficientnet_b0(
    num_classes=NUM_CLASSES,
    pretrained=True,
    freeze_backbone=True,
    dropout_rate=0.3
):
    """
    Constructs an EfficientNet-B0 model configured for transfer learning.

    Args:
        num_classes (int): Number of target output classes (3: Normal, Benign, Malignant).
        pretrained (bool): Whether to load ImageNet pretrained weights.
        freeze_backbone (bool): If True, freezes backbone layers for feature extraction training.
        dropout_rate (float): Dropout probability applied before final linear layer.

    Returns:
        nn.Module: Configured EfficientNet-B0 model.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    # Freeze backbone parameters initially if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace original 1000-class classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )

    # Ensure classification head parameters are trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def unfreeze_efficientnet_layers(model, unfreeze_from="features.7"):
    """
    Unfreezes selected deeper blocks of EfficientNet-B0 for fine-tuning.

    Args:
        model (nn.Module): EfficientNet-B0 model instance.
        unfreeze_from (str): Layer string ("features.7", "features.6", or "all").
    """
    if unfreeze_from == "all":
        for param in model.parameters():
            param.requires_grad = True
        print("[INFO] All EfficientNet-B0 layers unfrozen for end-to-end fine-tuning.")
        return

    # Unfreeze final classification head always
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Map layer string to features stage
    if unfreeze_from in ["features.7", "layer7"]:
        target_stages = [model.features[7]]
    elif unfreeze_from in ["features.6", "layer6"]:
        target_stages = [model.features[6], model.features[7]]
    else:
        # Default unfreeze last 2 blocks if unknown
        target_stages = [model.features[6], model.features[7]]

    for stage in target_stages:
        for param in stage.parameters():
            param.requires_grad = True

    print(f"[INFO] EfficientNet-B0 layers unfrozen from '{unfreeze_from}' onwards.")
