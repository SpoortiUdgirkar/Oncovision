"""
DenseNet121 Transfer Learning Architecture for Breast Ultrasound Classification.

This module provides:
1. build_densenet121: Factory function to instantiate DenseNet121 with custom classification head,
   pretrained ImageNet weights, and configurable backbone freezing.
2. unfreeze_densenet_layers: Utility to unfreeze specific deeper blocks for fine-tuning.
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from src.config import NUM_CLASSES


def build_densenet121(
    num_classes=NUM_CLASSES,
    pretrained=True,
    freeze_backbone=True,
    dropout_rate=0.3
):
    """
    Constructs a DenseNet121 model configured for transfer learning.

    Args:
        num_classes (int): Number of target output classes (3: Normal, Benign, Malignant).
        pretrained (bool): Whether to load ImageNet pretrained weights.
        freeze_backbone (bool): If True, freezes backbone layers for feature extraction training.
        dropout_rate (float): Dropout probability applied before final linear layer.

    Returns:
        nn.Module: Configured DenseNet121 model.
    """
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)

    # Freeze backbone parameters initially if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace original 1000-class classifier head
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )

    # Ensure classification head parameters are trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def unfreeze_densenet_layers(model, unfreeze_from="denseblock4"):
    """
    Unfreezes selected deeper dense blocks of DenseNet121 for fine-tuning.

    Args:
        model (nn.Module): DenseNet121 model instance.
        unfreeze_from (str): Layer string ("denseblock4", "denseblock3", or "all").
    """
    if unfreeze_from == "all":
        for param in model.parameters():
            param.requires_grad = True
        print("[INFO] All DenseNet121 layers unfrozen for end-to-end fine-tuning.")
        return

    # Unfreeze classifier head always
    for param in model.classifier.parameters():
        param.requires_grad = True

    if hasattr(model, "features"):
        if unfreeze_from in ["denseblock4", "block4"]:
            target_blocks = [model.features.denseblock4, model.features.norm5]
        elif unfreeze_from in ["denseblock3", "block3"]:
            target_blocks = [
                model.features.denseblock3,
                model.features.transition3,
                model.features.denseblock4,
                model.features.norm5
            ]
        else:
            target_blocks = [model.features.denseblock4, model.features.norm5]

        for block in target_blocks:
            for param in block.parameters():
                param.requires_grad = True

    print(f"[INFO] DenseNet121 layers unfrozen from '{unfreeze_from}' onwards.")
