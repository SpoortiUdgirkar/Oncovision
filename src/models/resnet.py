"""
ResNet50 Transfer Learning Architecture for Breast Ultrasound Classification.

This module provides:
1. build_resnet50: Factory function to instantiate ResNet50 with custom classification head,
   pretrained ImageNet weights, and configurable backbone freezing.
2. unfreeze_backbone_layers: Utility to unfreeze specific deeper layers for fine-tuning.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from src.config import NUM_CLASSES


def build_resnet50(
    num_classes=NUM_CLASSES,
    pretrained=True,
    freeze_backbone=True,
    dropout_rate=0.3
):
    """
    Constructs a ResNet50 model configured for transfer learning.

    Args:
        num_classes (int): Number of target output classes (3: Normal, Benign, Malignant).
        pretrained (bool): Whether to load ImageNet pretrained weights.
        freeze_backbone (bool): If True, freezes backbone layers for feature extraction training.
        dropout_rate (float): Dropout probability applied before final linear layer.

    Returns:
        nn.Module: Configured ResNet50 model.
    """
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)

    # Freeze backbone parameters initially if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace original 1000-class fc head with custom classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )

    # Ensure classification head parameters are trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_backbone_layers(model, unfreeze_from="layer4"):
    """
    Unfreezes selected deeper blocks of ResNet50 for fine-tuning.

    Args:
        model (nn.Module): ResNet50 model instance.
        unfreeze_from (str): Layer name from which to unfreeze ("layer4", "layer3", or "all").
    """
    if unfreeze_from == "all":
        for param in model.parameters():
            param.requires_grad = True
        print("[INFO] All ResNet50 layers unfrozen for end-to-end fine-tuning.")
        return

    # Map layer names to ResNet50 children modules
    target_layers = []
    if unfreeze_from == "layer4":
        target_layers = [model.layer4, model.fc]
    elif unfreeze_from == "layer3":
        target_layers = [model.layer3, model.layer4, model.fc]
    else:
        raise ValueError(f"Unsupported unfreeze layer string: '{unfreeze_from}'. Choose 'layer4', 'layer3', or 'all'.")

    for layer in target_layers:
        for param in layer.parameters():
            param.requires_grad = True

    print(f"[INFO] ResNet50 layers unfrozen from '{unfreeze_from}' onwards.")
