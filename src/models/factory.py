"""
Unified Model Factory Interface for OncoVision.

Provides a single entry point to instantiate ResNet50, EfficientNet-B0, or DenseNet121
with identical target class dimensions, transfer learning settings, and head configuration.
"""

from src.config import NUM_CLASSES
from src.models.resnet import build_resnet50, unfreeze_backbone_layers
from src.models.efficientnet import build_efficientnet_b0, unfreeze_efficientnet_layers
from src.models.densenet import build_densenet121, unfreeze_densenet_layers

SUPPORTED_MODELS = {
    "resnet50": build_resnet50,
    "resnet": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
    "efficientnet": build_efficientnet_b0,
    "densenet121": build_densenet121,
    "densenet": build_densenet121
}


def build_model(
    model_name="resnet50",
    num_classes=NUM_CLASSES,
    pretrained=True,
    freeze_backbone=True,
    dropout_rate=0.3
):
    """
    Factory function to instantiate a model architecture by string name.

    Args:
        model_name (str): Architecture name ("resnet50", "efficientnet_b0", or "densenet121").
        num_classes (int): Output class count (default: 3).
        pretrained (bool): Whether to load ImageNet pretrained weights.
        freeze_backbone (bool): Whether to freeze backbone layers initially.
        dropout_rate (float): Dropout probability in final head.

    Returns:
        nn.Module: Configured PyTorch model instance.
    """
    key = model_name.lower().strip()
    if key not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model name '{model_name}'. "
            f"Supported architectures: {list(SUPPORTED_MODELS.keys())}"
        )

    builder_fn = SUPPORTED_MODELS[key]
    return builder_fn(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate
    )


def count_parameters(model):
    """
    Returns (total_params, trainable_params) tuple for a given PyTorch model.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
