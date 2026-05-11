"""Backbone configuration and constructors for image datasets."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch.nn as nn
    from torchvision import models
except ImportError:  # pragma: no cover - optional image dependency
    nn = None
    models = None


@dataclass(frozen=True)
class BackboneConfig:
    backbone: str
    image_size: int = 224
    pretrained: bool = True
    feature_batch_size: int = 32
    num_workers: int = 4


DATASET_BACKBONES = {
    "bach": BackboneConfig(backbone="resnet18"),
}


def get_backbone_config(dataset: str) -> BackboneConfig:
    try:
        return DATASET_BACKBONES[dataset]
    except KeyError as exc:
        raise ValueError(f"No backbone config registered for dataset '{dataset}'.") from exc


def create_backbone(backbone_name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Load backbone and remove classifier head."""
    if models is None or nn is None:
        raise RuntimeError("torchvision is required for image backbone features.")

    if backbone_name == "resnet18":
        print("Loading pretrained ResNet-18 backbone." if pretrained else "Loading ResNet-18 backbone.")
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif backbone_name == "resnet34":
        print("Loading pretrained ResNet-34 backbone." if pretrained else "Loading ResNet-34 backbone.")
        backbone = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif backbone_name == "resnet50":
        print("Loading pretrained ResNet-50 backbone." if pretrained else "Loading ResNet-50 backbone.")
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif backbone_name == "efficientnet_b0":
        print(
            "Loading pretrained EfficientNet-B0 backbone."
            if pretrained
            else "Loading EfficientNet-B0 backbone."
        )
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_ftrs = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
    else:
        raise ValueError(f"Backbone '{backbone_name}' is not supported.")

    return backbone, num_ftrs
