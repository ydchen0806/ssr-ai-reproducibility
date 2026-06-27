"""Factory for building backbone models."""

import torch.nn as nn
import torchvision.models as tv_models


def build_model(config: dict, num_classes: int) -> nn.Module:
    name = config["name"].lower()

    if name == "resnet18":
        model = tv_models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "resnet32":
        from .resnet_cifar import resnet32
        return resnet32(num_classes=num_classes)

    if name.startswith("vit"):
        import timm
        pretrained = config.get("pretrained", True)
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        return model

    raise ValueError(f"Unknown model: {name}")
