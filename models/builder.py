"""Factory for building backbone models."""

import torch.nn as nn
import torchvision.models as tv_models
from pathlib import Path


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
        checkpoint = config.get("pretrained_checkpoint")
        if checkpoint:
            checkpoint_path = Path(checkpoint).expanduser()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Pretrained ViT checkpoint not found: {checkpoint_path}. "
                    "Stage it on the offline cluster before launching."
                )
            model = timm.create_model(name, pretrained=False)
            if checkpoint_path.suffix == ".safetensors":
                from safetensors.torch import load_file

                state_dict = load_file(str(checkpoint_path), device="cpu")
            else:
                import torch

                payload = torch.load(checkpoint_path, map_location="cpu")
                state_dict = payload.get("state_dict", payload.get("model", payload))
            state_dict = {
                key.removeprefix("module."): value for key, value in state_dict.items()
            }
            model.load_state_dict(state_dict, strict=True)
            model.reset_classifier(num_classes)
        else:
            model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        return model

    raise ValueError(f"Unknown model: {name}")
