from collections.abc import Callable

import torch
import torch.nn as nn
from torchvision import models

from generic_image_classifier.logging import get_logger
from generic_image_classifier.schemas import ModelConfig

logger = get_logger(__name__)

BACKBONES: dict[str, Callable] = {
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "vgg16": models.vgg16,
    "vgg19": models.vgg19,
    "mobilenet": models.mobilenet_v2,
    "mobilenetv2": models.mobilenet_v2,
    "densenet121": models.densenet121,
    "densenet169": models.densenet169,
    "densenet201": models.densenet201,
    "efficientnetb0": models.efficientnet_b0,
    "efficientnetb1": models.efficientnet_b1,
    "efficientnetb2": models.efficientnet_b2,
    "efficientnetb3": models.efficientnet_b3,
    "efficientnetb4": models.efficientnet_b4,
    "efficientnetb5": models.efficientnet_b5,
    "efficientnetb6": models.efficientnet_b6,
    "efficientnetb7": models.efficientnet_b7,
}


def create_model(config: ModelConfig, input_shape: tuple[int, int, int]) -> nn.Module:
    logger.info(
        "Building %s (%d classes, pretrained=%s)",
        config.model_name,
        config.num_classes,
        config.pretrained,
    )
    base = _load_backbone(config.model_name, config.pretrained, input_shape)
    model = _replace_classifier(base, config.model_name, config.num_classes)
    params = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %s", f"{params:,}")
    return model


def _load_backbone(
    model_name: str, pretrained: bool, input_shape: tuple[int, int, int]
) -> nn.Module:
    key = model_name.lower()
    if key not in BACKBONES:
        logger.warning("Unknown backbone %s, using custom CNN", model_name)
        return _CustomCNN(input_shape)

    factory = BACKBONES[key]
    if pretrained:
        return factory(weights="IMAGENET1K_V1")
    return factory()


def _replace_classifier(model: nn.Module, model_name: str, num_classes: int) -> nn.Module:
    if isinstance(model, _CustomCNN):
        return _CustomHead(model, model.fc_input_size, num_classes)

    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential):
            in_features = classifier[-1].in_features
            classifier[-1] = nn.Linear(in_features, num_classes)
        elif hasattr(classifier, "fc"):
            in_features = classifier.fc.in_features
            classifier.fc = nn.Linear(in_features, num_classes)
        else:
            in_features = classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        return model

    logger.warning("Unrecognized head for %s, wrapping with linear layer", model_name)
    return nn.Sequential(model, nn.Linear(1000, num_classes))


class _CustomCNN(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int]):
        super().__init__()
        c = input_shape[2]
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        h, w = input_shape[0] // 16, input_shape[1] // 16
        self.fc_input_size = 128 * h * w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


class _CustomHead(nn.Module):
    def __init__(self, backbone: _CustomCNN, fc_input_size: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(fc_input_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
