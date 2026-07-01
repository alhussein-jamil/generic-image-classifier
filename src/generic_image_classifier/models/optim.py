import torch
import torch.nn as nn
import torch.optim as optim

from generic_image_classifier.logging import get_logger

logger = get_logger(__name__)

OPTIMIZERS: dict[str, type] = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
    "adadelta": optim.Adadelta,
    "adamw": optim.AdamW,
}

LOSSES: dict[str, type[nn.Module]] = {
    "cross_entropy": nn.CrossEntropyLoss,
    "categorical_crossentropy": nn.CrossEntropyLoss,
    "bce": nn.BCEWithLogitsLoss,
    "binary_crossentropy": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
    "mean_squared_error": nn.MSELoss,
}


def build_optimizer(name: str, parameters, learning_rate: float) -> torch.optim.Optimizer:
    key = name.lower()
    if key not in OPTIMIZERS:
        logger.warning("Unknown optimizer %s, using Adam", name)
        return optim.Adam(parameters, lr=learning_rate)
    if key == "sgd":
        return optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    return OPTIMIZERS[key](parameters, lr=learning_rate)


def build_criterion(name: str) -> nn.Module:
    key = name.lower()
    if key not in LOSSES:
        logger.warning("Unknown loss %s, using CrossEntropyLoss", name)
        return nn.CrossEntropyLoss()
    return LOSSES[key]()
