import os

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def dataloader_workers() -> int:
    cpus = os.cpu_count() or 2
    return min(4, max(0, cpus - 1))
