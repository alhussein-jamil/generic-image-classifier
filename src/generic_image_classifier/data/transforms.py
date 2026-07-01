from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from generic_image_classifier.device import dataloader_workers
from generic_image_classifier.logging import get_logger
from generic_image_classifier.schemas import DataConfig, ImageInfo
from generic_image_classifier.schemas import Dataset as ImageDatasetSchema

logger = get_logger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: tuple[int, int], *, augment: bool = False) -> transforms.Compose:
    if augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ImageDataset(Dataset):
    def __init__(
        self,
        images: list[ImageInfo],
        transform: transforms.Compose | None = None,
    ):
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_info = self.images[index]
        image = Image.open(img_info.path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(img_info.class_id, dtype=torch.long)
        return image, label


def create_dataloaders(
    dataset: ImageDatasetSchema, config: DataConfig
) -> tuple[DataLoader, DataLoader]:
    num_classes = dataset.metadata.num_classes
    logger.info("Building dataloaders for %d classes", num_classes)

    train_ds = ImageDataset(
        dataset.train_images,
        transform=build_transforms(config.img_size, augment=config.augmentation),
    )
    val_ds = ImageDataset(
        dataset.val_images,
        transform=build_transforms(config.img_size, augment=False),
    )

    workers = dataloader_workers()
    pin = torch.cuda.is_available()
    loader_kwargs: dict = {
        "batch_size": config.batch_size,
        "num_workers": workers,
        "pin_memory": pin,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=config.shuffle, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def load_image_tensor(
    img_path: Path | str,
    img_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    transform = build_transforms(img_size, augment=False)
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor.to(device, non_blocking=device.type == "cuda")
