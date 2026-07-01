import random
import shutil
import zipfile
from pathlib import Path

from generic_image_classifier.logging import get_logger
from generic_image_classifier.schemas import (
    ClassInfo,
    DataConfig,
    Dataset,
    DatasetMetadata,
    ImageInfo,
)

logger = get_logger(__name__)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})


def is_valid_image(filepath: Path) -> bool:
    return filepath.suffix.lower() in IMAGE_EXTENSIONS


def extract_zip_dataset(config: DataConfig) -> Path:
    zip_path = config.zip_path
    extract_dir = config.extract_dir

    if extract_dir.exists():
        for item in extract_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting %s -> %s", zip_path, extract_dir)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    contents = list(extract_dir.iterdir())
    dataset_dir = contents[0] if len(contents) == 1 and contents[0].is_dir() else extract_dir
    logger.info("Dataset root: %s", dataset_dir)
    return dataset_dir


def load_dataset(config: DataConfig) -> Dataset:
    dataset_dir = extract_zip_dataset(config)
    class_dirs = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found in {dataset_dir}")

    class_mapping = {d.name: idx for idx, d in enumerate(class_dirs)}
    inv_class_mapping = {idx: name for name, idx in class_mapping.items()}

    class_counts: dict[str, int] = {}
    all_images: list[ImageInfo] = []

    for class_dir in class_dirs:
        class_name = class_dir.name
        class_id = class_mapping[class_name]
        image_files = [f for f in class_dir.rglob("*") if f.is_file() and is_valid_image(f)]
        class_counts[class_name] = len(image_files)
        all_images.extend(
            ImageInfo(path=path, class_name=class_name, class_id=class_id) for path in image_files
        )

    if config.shuffle:
        random.shuffle(all_images)

    split_idx = int(len(all_images) * (1 - config.val_split))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    logger.info(
        "Split: %d train, %d val (%d classes)",
        len(train_images),
        len(val_images),
        len(class_dirs),
    )

    classes = [
        ClassInfo(name=name, id=class_mapping[name], sample_count=class_counts[name])
        for name in sorted(class_mapping)
    ]
    metadata = DatasetMetadata(
        name=dataset_dir.name,
        num_classes=len(class_dirs),
        total_samples=len(all_images),
        class_distribution=class_counts,
        class_mapping=class_mapping,
        inv_class_mapping=inv_class_mapping,
    )

    return Dataset(
        train_images=train_images,
        val_images=val_images,
        classes=classes,
        metadata=metadata,
    )
