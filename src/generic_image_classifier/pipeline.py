import shutil
from pathlib import Path

from generic_image_classifier.config import save_config
from generic_image_classifier.data import create_dataloaders, load_dataset
from generic_image_classifier.device import resolve_device
from generic_image_classifier.logging import get_logger
from generic_image_classifier.models import create_model, train_model
from generic_image_classifier.schemas import (
    AppConfig,
    Config,
    DataConfig,
    Dataset,
    Model,
    ModelConfig,
)

logger = get_logger(__name__)


def build_training_config(
    *,
    zip_path: Path,
    extract_dir: Path,
    model_name: str,
    img_size: tuple[int, int],
    batch_size: int,
    val_split: float,
    augmentation: bool,
    pretrained: bool,
    learning_rate: float,
    epochs: int,
) -> Config:
    checkpoint_dir = Path("checkpoints") / f"{model_name}_{zip_path.stem}"
    return Config(
        data=DataConfig(
            zip_path=zip_path,
            extract_dir=extract_dir,
            img_size=img_size,
            batch_size=batch_size,
            val_split=val_split,
            augmentation=augmentation,
        ),
        model=ModelConfig(
            model_name=model_name,
            pretrained=pretrained,
            learning_rate=learning_rate,
            epochs=epochs,
        ),
        app=AppConfig(),
        checkpoint_dir=checkpoint_dir,
    )


def run_training(config: Config) -> tuple[str, Model, Dataset]:
    device = resolve_device(config.device)
    logger.info("Device: %s", device)

    dataset = load_dataset(config.data)
    config.model.num_classes = dataset.metadata.num_classes

    train_loader, val_loader = create_dataloaders(dataset, config.data)
    model = create_model(config.model, input_shape=(*config.data.img_size, 3))
    _, model_schema = train_model(
        model,
        train_loader,
        val_loader,
        config.model,
        dataset,
        config.checkpoint_dir,
        device=device,
    )

    _save_class_examples(dataset, config.checkpoint_dir / "examples")
    config.app.examples_dir = config.checkpoint_dir / "examples"
    save_config(config, config.checkpoint_dir / "config.json")

    return config.model.model_name, model_schema, dataset


def _save_class_examples(dataset: Dataset, examples_dir: Path) -> None:
    examples_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for img_info in dataset.val_images:
        if img_info.class_name in seen:
            continue
        dest = examples_dir / f"{img_info.class_name}{img_info.path.suffix}"
        shutil.copy2(img_info.path, dest)
        seen.add(img_info.class_name)
        if len(seen) == dataset.metadata.num_classes:
            break
