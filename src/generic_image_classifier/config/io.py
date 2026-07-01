import json
from pathlib import Path
from typing import Any

from generic_image_classifier.logging import get_logger
from generic_image_classifier.schemas import AppConfig, Config, DataConfig, ModelConfig

logger = get_logger(__name__)


def create_default_config(zip_path: Path) -> Config:
    zip_name = zip_path.stem
    return Config(
        data=DataConfig(
            zip_path=zip_path,
            extract_dir=Path("data") / zip_name,
        ),
        model=ModelConfig(),
        app=AppConfig(),
    )


def create_interactive_config(port: int = 7860) -> Config:
    return Config(
        data=DataConfig(
            zip_path=Path(),
            extract_dir=Path("data"),
        ),
        model=ModelConfig(),
        app=AppConfig(
            title="All-in-One Image Classification",
            description="Upload a dataset, train a model, and classify images.",
            port=port,
        ),
    )


def save_config(config: Config, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info("Saved config to %s", config_path)


def load_config(config_path: Path) -> Config | None:
    if not config_path.exists():
        logger.warning("Config not found: %s", config_path)
        return None
    try:
        with config_path.open() as f:
            config = Config.from_dict(json.load(f))
        logger.info("Loaded config from %s", config_path)
        return config
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error("Failed to load config: %s", exc)
        return None


def update_config(config: Config, updates: dict[str, Any]) -> Config:
    config_dict = config.to_dict()
    for key_path, value in updates.items():
        parts = key_path.split(".")
        if len(parts) != 2:
            logger.warning("Skipping invalid key: %s", key_path)
            continue
        section, key = parts
        if section in config_dict and key in config_dict[section]:
            config_dict[section][key] = value
        else:
            logger.warning("Unknown config key: %s", key_path)
    return Config.from_dict(config_dict)
