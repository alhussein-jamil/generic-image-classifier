from pathlib import Path

from generic_image_classifier.config import (
    create_default_config,
    load_config,
    save_config,
    update_config,
)
from generic_image_classifier.schemas import AppConfig, Config, DataConfig, ModelConfig


def test_default_config_roundtrip(tmp_path: Path):
    zip_path = tmp_path / "data.zip"
    zip_path.touch()
    config = create_default_config(zip_path)
    config_path = tmp_path / "config.json"
    save_config(config, config_path)
    loaded = load_config(config_path)
    assert loaded is not None
    assert loaded.data.zip_path == zip_path
    assert loaded.model.model_name == "mobilenetv2"


def test_update_config(tmp_path: Path):
    zip_path = tmp_path / "data.zip"
    zip_path.touch()
    config = create_default_config(zip_path)
    updated = update_config(config, {"model.epochs": 25, "data.batch_size": 16})
    assert updated.model.epochs == 25
    assert updated.data.batch_size == 16


def test_config_to_dict_from_dict():
    config = Config(
        data=DataConfig(zip_path=Path("a.zip"), extract_dir=Path("data/a")),
        model=ModelConfig(num_classes=3),
        app=AppConfig(),
    )
    restored = Config.from_dict(config.to_dict())
    assert restored.model.num_classes == 3
    assert restored.data.zip_path == Path("a.zip")
