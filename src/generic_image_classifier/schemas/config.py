from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    zip_path: Path
    extract_dir: Path
    img_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    val_split: float = 0.2
    augmentation: bool = True
    shuffle: bool = True


@dataclass
class ModelConfig:
    model_name: str = "mobilenetv2"
    num_classes: int = 0
    pretrained: bool = True
    learning_rate: float = 0.001
    epochs: int = 10
    optimizer: str = "adam"
    loss: str = "cross_entropy"


@dataclass
class AppConfig:
    title: str = "Image Classification App"
    description: str = "Upload an image to classify it using the trained model"
    examples_dir: Path | None = None
    max_file_size: int = 5
    port: int = 7860


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    app: AppConfig
    checkpoint_dir: Path = Path("checkpoints")
    device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.data).items()},
            "model": vars(self.model),
            "app": {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.app).items()},
            "checkpoint_dir": str(self.checkpoint_dir),
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        data = config_dict["data"]
        model = config_dict["model"]
        app = config_dict["app"]

        return cls(
            data=DataConfig(
                zip_path=Path(data["zip_path"]),
                extract_dir=Path(data["extract_dir"]),
                img_size=tuple(data["img_size"]),
                batch_size=data["batch_size"],
                val_split=data["val_split"],
                augmentation=data["augmentation"],
                shuffle=data["shuffle"],
            ),
            model=ModelConfig(
                model_name=model["model_name"],
                num_classes=model["num_classes"],
                pretrained=model["pretrained"],
                learning_rate=model["learning_rate"],
                epochs=model["epochs"],
                optimizer=model["optimizer"],
                loss=model.get("loss", "cross_entropy"),
            ),
            app=AppConfig(
                title=app["title"],
                description=app["description"],
                examples_dir=Path(app["examples_dir"]) if app.get("examples_dir") else None,
                max_file_size=app["max_file_size"],
                port=app["port"],
            ),
            checkpoint_dir=Path(config_dict.get("checkpoint_dir", "checkpoints")),
            device=config_dict.get("device"),
        )
