from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingHistory:
    accuracy: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    epochs_trained: int = 0

    def update(self, epoch_history: dict[str, float]) -> None:
        for key in ("accuracy", "val_accuracy", "loss", "val_loss"):
            if key in epoch_history:
                getattr(self, key).append(epoch_history[key])
        self.epochs_trained += 1


@dataclass
class ModelInfo:
    name: str
    num_classes: int
    input_shape: tuple[int, int, int]
    class_mapping: dict[int, str]
    date_trained: str
    params_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "class_mapping": {str(k): v for k, v in self.class_mapping.items()},
            "date_trained": self.date_trained,
            "params_count": self.params_count,
        }

    @classmethod
    def from_dict(cls, info_dict: dict[str, Any]) -> "ModelInfo":
        return cls(
            name=info_dict["name"],
            num_classes=info_dict["num_classes"],
            input_shape=tuple(info_dict["input_shape"]),
            class_mapping={int(k): v for k, v in info_dict["class_mapping"].items()},
            date_trained=info_dict["date_trained"],
            params_count=info_dict.get("params_count", 0),
        )


@dataclass
class ModelResults:
    accuracy: float
    class_accuracies: dict[str, float]
    confusion_matrix: list[list[int]]
    f1_score: float | None = None
    precision: float | None = None
    recall: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "class_accuracies": self.class_accuracies,
            "confusion_matrix": self.confusion_matrix,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
        }

    @classmethod
    def from_dict(cls, results_dict: dict[str, Any]) -> "ModelResults":
        return cls(
            accuracy=results_dict["accuracy"],
            class_accuracies=results_dict["class_accuracies"],
            confusion_matrix=results_dict["confusion_matrix"],
            f1_score=results_dict.get("f1_score"),
            precision=results_dict.get("precision"),
            recall=results_dict.get("recall"),
        )


@dataclass
class Model:
    info: ModelInfo
    history: TrainingHistory = field(default_factory=TrainingHistory)
    results: ModelResults | None = None
    checkpoint_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "info": self.info.to_dict(),
            "history": {
                "accuracy": self.history.accuracy,
                "val_accuracy": self.history.val_accuracy,
                "loss": self.history.loss,
                "val_loss": self.history.val_loss,
                "epochs_trained": self.history.epochs_trained,
            },
            "results": self.results.to_dict() if self.results else None,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
        }

    @classmethod
    def from_dict(cls, model_dict: dict[str, Any]) -> "Model":
        history_data = model_dict["history"]
        history = TrainingHistory(
            accuracy=history_data["accuracy"],
            val_accuracy=history_data["val_accuracy"],
            loss=history_data["loss"],
            val_loss=history_data["val_loss"],
            epochs_trained=history_data["epochs_trained"],
        )
        results = None
        if model_dict.get("results"):
            results = ModelResults.from_dict(model_dict["results"])
        checkpoint_path = None
        if model_dict.get("checkpoint_path"):
            checkpoint_path = Path(model_dict["checkpoint_path"])
        return cls(
            info=ModelInfo.from_dict(model_dict["info"]),
            history=history,
            results=results,
            checkpoint_path=checkpoint_path,
        )
