from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImageInfo:
    path: Path
    class_name: str
    class_id: int


@dataclass
class ClassInfo:
    name: str
    id: int
    sample_count: int = 0


@dataclass
class DatasetMetadata:
    name: str
    num_classes: int
    total_samples: int
    class_distribution: dict[str, int]
    class_mapping: dict[str, int]
    inv_class_mapping: dict[int, str]


@dataclass
class Dataset:
    train_images: list[ImageInfo] = field(default_factory=list)
    val_images: list[ImageInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    metadata: DatasetMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_images": [
                {
                    "path": str(img.path),
                    "class_name": img.class_name,
                    "class_id": img.class_id,
                }
                for img in self.train_images
            ],
            "val_images": [
                {
                    "path": str(img.path),
                    "class_name": img.class_name,
                    "class_id": img.class_id,
                }
                for img in self.val_images
            ],
            "classes": [
                {"name": cls.name, "id": cls.id, "sample_count": cls.sample_count}
                for cls in self.classes
            ],
            "metadata": {
                "name": self.metadata.name,
                "num_classes": self.metadata.num_classes,
                "total_samples": self.metadata.total_samples,
                "class_distribution": self.metadata.class_distribution,
                "class_mapping": self.metadata.class_mapping,
                "inv_class_mapping": {
                    str(k): v for k, v in self.metadata.inv_class_mapping.items()
                },
            }
            if self.metadata
            else None,
        }

    @classmethod
    def from_dict(cls, dataset_dict: dict[str, Any]) -> "Dataset":
        train_images = [
            ImageInfo(
                path=Path(img["path"]),
                class_name=img["class_name"],
                class_id=img["class_id"],
            )
            for img in dataset_dict.get("train_images", [])
        ]
        val_images = [
            ImageInfo(
                path=Path(img["path"]),
                class_name=img["class_name"],
                class_id=img["class_id"],
            )
            for img in dataset_dict.get("val_images", [])
        ]
        classes = [
            ClassInfo(
                name=cls["name"],
                id=cls["id"],
                sample_count=cls["sample_count"],
            )
            for cls in dataset_dict.get("classes", [])
        ]

        metadata = None
        if dataset_dict.get("metadata"):
            meta = dataset_dict["metadata"]
            metadata = DatasetMetadata(
                name=meta["name"],
                num_classes=meta["num_classes"],
                total_samples=meta["total_samples"],
                class_distribution=meta["class_distribution"],
                class_mapping=meta["class_mapping"],
                inv_class_mapping={int(k): v for k, v in meta["inv_class_mapping"].items()},
            )

        return cls(
            train_images=train_images,
            val_images=val_images,
            classes=classes,
            metadata=metadata,
        )
