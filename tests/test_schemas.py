from pathlib import Path

from generic_image_classifier.schemas import (
    ClassInfo,
    Dataset,
    DatasetMetadata,
    ImageInfo,
    Model,
    ModelInfo,
    ModelResults,
    TrainingHistory,
)


def test_dataset_roundtrip():
    dataset = Dataset(
        train_images=[
            ImageInfo(path=Path("a.jpg"), class_name="cat", class_id=0),
        ],
        val_images=[
            ImageInfo(path=Path("b.jpg"), class_name="dog", class_id=1),
        ],
        classes=[
            ClassInfo(name="cat", id=0, sample_count=1),
            ClassInfo(name="dog", id=1, sample_count=1),
        ],
        metadata=DatasetMetadata(
            name="test",
            num_classes=2,
            total_samples=2,
            class_distribution={"cat": 1, "dog": 1},
            class_mapping={"cat": 0, "dog": 1},
            inv_class_mapping={0: "cat", 1: "dog"},
        ),
    )
    restored = Dataset.from_dict(dataset.to_dict())
    assert restored.metadata.num_classes == 2
    assert restored.train_images[0].class_name == "cat"


def test_model_roundtrip():
    model = Model(
        info=ModelInfo(
            name="mobilenetv2",
            num_classes=2,
            input_shape=(224, 224, 3),
            class_mapping={0: "cat", 1: "dog"},
            date_trained="20260101-120000",
            params_count=1000,
        ),
        history=TrainingHistory(
            accuracy=[0.5, 0.8],
            val_accuracy=[0.4, 0.7],
            loss=[1.0, 0.5],
            val_loss=[1.1, 0.6],
            epochs_trained=2,
        ),
        results=ModelResults(
            accuracy=0.7,
            class_accuracies={"cat": 0.8, "dog": 0.6},
            confusion_matrix=[[5, 1], [2, 4]],
            f1_score=0.7,
            precision=0.72,
            recall=0.68,
        ),
        checkpoint_path=Path("checkpoints/mobilenetv2_best.pt"),
    )
    restored = Model.from_dict(model.to_dict())
    assert restored.info.name == "mobilenetv2"
    assert restored.results.accuracy == 0.7
    assert restored.checkpoint_path.name == "mobilenetv2_best.pt"


def test_training_history_update():
    history = TrainingHistory()
    history.update({"accuracy": 0.9, "loss": 0.1, "val_accuracy": 0.8, "val_loss": 0.2})
    assert history.epochs_trained == 1
    assert history.accuracy == [0.9]
