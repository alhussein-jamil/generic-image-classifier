import zipfile
from pathlib import Path

from generic_image_classifier.schemas import Model, ModelInfo, ModelResults, TrainingHistory
from generic_image_classifier.ui.components import (
    metrics_markdown,
    model_blurb,
    preview_zip_contents,
)


def _make_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("cats/a.jpg", b"x")
        zf.writestr("cats/b.jpg", b"x")
        zf.writestr("dogs/c.jpg", b"x")


def test_preview_zip_contents(tmp_path: Path):
    zip_path = tmp_path / "data.zip"
    _make_zip(zip_path)

    class FileStub:
        name = str(zip_path)

    banner, table, ready = preview_zip_contents(FileStub())
    assert ready is True
    assert "cats" in table
    assert "dogs" in table
    assert "gic-banner-ok" in banner


def test_model_blurb_known():
    text = model_blurb("mobilenetv2")
    assert "MobileNetV2" in text


def test_metrics_markdown():
    schema = Model(
        info=ModelInfo("mobilenetv2", 2, (224, 224, 3), {0: "a", 1: "b"}, "20260101"),
        history=TrainingHistory(
            accuracy=[0.9],
            val_accuracy=[0.8],
            loss=[0.2],
            val_loss=[0.3],
            epochs_trained=1,
        ),
        results=ModelResults(
            accuracy=0.8,
            class_accuracies={"a": 0.9, "b": 0.7},
            confusion_matrix=[[4, 1], [1, 3]],
            f1_score=0.75,
            precision=0.8,
            recall=0.7,
        ),
    )
    html = metrics_markdown(schema)
    assert "gic-metric-grid" in html
    assert "80.0%" in html or "80%" in html
