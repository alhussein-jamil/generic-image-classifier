import zipfile
from pathlib import Path

from generic_image_classifier.data.loading import load_dataset
from generic_image_classifier.schemas import DataConfig


def _make_zip(path: Path, classes: dict[str, list[str]]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for class_name, files in classes.items():
            for filename in files:
                zf.writestr(f"{class_name}/{filename}", b"fake")


def test_load_dataset_split(tmp_path: Path):
    zip_path = tmp_path / "dataset.zip"
    _make_zip(
        zip_path,
        {
            "cats": ["a.jpg", "b.jpg", "c.jpg"],
            "dogs": ["d.jpg", "e.jpg", "f.jpg"],
        },
    )
    extract_dir = tmp_path / "extracted"
    config = DataConfig(
        zip_path=zip_path,
        extract_dir=extract_dir,
        val_split=0.33,
        shuffle=False,
    )
    dataset = load_dataset(config)
    assert dataset.metadata.num_classes == 2
    assert dataset.metadata.total_samples == 6
    assert len(dataset.train_images) == 4
    assert len(dataset.val_images) == 2
