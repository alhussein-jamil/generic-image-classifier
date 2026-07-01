from pathlib import Path

import pytest

from factories import make_zip
from generic_image_classifier.data.loading import is_valid_image, load_dataset
from generic_image_classifier.schemas import DataConfig


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("photo.jpg", True),
        ("photo.JPEG", True),
        ("photo.png", True),
        ("photo.txt", False),
        ("photo", False),
    ],
)
def test_is_valid_image(name: str, expected: bool):
    assert is_valid_image(Path(name)) is expected


def test_load_dataset_split(tmp_path: Path):
    zip_path = tmp_path / "dataset.zip"
    make_zip(
        zip_path,
        {
            "cats": ["a.jpg", "b.jpg", "c.jpg"],
            "dogs": ["d.jpg", "e.jpg", "f.jpg"],
        },
    )
    config = DataConfig(
        zip_path=zip_path,
        extract_dir=tmp_path / "extracted",
        val_split=0.33,
        shuffle=False,
    )
    dataset = load_dataset(config)
    assert dataset.metadata.num_classes == 2
    assert dataset.metadata.total_samples == 6
    assert len(dataset.train_images) == 4
    assert len(dataset.val_images) == 2
