import zipfile
from pathlib import Path

import pytest

from generic_image_classifier.data.zip import (
    ZipUploadError,
    count_classes_in_zip,
    extract_uploaded_dataset,
    validate_class_zip,
)


def _write_zip(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


def test_count_classes_in_zip(tmp_path: Path):
    path = tmp_path / "data.zip"
    _write_zip(
        path,
        {
            "cats/a.jpg": b"x",
            "cats/b.jpg": b"x",
            "dogs/c.jpg": b"x",
        },
    )
    assert count_classes_in_zip(path) == {"cats": 2, "dogs": 1}


def test_validate_class_zip_rejects_loose_files(tmp_path: Path):
    path = tmp_path / "bad.zip"
    _write_zip(path, {"image.jpg": b"x"})
    assert validate_class_zip(path) is not None


def test_extract_uploaded_dataset(tmp_path: Path):
    path = tmp_path / "data.zip"
    _write_zip(path, {"birds/a.jpg": b"x"})

    class Upload:
        name = str(path)

    zip_path, extract_dir = extract_uploaded_dataset(Upload())
    assert zip_path == path
    assert (extract_dir / "birds").is_dir()


def test_extract_uploaded_dataset_missing_file():
    with pytest.raises(ZipUploadError):
        extract_uploaded_dataset(None)
