import tempfile
import zipfile
from pathlib import Path

from generic_image_classifier.logging import get_logger

logger = get_logger(__name__)


class ZipUploadError(Exception):
    """Raised when an uploaded ZIP cannot be used as a dataset."""


def count_classes_in_zip(zip_path: Path | str) -> dict[str, int]:
    counts: dict[str, int] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            parts = name.split("/")
            if len(parts) < 2:
                continue
            top = parts[0]
            if top and not top.startswith("."):
                counts[top] = counts.get(top, 0) + 1
    return counts


def validate_class_zip(zip_path: Path | str) -> str | None:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            if not names:
                return "ZIP file is empty."
            if not any("/" in name for name in names):
                return "ZIP must contain class folders, not loose files."
    except zipfile.BadZipFile:
        return "Invalid ZIP file."
    except OSError as exc:
        logger.error("ZIP validation failed: %s", exc)
        return f"Could not read ZIP: {exc}"

    if not count_classes_in_zip(zip_path):
        return "No images found inside class folders."
    return None


def extract_uploaded_dataset(zip_file) -> tuple[Path, Path]:
    if zip_file is None:
        raise ZipUploadError("No zip file provided")

    zip_path = Path(zip_file.name)
    error = validate_class_zip(zip_path)
    if error:
        raise ZipUploadError(error)

    temp_dir = Path(tempfile.mkdtemp())
    extract_dir = temp_dir / "dataset"
    extract_dir.mkdir()

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    if not any(d.is_dir() for d in extract_dir.iterdir()):
        raise ZipUploadError("No class folders found after extraction")

    return zip_path, extract_dir
