import zipfile
from pathlib import Path


def make_zip(path: Path, classes: dict[str, list[str]]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for class_name, files in classes.items():
            for filename in files:
                zf.writestr(f"{class_name}/{filename}", b"fake")
