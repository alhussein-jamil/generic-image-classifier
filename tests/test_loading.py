from pathlib import Path

import pytest

from generic_image_classifier.data.loading import is_valid_image


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
