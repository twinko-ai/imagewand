import os

import cv2
import numpy as np
import pytest

from imagewand.autofix import autofix, crop_framed_photo


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample image for testing"""
    img = np.ones((300, 400, 3), dtype=np.uint8) * 128
    path = str(tmp_path / "test.jpg")
    cv2.imwrite(str(path), img)
    return str(path)


def test_crop_framed_photo(tmp_path, sample_image):
    # Create a test image with a frame
    img = cv2.imread(sample_image)
    output_path = str(tmp_path / "test_frame.jpg")
    cv2.imwrite(output_path, img)
    result = crop_framed_photo(output_path)
    assert result is not None
    assert os.path.exists(result)


def test_autofix_border_mode(sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    result = autofix(sample_image, output_path=output, mode="border")
    assert os.path.exists(result)


def test_autofix_frame_mode(sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    result = autofix(sample_image, output_path=output, mode="frame")
    assert os.path.exists(result)


def test_autofix_with_margin(sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    result = autofix(sample_image, output_path=output, mode="frame", margin=10)
    assert os.path.exists(result)


def test_autofix_with_border(sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    result = autofix(sample_image, output_path=output, mode="border", border_percent=5)
    assert os.path.exists(result)


def test_autofix_invalid_mode(sample_image):
    with pytest.raises(ValueError):
        autofix(sample_image, mode="invalid")
