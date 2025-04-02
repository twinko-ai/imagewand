import os

import cv2
import numpy as np
import pytest

from imagewand.autofix import autofix, crop_framed_photo


@pytest.fixture
def sample_image(tmp_path):
    # Create a sample image with a white border
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[50:-50, 50:-50] = 128  # Gray content
    img[:50, :] = 255  # White border top
    img[-50:, :] = 255  # White border bottom
    img[:, :50] = 255  # White border left
    img[:, -50:] = 255  # White border right
    
    path = tmp_path / "test.jpg"
    cv2.imwrite(str(path), img)
    return str(path)


def test_crop_framed_photo(tmp_path, sample_image):
    # Save sample image
    input_path = str(tmp_path / "test_frame.jpg")
    cv2.imwrite(input_path, sample_image)

    # Test frame mode
    output_path = str(tmp_path / "output_frame.jpg")
    result = crop_framed_photo(input_path, output_path, margin=-2)

    # Check if output file exists
    assert os.path.exists(result)

    # Load and check result
    result_img = cv2.imread(result)
    assert result_img.shape[0] < sample_image.shape[0]  # Should be smaller
    assert result_img.shape[1] < sample_image.shape[1]


def test_autofix_border_mode(sample_image, tmp_path):
    result = autofix(sample_image, mode="border")
    assert os.path.exists(result)
    result_img = cv2.imread(result)
    assert result_img.shape[0] < 300  # Should be smaller
    assert result_img.shape[1] < 400


def test_autofix_frame_mode(sample_image, tmp_path):
    result = autofix(sample_image, mode="frame")
    assert os.path.exists(result)
    result_img = cv2.imread(result)
    assert result_img.shape[0] < 300
    assert result_img.shape[1] < 400


def test_autofix_with_margin(sample_image, tmp_path):
    result = autofix(sample_image, mode="frame", margin=10)
    assert os.path.exists(result)
    assert "_m10" in result


def test_autofix_with_border(sample_image, tmp_path):
    result = autofix(sample_image, mode="border", border_percent=5)
    assert os.path.exists(result)
    assert "_b5" in result


def test_autofix_invalid_mode(sample_image):
    with pytest.raises(ValueError):
        autofix(sample_image, mode="invalid")
