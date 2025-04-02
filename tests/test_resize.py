import os

import cv2
import numpy as np
import pytest

from imagewand.resize import resize_image


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample image for testing"""
    img = np.ones((300, 400, 3), dtype=np.uint8) * 128
    path = tmp_path / "test.jpg"
    cv2.imwrite(str(path), img)
    return str(path)


def test_resize_by_width(sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    result = resize_image(sample_image, output, width=200)
    
    assert os.path.exists(result)
    img = cv2.imread(result)
    assert img.shape[1] == 200  # width should be 200
    assert img.shape[0] == 150  # height should maintain aspect ratio


def test_resize_by_height(sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    result = resize_image(sample_image, output, height=150)
    
    assert os.path.exists(result)
    img = cv2.imread(result)
    assert img.shape[0] == 150  # height should be 150
    assert img.shape[1] == 200  # width should maintain aspect ratio


def test_resize_both_dimensions(sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    result = resize_image(sample_image, output, width=200, height=150)
    
    assert os.path.exists(result)
    img = cv2.imread(result)
    assert img.shape[1] == 200
    assert img.shape[0] == 150


def test_resize_invalid_input():
    with pytest.raises(ValueError):
        resize_image("nonexistent.jpg", "output.jpg", width=200)


def test_resize_no_dimensions(sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    with pytest.raises(ValueError):
        resize_image(sample_image, output) 