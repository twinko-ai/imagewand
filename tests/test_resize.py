import os

import cv2
import numpy as np
import pytest

from imagewand.resize import resize_image, parse_file_size, resize_to_target_size


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample image for testing"""
    img = np.ones((300, 400, 3), dtype=np.uint8) * 128
    path = tmp_path / "test.jpg"
    cv2.imwrite(str(path), img)
    return str(path)


def test_parse_file_size():
    """Test file size parsing function"""
    assert parse_file_size("5MB") == 5 * 1024 * 1024
    assert parse_file_size("500KB") == 500 * 1024
    assert parse_file_size("2.5GB") == int(2.5 * 1024 * 1024 * 1024)
    assert parse_file_size("1024B") == 1024
    assert parse_file_size("1024") == 1024
    assert parse_file_size("5M") == 5 * 1024 * 1024
    assert parse_file_size("500K") == 500 * 1024

    with pytest.raises(ValueError):
        parse_file_size("invalid")

    with pytest.raises(ValueError):
        parse_file_size("5XB")


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


def test_resize_by_percent(sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    result = resize_image(sample_image, output, percent=50)
    assert os.path.exists(result)
    img = cv2.imread(result)
    assert img.shape[1] == 200  # 50% of 400
    assert img.shape[0] == 150  # 50% of 300


def test_resize_to_target_size(sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    result = resize_to_target_size(sample_image, "50KB", output)
    assert os.path.exists(result)

    # Check that file size is reasonably close to target
    file_size = os.path.getsize(result)
    target_size = 50 * 1024
    # Allow some tolerance (within 20% of target)
    assert abs(file_size - target_size) / target_size < 0.2


def test_resize_target_size_via_main_function(sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    result = resize_image(sample_image, output, target_size="30KB")
    assert os.path.exists(result)

    # Check that file size is reasonably close to target
    file_size = os.path.getsize(result)
    target_size = 30 * 1024
    # Allow some tolerance (within 20% of target)
    assert abs(file_size - target_size) / target_size < 0.2


def test_resize_invalid_input():
    with pytest.raises((ValueError, FileNotFoundError)):
        resize_image("nonexistent.jpg", "output.jpg", width=200)


def test_resize_no_dimensions(sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    with pytest.raises(
        ValueError,
        match="Either width, height, percent, or target_size must be specified",
    ):
        resize_image(sample_image, output)
