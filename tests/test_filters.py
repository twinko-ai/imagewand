import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from imagewand.filters import (
    FILTERS,
    apply_filter,
    apply_filters,
    batch_apply_filters,
    list_filters,
)


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample image file for testing"""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    path = tmp_path / "test.jpg"
    cv2.imwrite(str(path), img)
    return str(path)


def test_single_filter(sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    result = apply_filters([sample_image], ["contrast"], output)
    assert os.path.exists(result)
    assert cv2.imread(result) is not None


def test_multiple_filters(sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    result = apply_filters([sample_image], ["contrast", "sharpen"], output)
    assert os.path.exists(result)
    assert cv2.imread(result) is not None


def test_filter_with_params(sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    result = apply_filters([sample_image], ["contrast:factor=1.5"], output)
    assert os.path.exists(result)
    assert cv2.imread(result) is not None


def test_batch_processing(tmp_path):
    # Create multiple test images
    image_paths = []
    for i in range(3):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        path = tmp_path / f"test_{i}.jpg"
        cv2.imwrite(str(path), img)
        image_paths.append(str(path))

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    results = batch_apply_filters(
        image_paths,
        ["contrast", "sharpen"],
        str(output_dir),
    )
    assert len(results) == len(image_paths)
    for path in results:
        assert os.path.exists(path)


def test_list_filters():
    filters = list_filters()
    assert isinstance(filters, list)
    assert len(filters) > 0
    assert "contrast" in filters
    assert "sharpen" in filters


def test_invalid_filter(sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    with pytest.raises(ValueError, match="Filter 'invalid' not found"):
        apply_filters([sample_image], ["invalid"], output)


@pytest.mark.parametrize(
    "filter_name",
    [
        "grayscale",
        "sepia",
        "blur",
        "sharpen",
        "brightness",
        "contrast",
        "saturation",
        "edge_enhance",
        "emboss",
    ],
)
def test_all_filters(filter_name, sample_image, tmp_path):
    output = str(tmp_path / "output.jpg")
    result = apply_filters([sample_image], [filter_name], output)
    assert os.path.exists(result)
    assert cv2.imread(result) is not None
