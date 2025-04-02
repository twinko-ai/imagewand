import os

import cv2
import numpy as np
import pytest

from imagewand.filters import apply_filter


@pytest.fixture
def sample_image():
    # Create a sample image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    return img


def test_contrast_filter(tmp_path, sample_image):
    input_path = str(tmp_path / "test.jpg")
    cv2.imwrite(input_path, sample_image)

    output_path = str(tmp_path / "output.jpg")
    result = apply_filter(input_path, ["contrast"], output_path)

    assert os.path.exists(result)
    result_img = cv2.imread(result)
    assert not np.all(result_img == sample_image)


def test_multiple_filters(tmp_path, sample_image):
    input_path = str(tmp_path / "test.jpg")
    cv2.imwrite(input_path, sample_image)

    output_path = str(tmp_path / "output.jpg")
    result = apply_filter(input_path, ["contrast", "sharpen"], output_path)

    assert os.path.exists(result)
    result_img = cv2.imread(result)
    assert not np.all(result_img == sample_image)
