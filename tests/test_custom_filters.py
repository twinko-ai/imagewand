import cv2
import numpy as np
import pytest

from imagewand.plugins.custom_filters import my_custom_filter


@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    return np.ones((100, 100, 3), dtype=np.uint8) * 128


def test_custom_filter_no_params(sample_image):
    result = my_custom_filter(sample_image)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape


def test_custom_filter_with_params(sample_image):
    params = {"intensity": 1.5}
    result = my_custom_filter(sample_image, params)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape


def test_custom_filter_invalid_params(sample_image):
    params = {"invalid_param": 1.0}
    result = my_custom_filter(sample_image, params)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape
