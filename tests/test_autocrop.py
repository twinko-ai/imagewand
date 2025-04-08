import os
import pytest
import cv2
import numpy as np
from pathlib import Path

from imagewand.autocrop import (
    autocrop,
    crop_framed_photo,
    crop_with_content_detection,
    crop_dark_background,
)


# Define test image path
TEST_IMAGE = "tests/test_data/images/octopus.jpg"


def test_autocrop_exists():
    """Test that the test image exists"""
    assert os.path.exists(TEST_IMAGE), f"Test image not found: {TEST_IMAGE}"


def test_autocrop_basic():
    """Test basic autocrop functionality"""
    result = autocrop(TEST_IMAGE)
    assert os.path.exists(result), f"Output file not created: {result}"

    # Clean up
    if os.path.exists(result) and result != TEST_IMAGE:
        os.remove(result)


def test_autocrop_with_output():
    """Test autocrop with specified output path"""
    output_path = "tests/test_data/images/octopus_output.jpg"

    # Remove output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    result = autocrop(TEST_IMAGE, output_path)
    assert result == output_path, f"Output path mismatch: {result} != {output_path}"
    assert os.path.exists(result), f"Output file not created: {result}"

    # Clean up
    if os.path.exists(result) and result != TEST_IMAGE:
        os.remove(result)


def test_autocrop_modes():
    """Test autocrop with different modes"""
    modes = ["auto", "frame", "border"]

    for mode in modes:
        result = autocrop(TEST_IMAGE, mode=mode)
        assert os.path.exists(
            result
        ), f"Output file not created for mode {mode}: {result}"

        # Verify the output filename contains the mode
        if mode == "auto":
            assert (
                "_auto" in result or "_auto_frame" in result or "_auto_border" in result
            )
        elif mode == "frame":
            assert "_frame" in result
        elif mode == "border":
            assert "_border" in result

        # Clean up
        if os.path.exists(result) and result != TEST_IMAGE:
            os.remove(result)


def test_crop_framed_photo():
    """Test frame detection and cropping"""
    result = crop_framed_photo(TEST_IMAGE)
    assert os.path.exists(result), f"Output file not created: {result}"

    # Verify the output is different from input (cropping happened)
    input_img = cv2.imread(TEST_IMAGE)
    output_img = cv2.imread(result)
    assert (
        output_img.shape != input_img.shape
    ), "Image dimensions should change after cropping"

    # Clean up
    if os.path.exists(result) and result != TEST_IMAGE:
        os.remove(result)


def test_crop_with_content_detection():
    """Test content-based border removal"""
    result = crop_with_content_detection(TEST_IMAGE, mode="border")
    assert os.path.exists(result), f"Output file not created: {result}"

    # Verify the output is different from input (cropping happened)
    input_img = cv2.imread(TEST_IMAGE)
    output_img = cv2.imread(result)
    assert (
        output_img.shape != input_img.shape
    ), "Image dimensions should change after cropping"

    # Clean up
    if os.path.exists(result) and result != TEST_IMAGE:
        os.remove(result)


def test_border_percentage():
    """Test border percentage parameter"""
    # Test with positive border percentage (keep more border)
    result_more = autocrop(TEST_IMAGE, mode="border", border_percent=10)

    # Test with negative border percentage (more aggressive crop)
    result_less = autocrop(TEST_IMAGE, mode="border", border_percent=-10)

    # Both should exist
    assert os.path.exists(result_more), f"Output file not created: {result_more}"
    assert os.path.exists(result_less), f"Output file not created: {result_less}"

    # Load images
    img_more = cv2.imread(result_more)
    img_less = cv2.imread(result_less)

    # The more border version should be larger than the less border version
    assert img_more.shape[0] >= img_less.shape[0], "Height comparison failed"
    assert img_more.shape[1] >= img_less.shape[1], "Width comparison failed"

    # Clean up
    for result in [result_more, result_less]:
        if os.path.exists(result) and result != TEST_IMAGE:
            os.remove(result)


def test_margin_parameter():
    """Test margin parameter for frame mode"""
    # Test with positive margin (add margin)
    result_more = autocrop(TEST_IMAGE, mode="frame", margin=10)

    # Test with negative margin (tighter crop)
    result_less = autocrop(TEST_IMAGE, mode="frame", margin=-10)

    # Both should exist
    assert os.path.exists(result_more), f"Output file not created: {result_more}"
    assert os.path.exists(result_less), f"Output file not created: {result_less}"

    # Load images
    img_more = cv2.imread(result_more)
    img_less = cv2.imread(result_less)

    # The more margin version should be larger than the less margin version
    assert img_more.shape[0] >= img_less.shape[0], "Height comparison failed"
    assert img_more.shape[1] >= img_less.shape[1], "Width comparison failed"

    # Clean up
    for result in [result_more, result_less]:
        if os.path.exists(result) and result != TEST_IMAGE:
            os.remove(result)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
