import os
import pytest
import cv2
import numpy as np
from imagewand.align import (
    align_image,
    _detect_angle_hough,
    _detect_angle_contour,
    _detect_angle_center,
)


# Path to test data
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data", "images")
OCTOPUS_TILT_PATH = os.path.join(TEST_DATA_DIR, "octopus_tilt.jpg")


def test_align_image_exists():
    """Test that the test image exists"""
    assert os.path.exists(
        OCTOPUS_TILT_PATH
    ), f"Test image not found: {OCTOPUS_TILT_PATH}"


def test_align_image_basic():
    """Test basic alignment functionality"""
    # Create a temporary output path
    output_path = os.path.join(
        os.path.dirname(OCTOPUS_TILT_PATH), "octopus_aligned.jpg"
    )

    try:
        # Run alignment
        result = align_image(OCTOPUS_TILT_PATH, output_path)

        # Check that the output file exists
        assert os.path.exists(result), f"Output file not created: {result}"

        # Check that the output file is not empty
        assert os.path.getsize(result) > 0, "Output file is empty"

        # Load the original and aligned images
        original = cv2.imread(OCTOPUS_TILT_PATH)
        aligned = cv2.imread(result)

        # Check that the images have the same number of channels
        assert original.shape[2] == aligned.shape[2], "Channel count mismatch"

    finally:
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)


def test_align_image_methods():
    """Test different alignment methods"""
    methods = ["auto", "hough", "contour", "center"]

    for method in methods:
        # Create a temporary output path
        output_path = os.path.join(
            os.path.dirname(OCTOPUS_TILT_PATH), f"octopus_aligned_{method}.jpg"
        )

        try:
            # Run alignment with the current method
            result = align_image(OCTOPUS_TILT_PATH, output_path, method=method)

            # Check that the output file exists
            assert os.path.exists(
                result
            ), f"Output file not created for method {method}: {result}"

            # Check that the output file is not empty
            assert (
                os.path.getsize(result) > 0
            ), f"Output file is empty for method {method}"

        finally:
            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)


def test_align_image_angle_threshold():
    """Test alignment with different angle thresholds"""
    thresholds = [0.5, 1.0, 2.0]

    for threshold in thresholds:
        # Create a temporary output path
        output_path = os.path.join(
            os.path.dirname(OCTOPUS_TILT_PATH),
            f"octopus_aligned_threshold_{threshold}.jpg",
        )

        try:
            # Run alignment with the current threshold
            result = align_image(
                OCTOPUS_TILT_PATH, output_path, angle_threshold=threshold
            )

            # Check that the output file exists
            assert os.path.exists(
                result
            ), f"Output file not created for threshold {threshold}: {result}"

            # Check that the output file is not empty
            assert (
                os.path.getsize(result) > 0
            ), f"Output file is empty for threshold {threshold}"

        finally:
            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)


def test_detect_angle_functions():
    """Test the individual angle detection functions"""
    # Load the test image
    img = cv2.imread(OCTOPUS_TILT_PATH)
    assert img is not None, f"Failed to load test image: {OCTOPUS_TILT_PATH}"

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Test each angle detection function
    angle_hough = _detect_angle_hough(gray)
    angle_contour = _detect_angle_contour(gray)
    angle_center = _detect_angle_center(gray)

    # Check that the angles are within a reasonable range (-45 to 45 degrees)
    assert -45 <= angle_hough <= 45, f"Hough angle out of range: {angle_hough}"
    assert -45 <= angle_contour <= 45, f"Contour angle out of range: {angle_contour}"
    assert -45 <= angle_center <= 45, f"Center angle out of range: {angle_center}"

    # Print the detected angles for information
    print(
        f"Detected angles - Hough: {angle_hough:.2f}°, Contour: {angle_contour:.2f}°, Center: {angle_center:.2f}°"
    )


def test_filename_generation():
    """Test that the output filename includes method and threshold when not specified"""
    # Test with non-default parameters
    method = "hough"
    threshold = 0.5

    # Run alignment without specifying output path
    try:
        result = align_image(
            OCTOPUS_TILT_PATH, method=method, angle_threshold=threshold
        )

        # Check that the output filename contains the method and threshold
        expected_suffix = f"_aligned_{method}_a{threshold:.1f}.jpg"
        assert (
            expected_suffix in result
        ), f"Output filename {result} does not contain expected suffix {expected_suffix}"

    finally:
        # Clean up
        if os.path.exists(result):
            os.remove(result)

    # Test with default parameters
    try:
        result = align_image(OCTOPUS_TILT_PATH)

        # Check that the output filename is simple for default parameters
        expected_suffix = "_aligned.jpg"
        assert (
            expected_suffix in result
        ), f"Output filename {result} does not contain expected suffix {expected_suffix}"

    finally:
        # Clean up
        if os.path.exists(result):
            os.remove(result)
