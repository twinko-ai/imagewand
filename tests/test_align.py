import os
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from imagewand.align import (
    _detect_angle_center,
    _detect_angle_contour,
    _detect_angle_hough,
    align_image,
)

# Path to test data
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data", "images")
OCTOPUS_TILT_PATH = os.path.join(TEST_DATA_DIR, "octopus_tilt.jpg")


@pytest.fixture
def sample_image():
    """Create a temporary sample image for testing."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        # Create a simple test image with a tilted line
        img = np.zeros((200, 200), dtype=np.uint8)
        # Draw a line at 15 degrees
        cv2.line(img, (50, 100), (150, 130), 255, 5)
        # Save the image
        cv2.imwrite(tmp.name, img)

    yield tmp.name
    # Clean up
    if os.path.exists(tmp.name):
        os.remove(tmp.name)


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


def test_align_image_invalid_method():
    """Test alignment with an invalid method"""
    # Create a temporary output path
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_out:
        output_path = tmp_out.name

    try:
        # Call align_image with an invalid method
        result = align_image(OCTOPUS_TILT_PATH, output_path, method="invalid_method")

        # If we get here, the function didn't raise an error
        # Let's check that it fell back to a default method by verifying the output exists
        assert os.path.exists(result), "Output file was not created"
        assert os.path.getsize(result) > 0, "Output file is empty"

        # Check that the method name is not in the output path (should use default)
        assert (
            "invalid_method" not in result
        ), "Invalid method name should not be in output path"

    finally:
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)


def test_align_image_nonexistent_file():
    """Test alignment with a nonexistent file"""
    with pytest.raises(ValueError, match="Could not read image"):
        align_image("nonexistent_file.jpg")


def test_align_image_no_rotation_needed(sample_image):
    """Test alignment when no rotation is needed (angle below threshold)"""
    # Create a temporary output path
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_out:
        output_path = tmp_out.name

    try:
        # Set a high threshold so no rotation is performed
        result = align_image(sample_image, output_path, angle_threshold=20.0)

        # Check that the output file exists
        assert os.path.exists(result), f"Output file not created: {result}"

        # Load the original and aligned images
        original = cv2.imread(sample_image)
        aligned = cv2.imread(result)

        # Check that the images have the same dimensions (no rotation applied)
        assert original.shape == aligned.shape, "Images should have the same dimensions"

    finally:
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)


def test_align_image_with_numpy_array():
    """Test alignment with a numpy array instead of a file path"""
    # Load the test image as a numpy array
    img = cv2.imread(OCTOPUS_TILT_PATH)
    assert img is not None, f"Failed to load test image: {OCTOPUS_TILT_PATH}"

    # Create a temporary output path
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_out:
        output_path = tmp_out.name

    try:
        # Since align_image doesn't support numpy arrays directly, we'll save it first
        temp_input = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
        cv2.imwrite(temp_input, img)

        # Run alignment with the temporary file
        result = align_image(temp_input, output_path)

        # Check that the output file exists
        assert os.path.exists(result), f"Output file not created: {result}"

        # Check that the output file is not empty
        assert os.path.getsize(result) > 0, "Output file is empty"

        # Clean up the temporary input file
        if os.path.exists(temp_input):
            os.remove(temp_input)

    finally:
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)


def test_align_image_with_debug_visualization():
    """Test alignment with visualization of intermediate steps"""
    # Create a temporary output path
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_out:
        output_path = tmp_out.name

    try:
        # Create a real mock for cv2.imwrite that allows the actual function to be called
        real_imwrite = cv2.imwrite

        def mock_imwrite(path, img):
            # Call the real function for all paths
            return real_imwrite(path, img)

        # Patch cv2.imwrite with our mock
        with patch("cv2.imwrite", side_effect=mock_imwrite) as mock_imwrite:
            # Run alignment
            result = align_image(OCTOPUS_TILT_PATH, output_path)

            # Check that the output file exists
            assert os.path.exists(result), f"Output file not created: {result}"

            # Check that cv2.imwrite was called at least once
            assert mock_imwrite.call_count >= 1, "cv2.imwrite was not called"

    finally:
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)


def test_method_selection():
    """Test that different methods are selected correctly"""
    # Create a temporary output path
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_out:
        output_path = tmp_out.name

    try:
        # Test with each method
        methods = ["hough", "contour", "center"]
        for method in methods:
            # Patch the specific detection function
            with patch(f"imagewand.align._detect_angle_{method}") as mock_detect:
                mock_detect.return_value = 10.0  # Mock a 10-degree angle

                # Run alignment with this method
                result = align_image(OCTOPUS_TILT_PATH, output_path, method=method)

                # Check that the specific detection function was called
                mock_detect.assert_called_once()

                # Clean up the result
                if os.path.exists(result) and result != output_path:
                    os.remove(result)

    finally:
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
