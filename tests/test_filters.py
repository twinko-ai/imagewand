import os
import pytest
import shutil
from pathlib import Path
import tempfile
import subprocess
from PIL import Image

from imagewand.filters import apply_filters, parse_filter_string, list_filters

# Test image path
TEST_IMAGE = "tests/test_data/images/octopus.jpg"
TEST_OUTPUT_DIR = "tests/test_data/output"


@pytest.fixture(scope="module")
def setup_test_directory():
    """Setup test directory and ensure test image exists"""
    # Create output directory if it doesn't exist
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    # Check if test image exists
    assert os.path.exists(TEST_IMAGE), f"Test image not found: {TEST_IMAGE}"

    # Create a temporary directory for batch processing tests
    temp_dir = tempfile.mkdtemp()
    try:
        # Copy test image to temp dir
        shutil.copy(TEST_IMAGE, temp_dir)
        yield temp_dir
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)


def test_single_filter():
    """Test applying a single filter"""
    output_path = os.path.join(TEST_OUTPUT_DIR, "octopus_grayscale.jpg")

    # Remove output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Apply grayscale filter
    result = apply_filters(TEST_IMAGE, ["grayscale"], output_path)

    assert os.path.exists(result), f"Output file not created: {result}"

    # Verify the filtered image exists and has reasonable dimensions
    img = Image.open(result)
    assert img is not None, "Failed to read filtered image"
    assert img.size[0] > 0 and img.size[1] > 0, "Filtered image has invalid dimensions"

    # Clean up
    if os.path.exists(result):
        os.remove(result)


def test_multiple_filters():
    """Test applying multiple filters"""
    output_path = os.path.join(TEST_OUTPUT_DIR, "octopus_grayscale_sharpen.jpg")

    # Remove output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Apply grayscale and sharpen filters
    result = apply_filters(TEST_IMAGE, ["grayscale", "sharpen"], output_path)

    assert os.path.exists(result), f"Output file not created: {result}"

    # Verify the filtered image exists and has reasonable dimensions
    img = Image.open(result)
    assert img is not None, "Failed to read filtered image"
    assert img.size[0] > 0 and img.size[1] > 0, "Filtered image has invalid dimensions"

    # Clean up
    if os.path.exists(result):
        os.remove(result)


def test_filter_with_parameters():
    """Test applying a filter with custom parameters"""
    output_path = os.path.join(TEST_OUTPUT_DIR, "octopus_contrast.jpg")

    # Remove output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Apply contrast filter with custom factor
    filter_string = "contrast:factor=1.2"
    filter_name, params = parse_filter_string(filter_string)

    result = apply_filters(TEST_IMAGE, [filter_string], output_path)

    assert os.path.exists(result), f"Output file not created: {result}"
    assert params["factor"] == 1.2, "Parameter not parsed correctly"

    # Clean up
    if os.path.exists(result):
        os.remove(result)


def test_multiple_filters_with_parameters():
    """Test applying multiple filters with custom parameters"""
    output_path = os.path.join(TEST_OUTPUT_DIR, "octopus_multi_params.jpg")

    # Remove output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Apply multiple filters with custom parameters
    filter_strings = [
        "saturation:factor=1.3",
        "contrast:factor=1.2",
        "sharpen:factor=1.8",
    ]

    result = apply_filters(TEST_IMAGE, filter_strings, output_path)

    assert os.path.exists(result), f"Output file not created: {result}"

    # Clean up
    if os.path.exists(result):
        os.remove(result)


def test_mix_default_and_custom_parameters():
    """Test applying filters with mix of default and custom parameters"""
    output_path = os.path.join(TEST_OUTPUT_DIR, "octopus_mix_params.jpg")

    # Remove output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Apply filters with mix of default and custom parameters
    filter_strings = ["saturation:factor=1.3", "contrast", "sharpen:factor=2.0"]

    result = apply_filters(TEST_IMAGE, filter_strings, output_path)

    assert os.path.exists(result), f"Output file not created: {result}"

    # Clean up
    if os.path.exists(result):
        os.remove(result)


def test_batch_processing(setup_test_directory):
    """Test batch processing of images"""
    from imagewand.filters import batch_apply_filters

    temp_dir = setup_test_directory
    output_dir = os.path.join(TEST_OUTPUT_DIR, "batch")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all images in temp directory
    image_paths = [
        os.path.join(temp_dir, f)
        for f in os.listdir(temp_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Apply grayscale filter to all images
    results = batch_apply_filters(image_paths, ["grayscale"], output_dir)

    assert len(results) > 0, "No images were processed"
    for result in results:
        assert os.path.exists(result), f"Output file not created: {result}"

    # Clean up
    shutil.rmtree(output_dir, ignore_errors=True)


def test_list_filters():
    """Test listing available filters"""
    filters = list_filters()

    # Check that we have the expected filters
    expected_filters = [
        "grayscale",
        "sepia",
        "blur",
        "sharpen",
        "brightness",
        "contrast",
        "saturation",
        "vignette",
        "edge_enhance",
    ]

    for filter_name in expected_filters:
        assert filter_name in filters, f"Expected filter '{filter_name}' not found"


def test_cli_filter_command():
    """Test the CLI filter command"""
    output_path = os.path.join(TEST_OUTPUT_DIR, "octopus_cli_grayscale.jpg")

    # Remove output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Run CLI command
    cmd = ["imagewand", "filter", TEST_IMAGE, "-f", "grayscale", "-o", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, f"CLI command failed: {result.stderr}"
    assert os.path.exists(output_path), f"Output file not created: {output_path}"

    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)


def test_cli_multiple_filters():
    """Test the CLI command with multiple filters"""
    output_path = os.path.join(TEST_OUTPUT_DIR, "octopus_cli_multi.jpg")

    # Remove output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Run CLI command
    cmd = [
        "imagewand",
        "filter",
        TEST_IMAGE,
        "-f",
        "grayscale,sharpen",
        "-o",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, f"CLI command failed: {result.stderr}"
    assert os.path.exists(output_path), f"Output file not created: {output_path}"

    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)


def test_cli_filter_with_parameters():
    """Test the CLI command with filter parameters"""
    output_path = os.path.join(TEST_OUTPUT_DIR, "octopus_cli_params.jpg")

    # Remove output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Run CLI command
    cmd = [
        "imagewand",
        "filter",
        TEST_IMAGE,
        "-f",
        "contrast:factor=1.2",
        "-o",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, f"CLI command failed: {result.stderr}"
    assert os.path.exists(output_path), f"Output file not created: {output_path}"

    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
