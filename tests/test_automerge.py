import os
import pytest
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil
import glob

from imagewand.automerge import (
    automerge,
    AutoMerge
)


# Define test image paths
TEST_IMAGES_DIR = "tests/test_data/images/shark"
TEST_OUTPUT_DIR = "tests/test_data/output"


@pytest.fixture(scope="module")
def setup_test_directory():
    """Setup test directory and ensure shark images exist"""
    # Create output directory if it doesn't exist
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    
    # Check if shark images exist
    shark_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
    assert len(shark_images) >= 2, f"Not enough test images found in {TEST_IMAGES_DIR}"
    
    # Return sorted list of shark images
    return sorted(shark_images)


def test_automerge_basic(setup_test_directory):
    """Test basic automerge functionality with shark images"""
    input_images = setup_test_directory
    output_path = os.path.join(TEST_OUTPUT_DIR, "merged_shark.jpg")
    
    # Remove output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
    
    result = automerge(input_images, output_path)
    assert os.path.exists(result), f"Output file not created: {result}"
    
    # Verify the merged image exists and has reasonable dimensions
    merged_img = cv2.imread(result)
    assert merged_img is not None, "Failed to read merged image"
    assert merged_img.shape[0] > 0 and merged_img.shape[1] > 0, "Merged image has invalid dimensions"
    
    # Clean up
    if os.path.exists(result):
        os.remove(result)


def test_automerge_directory(setup_test_directory):
    """Test automerge with directory input"""
    output_path = os.path.join(TEST_OUTPUT_DIR, "merged_shark_dir.jpg")
    
    # Remove output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Test automerge with directory input
    result = automerge(TEST_IMAGES_DIR, output_path)
    
    assert os.path.exists(result), f"Output file not created: {result}"
    
    # Verify the merged image exists and has reasonable dimensions
    merged_img = cv2.imread(result)
    assert merged_img is not None, "Failed to read merged image"
    assert merged_img.shape[0] > 0 and merged_img.shape[1] > 0, "Merged image has invalid dimensions"
    
    # Clean up
    if os.path.exists(result):
        os.remove(result)


def test_automerge_glob_pattern(setup_test_directory):
    """Test automerge with glob pattern input"""
    output_path = os.path.join(TEST_OUTPUT_DIR, "merged_shark_glob.jpg")
    
    # Remove output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Test automerge with glob pattern input
    glob_pattern = os.path.join(TEST_IMAGES_DIR, "*.jpg")
    result = automerge(glob_pattern, output_path)
    
    assert os.path.exists(result), f"Output file not created: {result}"
    
    # Verify the merged image exists and has reasonable dimensions
    merged_img = cv2.imread(result)
    assert merged_img is not None, "Failed to read merged image"
    assert merged_img.shape[0] > 0 and merged_img.shape[1] > 0, "Merged image has invalid dimensions"
    
    # Clean up
    if os.path.exists(result):
        os.remove(result)


def test_automerge_subset_of_images(setup_test_directory):
    """Test automerge with a subset of shark images"""
    # Use only the first 2-3 images
    input_images = setup_test_directory[:3]
    assert len(input_images) >= 2, "Need at least 2 test images"
    
    output_path = os.path.join(TEST_OUTPUT_DIR, "merged_shark_subset.jpg")
    
    # Remove output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
    
    result = automerge(input_images, output_path)
    assert os.path.exists(result), f"Output file not created: {result}"
    
    # Verify the merged image exists and has reasonable dimensions
    merged_img = cv2.imread(result)
    assert merged_img is not None, "Failed to read merged image"
    assert merged_img.shape[0] > 0 and merged_img.shape[1] > 0, "Merged image has invalid dimensions"
    
    # Clean up
    if os.path.exists(result):
        os.remove(result)


def test_automerge_with_debug_flag(setup_test_directory):
    """Test automerge with debug flag enabled"""
    input_images = setup_test_directory
    output_path = os.path.join(TEST_OUTPUT_DIR, "merged_shark_debug.jpg")
    
    # Remove output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # This should run without errors, even with debug=True
    result = automerge(input_images, output_path, debug=True)
    assert os.path.exists(result), f"Output file not created: {result}"
    
    # Clean up
    if os.path.exists(result):
        os.remove(result)


def test_automerge_class_directly(setup_test_directory):
    """Test using the AutoMerge class directly"""
    input_images = setup_test_directory
    
    # Load images
    images = []
    for path in input_images[:3]:  # Use first 3 images
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    
    assert len(images) >= 2, "Need at least 2 valid images"
    
    # Create AutoMerge instance and merge images
    merger = AutoMerge(debug=False)
    result = merger.merge_images(images)
    
    # Verify the result
    assert result is not None, "Merge result should not be None"
    assert isinstance(result, np.ndarray), "Result should be a numpy array"
    assert result.shape[0] > 0 and result.shape[1] > 0, "Result has invalid dimensions"
    assert len(result.shape) == 3, "Result should be a color image"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 