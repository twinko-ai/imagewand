"""Tests for the background removal functionality."""

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
from PIL import Image

from imagewand.rmbg import batch_remove_background, remove_background


@pytest.fixture
def test_image():
    """Create a test image for background removal tests."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create a test image
    img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 255))
    test_image_path = os.path.join(temp_dir, 'test_image.png')
    img.save(test_image_path)
    
    yield test_image_path
    
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_images():
    """Create multiple test images for batch processing tests."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create test images
    img_paths = []
    for i in range(3):
        # Create different colored test images
        color = (255, i * 80, 0, 255)
        img = Image.new('RGBA', (100, 100), color=color)
        img_path = os.path.join(temp_dir, f'test_image_{i}.png')
        img.save(img_path)
        img_paths.append(img_path)
    
    yield img_paths, temp_dir
    
    # Clean up
    shutil.rmtree(temp_dir)


def mock_rembg_remove(input_image, **kwargs):
    """Mock for rembg.remove function."""
    # Create a copy of the input image with transparency
    result = input_image.copy()
    # Add an alpha channel if not present
    if result.mode != 'RGBA':
        result = result.convert('RGBA')
    # Make a transparent border to simulate background removal
    width, height = result.size
    border_width = 10
    for y in range(height):
        for x in range(width):
            if (x < border_width or x >= width - border_width or 
                y < border_width or y >= height - border_width):
                # Make border pixels transparent
                result.putpixel((x, y), (0, 0, 0, 0))
    return result


@patch('rembg.remove', mock_rembg_remove)
@patch('rembg.new_session', lambda model_name: None)
def test_remove_background(test_image):
    """Test the background removal function."""
    # Run the function
    output_path = remove_background(test_image)
    
    # Check that output file exists
    assert os.path.exists(output_path)
    
    # Check that the output is a PNG (supports transparency)
    assert output_path.endswith('.png')
    
    # Check the output image
    output_img = Image.open(output_path)
    
    # Verify that the output image has an alpha channel
    assert output_img.mode == 'RGBA'
    
    # Check that the border pixels are transparent
    width, height = output_img.size
    assert output_img.getpixel((5, 5))[3] == 0  # Border should be transparent
    assert output_img.getpixel((width // 2, height // 2))[3] == 255  # Center should be opaque
    
    # Clean up
    os.remove(output_path)


@patch('rembg.remove', mock_rembg_remove)
@patch('rembg.new_session', lambda model_name: None)
def test_batch_remove_background(test_images):
    """Test the batch background removal function."""
    img_paths, temp_dir = test_images
    output_dir = os.path.join(temp_dir, 'output')
    
    # Run the function
    output_paths = batch_remove_background(img_paths, output_dir)
    
    # Check that the output directory was created
    assert os.path.exists(output_dir)
    
    # Check that we got the expected number of output files
    assert len(output_paths) == len(img_paths)
    
    # Check each output file
    for path in output_paths:
        # Check that the file exists
        assert os.path.exists(path)
        
        # Check that it's a PNG
        assert path.endswith('.png')
        
        # Check the output image
        output_img = Image.open(path)
        
        # Verify that the output image has an alpha channel
        assert output_img.mode == 'RGBA'
        
        # Check that the border pixels are transparent
        width, height = output_img.size
        assert output_img.getpixel((5, 5))[3] == 0  # Border should be transparent
        assert output_img.getpixel((width // 2, height // 2))[3] == 255  # Center should be opaque 