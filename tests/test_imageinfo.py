import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from imagewand.imageinfo import (
    format_file_size,
    get_color_depth,
    get_exif_data,
    get_image_info,
    print_image_info,
)


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "test_data" / "images"


@pytest.fixture
def sample_jpg(test_data_dir):
    """Return the path to the octopus.jpg test image."""
    jpg_path = test_data_dir / "octopus.jpg"
    if not jpg_path.exists():
        pytest.skip(f"Test image not found: {jpg_path}")
    return str(jpg_path)


@pytest.fixture
def sample_pdf(test_data_dir):
    """Return the path to the octopus.pdf test file."""
    pdf_path = test_data_dir / "octopus.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    return str(pdf_path)


@pytest.fixture
def grayscale_image():
    """Create a temporary grayscale image."""
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, "grayscale.png")

    # Create a grayscale image
    img = np.ones((100, 100), dtype=np.uint8) * 128
    cv2.imwrite(img_path, img)

    yield img_path

    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def rgba_image():
    """Create a temporary RGBA image."""
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, "rgba.png")

    # Create an RGBA image
    img = np.ones((100, 100, 4), dtype=np.uint8) * 128
    img[:, :, 3] = 255  # Alpha channel

    # Save as PNG to preserve alpha
    Image.fromarray(img).save(img_path)

    yield img_path

    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def corrupted_image():
    """Create a corrupted image file."""
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, "corrupted.jpg")

    # Create a text file with .jpg extension
    with open(img_path, "w") as f:
        f.write("This is not a valid image file")

    yield img_path

    # Clean up
    shutil.rmtree(temp_dir)


class TestImageInfo:
    """Tests for the image info functionality."""

    def test_get_image_info_jpg(self, sample_jpg):
        """Test getting information from the octopus.jpg image."""
        info = get_image_info(sample_jpg)

        # Check basic file info
        assert info["filename"] == "octopus.jpg"
        assert info["file_extension"] == ".jpg"
        assert info["file_size"] > 0

        # Check image properties
        assert info["width"] > 0
        assert info["height"] > 0
        assert info["format"] in ["JPEG", "JPG"]

        # Check color info
        assert "brightness_mean" in info
        assert "red_mean" in info
        assert "green_mean" in info
        assert "blue_mean" in info
        assert "sharpness" in info

    def test_get_image_info_grayscale(self, grayscale_image):
        """Test getting information from a grayscale image."""
        info = get_image_info(grayscale_image)

        # Check image properties
        assert info["mode"] == "L"
        assert "8-bit" in info["color_depth"]
        assert "grayscale" in info["color_depth"].lower()

        # OpenCV might convert grayscale to BGR, so we can't reliably test for absence of RGB channels
        # Instead, we'll check that the brightness is correct
        assert info["brightness_mean"] == 128.0

    def test_get_image_info_rgba(self, rgba_image):
        """Test getting information from an RGBA image."""
        info = get_image_info(rgba_image)

        # Check image properties
        assert info["mode"] == "RGBA"
        assert "32-bit" in info["color_depth"]
        assert "RGBA" in info["color_depth"]

        # Check color info
        assert "brightness_mean" in info
        assert "red_mean" in info
        assert "green_mean" in info
        assert "blue_mean" in info

    def test_nonexistent_file(self):
        """Test handling of nonexistent files."""
        with pytest.raises(FileNotFoundError):
            get_image_info("nonexistent_image.jpg")

    def test_corrupted_image(self, corrupted_image):
        """Test handling of corrupted image files."""
        try:
            info = get_image_info(corrupted_image)
            # If it doesn't raise an exception, it should have an error field
            assert "error" in info
        except Exception as e:
            # If it raises an exception, it should be a controlled error
            assert "not a valid image file" in str(
                e
            ) or "cannot identify image file" in str(e)

    def test_print_image_info(self, sample_jpg, capsys):
        """Test the print_image_info function."""
        print_image_info(sample_jpg)
        captured = capsys.readouterr()

        # Check that output contains expected sections
        assert "=== IMAGE INFORMATION ===" in captured.out
        assert "File:" in captured.out
        assert "Path:" in captured.out
        assert "Size:" in captured.out
        assert "Dimensions:" in captured.out
        assert "Color Mode:" in captured.out
        assert "DPI:" in captured.out

        # Check that color information is present
        assert "=== COLOR INFORMATION ===" in captured.out
        assert "Average Brightness:" in captured.out
        assert "Sharpness Factor:" in captured.out
        assert "Red:" in captured.out
        assert "Green:" in captured.out
        assert "Blue:" in captured.out

    def test_print_image_info_verbose(self, sample_jpg, capsys):
        """Test the print_image_info function with verbose flag."""
        print_image_info(sample_jpg, verbose=True)
        captured = capsys.readouterr()

        # Basic info should be present
        assert "=== IMAGE INFORMATION ===" in captured.out

        # EXIF section might be present depending on the image
        # But we shouldn't get an error
        assert "Error:" not in captured.out

    def test_print_image_info_error(self, capsys):
        """Test print_image_info with a non-existent file."""
        print_image_info("nonexistent_image.jpg")
        captured = capsys.readouterr()

        assert "Error:" in captured.out
        assert "not found" in captured.out

    def test_format_file_size(self):
        """Test the format_file_size function."""
        assert format_file_size(100) == "100.00 B"
        assert format_file_size(1024) == "1.00 KB"
        assert format_file_size(1024 * 1024) == "1.00 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.00 GB"
        assert format_file_size(1024 * 1024 * 1024 * 1024) == "1.00 TB"

    def test_get_color_depth(self):
        """Test the get_color_depth function."""
        assert "1-bit" in get_color_depth("1")
        assert "8-bit" in get_color_depth("L")
        assert "grayscale" in get_color_depth("L").lower()
        assert "24-bit" in get_color_depth("RGB")
        assert "32-bit" in get_color_depth("RGBA")
        assert "Unknown" in get_color_depth("XYZ")  # Unknown mode

    def test_get_exif_data(self, sample_jpg):
        """Test the get_exif_data function."""
        # Test with an image that has no EXIF data
        with Image.new("RGB", (100, 100)) as img:
            assert get_exif_data(img) is None

        # Test with sample image (may or may not have EXIF)
        with Image.open(sample_jpg) as img:
            exif = get_exif_data(img)
            # We don't know if it has EXIF, but the function should not raise an exception
            assert exif is None or isinstance(exif, dict)

    def test_pdf_file_handling(self, sample_pdf):
        """Test handling of PDF files (should handle gracefully)."""
        try:
            info = get_image_info(sample_pdf)
            # If it succeeds, it should have some basic info
            assert "filename" in info
            assert "file_size" in info
        except Exception as e:
            # If it fails, it should be a controlled error
            assert "not a valid image file" in str(
                e
            ) or "cannot identify image file" in str(e)

    def test_cli_command(self, sample_jpg, monkeypatch):
        """Test the CLI command for image info using monkeypatch."""
        import sys
        from io import StringIO

        # Capture stdout
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        # Import the CLI function
        from imagewand.cli import main

        # Save original argv
        original_argv = sys.argv.copy()

        try:
            # Set up argv for the command
            sys.argv = ["imagewand", "info", sample_jpg]

            # Run the command (should not raise an exception)
            try:
                main()
                # Check that output contains expected information
                output = captured_output.getvalue()
                assert "=== IMAGE INFORMATION ===" in output
                assert "File: octopus.jpg" in output
            except SystemExit as e:
                # main() might call sys.exit(0) which raises SystemExit
                assert e.code == 0
            except Exception as e:
                pytest.fail(f"CLI command raised an exception: {e}")

        finally:
            # Restore original argv
            sys.argv = original_argv
