import os
from pathlib import Path

import pytest

from imagewand.imageinfo import get_image_info, print_image_info


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


class TestImageInfo:
    """Tests for the image info functionality."""

    def test_get_image_info_jpg(self, sample_jpg):
        """Test getting information from the octopus.jpg image."""
        info = get_image_info(sample_jpg)

        # Check basic file information
        assert info["filename"] == "octopus.jpg"
        assert info["path"] == str(Path(sample_jpg).absolute())
        assert info["file_size"] > 0
        assert "file_size_human" in info
        assert "file_modified" in info
        assert info["file_extension"] == ".jpg"

        # Check image properties
        assert info["width"] > 0
        assert info["height"] > 0
        assert info["format"] == "JPEG"
        assert info["mode"] == "RGB"
        assert "24-bit" in info["color_depth"]

        # Check color information
        assert "red_min" in info
        assert "red_max" in info
        assert "red_mean" in info
        assert "green_min" in info
        assert "green_max" in info
        assert "green_mean" in info
        assert "blue_min" in info
        assert "blue_max" in info
        assert "blue_mean" in info
        assert "brightness_mean" in info
        assert "sharpness" in info

    def test_nonexistent_file(self):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            get_image_info("nonexistent_image.jpg")

    def test_print_image_info(self, sample_jpg, capsys):
        """Test the print_image_info function."""
        print_image_info(sample_jpg, verbose=False)
        captured = capsys.readouterr()

        # Check that output contains expected information
        assert "=== IMAGE INFORMATION ===" in captured.out
        assert "File: octopus.jpg" in captured.out
        assert "Dimensions:" in captured.out
        assert "Color Mode:" in captured.out

        # Check color information is printed
        assert "=== COLOR INFORMATION ===" in captured.out
        assert "Average Brightness:" in captured.out
        assert "Color Channels:" in captured.out
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
