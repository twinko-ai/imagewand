import os
import shutil
import subprocess
from pathlib import Path

import pytest

from imagewand.pdf2img import pdf_to_images


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "test_data" / "images"


@pytest.fixture
def pdf_path(test_data_dir):
    """Return the path to the test PDF file."""
    pdf_file = test_data_dir / "octopus.pdf"
    if not pdf_file.exists():
        pytest.skip(f"Test PDF file not found: {pdf_file}")
    return str(pdf_file)


class TestPDF2Image:
    """Tests for PDF to image conversion functionality."""

    def test_pdf_to_images_function(self, pdf_path, tmp_path):
        """Test the pdf_to_images function directly."""
        output_dir = str(tmp_path / "output")

        # Test with default parameters
        result = pdf_to_images(pdf_path, output_dir)

        # Verify results
        assert len(result) > 0, "Should generate at least one image"
        assert all(
            os.path.exists(img) for img in result
        ), "All image files should exist"
        assert all(
            img.endswith(".jpg") for img in result
        ), "Default format should be jpg"

        # Clean up
        shutil.rmtree(output_dir, ignore_errors=True)

        # Test with custom parameters
        output_dir_png = str(tmp_path / "output_png")
        result_png = pdf_to_images(pdf_path, output_dir_png, dpi=300, format="png")

        # Verify results with custom parameters
        assert len(result_png) > 0, "Should generate at least one image"
        assert all(
            os.path.exists(img) for img in result_png
        ), "All image files should exist"
        assert all(img.endswith(".png") for img in result_png), "Format should be png"

    def test_pdf2img_progress_callback(self, pdf_path, tmp_path):
        """Test that progress callback works correctly."""
        output_dir = str(tmp_path / "progress_test")
        progress_values = []

        def progress_callback(value):
            progress_values.append(value)

        # Run with progress callback
        pdf_to_images(pdf_path, output_dir, progress_callback=progress_callback)

        # Verify callback was called
        assert len(progress_values) > 0, "Progress callback should be called"
        assert progress_values[-1] > 0, "Progress should advance"

        # The last value should correspond to the number of pages
        num_pages = len(list(Path(output_dir).glob("*.jpg")))
        assert (
            progress_values[-1] == num_pages
        ), "Last progress value should equal page count"
