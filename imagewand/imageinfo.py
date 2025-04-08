import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import ExifTags, Image


def get_image_info(image_path: str) -> Dict[str, Any]:
    """
    Get comprehensive information about an image file.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary containing image information
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    file_path = Path(image_path)
    file_size = file_path.stat().st_size
    file_modified = datetime.fromtimestamp(file_path.stat().st_mtime)

    # Basic file info
    info = {
        "filename": file_path.name,
        "path": str(file_path.absolute()),
        "file_size": file_size,
        "file_size_human": format_file_size(file_size),
        "file_modified": file_modified.strftime("%Y-%m-%d %H:%M:%S"),
        "file_extension": file_path.suffix.lower(),
    }

    # Get image details using PIL
    try:
        with Image.open(image_path) as img:
            info["width"] = img.width
            info["height"] = img.height
            info["format"] = img.format
            info["mode"] = img.mode
            info["color_depth"] = get_color_depth(img.mode)

            # Get DPI information
            try:
                dpi = img.info.get("dpi")
                if dpi:
                    info["dpi_x"] = dpi[0]
                    info["dpi_y"] = dpi[1]
                else:
                    # Try to get from EXIF
                    exif = get_exif_data(img)
                    if (
                        exif and 282 in exif and 283 in exif
                    ):  # XResolution and YResolution
                        info["dpi_x"] = exif[282]
                        info["dpi_y"] = exif[283]
                    else:
                        info["dpi_x"] = "Unknown"
                        info["dpi_y"] = "Unknown"
            except Exception:
                info["dpi_x"] = "Unknown"
                info["dpi_y"] = "Unknown"

            # Get EXIF data
            exif_data = get_exif_data(img)
            if exif_data:
                info["has_exif"] = True
                info["exif"] = exif_data
            else:
                info["has_exif"] = False

    except Exception as e:
        info["error"] = f"Error reading image with PIL: {str(e)}"

    # Get additional details using OpenCV
    try:
        img_cv = cv2.imread(image_path)
        if img_cv is not None:
            # Color statistics
            if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
                channels = cv2.split(img_cv)
                for i, channel in enumerate(["blue", "green", "red"]):
                    info[f"{channel}_min"] = int(np.min(channels[i]))
                    info[f"{channel}_max"] = int(np.max(channels[i]))
                    info[f"{channel}_mean"] = round(float(np.mean(channels[i])), 2)

                # Calculate histogram for brightness
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                info["brightness_mean"] = round(float(np.mean(gray)), 2)

                # Detect if image might be blurry
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                info["sharpness"] = round(laplacian_var, 2)
                info["is_blurry"] = laplacian_var < 100  # Threshold can be adjusted

    except Exception as e:
        info["cv_error"] = f"Error analyzing with OpenCV: {str(e)}"

    return info


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_color_depth(mode: str) -> str:
    """Get color depth information from PIL mode"""
    mode_info = {
        "1": "1-bit (black and white)",
        "L": "8-bit (grayscale)",
        "P": "8-bit (palette)",
        "RGB": "24-bit (RGB)",
        "RGBA": "32-bit (RGBA)",
        "CMYK": "32-bit (CMYK)",
        "YCbCr": "24-bit (YCbCr)",
        "LAB": "24-bit (LAB)",
        "HSV": "24-bit (HSV)",
        "I": "32-bit (integer)",
        "F": "32-bit (float)",
    }
    return mode_info.get(mode, f"Unknown ({mode})")


def get_exif_data(img: Image.Image) -> Optional[Dict[str, Any]]:
    """Extract EXIF data from image if available"""
    try:
        exif = {
            ExifTags.TAGS.get(k, k): v
            for k, v in img._getexif().items()
            if k in ExifTags.TAGS
        }
        return exif
    except (AttributeError, TypeError, ValueError):
        return None


def print_image_info(image_path: str, verbose: bool = False) -> None:
    """
    Print image information to console.

    Args:
        image_path: Path to the image file
        verbose: Whether to print detailed information
    """
    try:
        info = get_image_info(image_path)

        print("\n=== IMAGE INFORMATION ===")
        print(f"File: {info['filename']}")
        print(f"Path: {info['path']}")
        print(f"Size: {info['file_size_human']} ({info['file_size']} bytes)")
        print(f"Modified: {info['file_modified']}")
        print(f"Format: {info['format']}")
        print(f"Dimensions: {info['width']} × {info['height']} pixels")
        print(f"Color Mode: {info['mode']} - {info['color_depth']}")

        print(f"DPI: {info.get('dpi_x', 'Unknown')} × {info.get('dpi_y', 'Unknown')}")

        if "brightness_mean" in info:
            print(f"\n=== COLOR INFORMATION ===")
            print(f"Average Brightness: {info['brightness_mean']}/255")
            print(f"Sharpness Factor: {info.get('sharpness', 'Unknown')}")
            if info.get("is_blurry"):
                print("Image appears to be blurry")

            if "red_mean" in info:
                print("\nColor Channels:")
                print(
                    f"  Red: min={info['red_min']}, max={info['red_max']}, avg={info['red_mean']}"
                )
                print(
                    f"  Green: min={info['green_min']}, max={info['green_max']}, avg={info['green_mean']}"
                )
                print(
                    f"  Blue: min={info['blue_min']}, max={info['blue_max']}, avg={info['blue_mean']}"
                )

        if verbose and info.get("has_exif"):
            print("\n=== EXIF INFORMATION ===")
            for key, value in info["exif"].items():
                if isinstance(value, bytes):
                    continue  # Skip binary data
                print(f"{key}: {value}")

        print("\n")

    except Exception as e:
        print(f"Error: {str(e)}")
