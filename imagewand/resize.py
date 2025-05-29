import os
import re

import cv2
from PIL import Image

MIN_WIDTH = 1500
MAX_WIDTH = 2500
MAX_SIZE_KB = 500
MAX_SIZE_BYTES = MAX_SIZE_KB * 1024


def parse_file_size(size_str: str) -> int:
    """
    Parse file size string to bytes.

    Args:
        size_str: Size string like "5MB", "500KB", "2.5GB"

    Returns:
        Size in bytes
    """
    size_str = size_str.upper().strip()

    # Match number and unit
    match = re.match(r"^([\d.]+)\s*([KMGT]?B?)$", size_str)
    if not match:
        raise ValueError(
            f"Invalid size format: {size_str}. Use formats like '5MB', '500KB', '2.5GB'"
        )

    number, unit = match.groups()
    number = float(number)

    # Convert to bytes
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 * 1024,
        "GB": 1024 * 1024 * 1024,
        "TB": 1024 * 1024 * 1024 * 1024,
    }

    # Handle cases where unit might be just 'K', 'M', 'G', 'T'
    if unit in ["K", "M", "G", "T"]:
        unit += "B"
    elif unit == "":
        unit = "B"

    if unit not in multipliers:
        raise ValueError(f"Unknown size unit: {unit}")

    return int(number * multipliers[unit])


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)


def resize_to_target_size(
    input_path: str,
    target_size: str,
    output_path: str = None,
    quality_range: tuple = (20, 95),
    max_iterations: int = 10,
    progress_callback=None,
) -> str:
    """
    Resize an image to achieve a target file size.

    Args:
        input_path: Path to input image
        target_size: Target file size (e.g., "5MB", "500KB")
        output_path: Output path (optional)
        quality_range: Min and max quality for JPEG compression (20-95)
        max_iterations: Maximum iterations to find target size
        progress_callback: Function to call with progress updates

    Returns:
        Path to resized image
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    target_bytes = parse_file_size(target_size)

    # Create output path if not specified
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_resized_{target_size.lower().replace('.', 'p')}{ext}"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Get original image info
    with Image.open(input_path) as img:
        original_width, original_height = img.size
        original_format = img.format
        dpi = img.info.get("dpi")

    # Determine output format
    output_ext = os.path.splitext(output_path)[1].lower()
    if output_ext in [".jpg", ".jpeg"]:
        use_jpeg = True
    elif output_ext in [".png"]:
        use_jpeg = False
    else:
        # Default to JPEG for size optimization
        use_jpeg = True
        base = os.path.splitext(output_path)[0]
        output_path = f"{base}.jpg"

    min_quality, max_quality = quality_range
    best_path = None
    best_size_diff = float("inf")

    # Binary search approach for finding optimal size
    scale_min, scale_max = 0.1, 1.0  # Scale factors for dimensions

    for iteration in range(max_iterations):
        if progress_callback:
            progress_callback(int((iteration / max_iterations) * 90))

        # Try different scale factors
        scale = (scale_min + scale_max) / 2
        new_width = max(1, int(original_width * scale))
        new_height = max(1, int(original_height * scale))

        # Create temporary file for testing
        temp_path = output_path + f".temp_{iteration}.jpg"

        try:
            # Resize image
            img = cv2.imread(input_path)
            resized = cv2.resize(
                img, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

            if use_jpeg:
                # Try different quality levels within current scale
                quality_min, quality_max = min_quality, max_quality

                for quality_iter in range(3):  # Max 3 quality iterations per scale
                    quality = int((quality_min + quality_max) / 2)

                    # Save with specific quality
                    cv2.imwrite(temp_path, resized, [cv2.IMWRITE_JPEG_QUALITY, quality])

                    # Add DPI back if available
                    if dpi:
                        try:
                            with Image.open(temp_path) as pil_img:
                                pil_img.save(temp_path, dpi=dpi, quality=quality)
                        except Exception:
                            pass

                    current_size = get_file_size(temp_path)
                    size_diff = abs(current_size - target_bytes)

                    # Check if this is the best result so far
                    if size_diff < best_size_diff:
                        best_size_diff = size_diff
                        if best_path and best_path != temp_path:
                            try:
                                os.remove(best_path)
                            except:
                                pass
                        best_path = temp_path + f"_best"
                        os.rename(temp_path, best_path)
                    else:
                        try:
                            os.remove(temp_path)
                        except:
                            pass

                    # Adjust quality for next iteration
                    if current_size > target_bytes:
                        quality_max = quality - 1
                    elif current_size < target_bytes:
                        quality_min = quality + 1
                    else:
                        break  # Perfect match

                    if quality_min >= quality_max:
                        break
            else:
                # PNG - only scale, no quality adjustment
                cv2.imwrite(temp_path, resized, [cv2.IMWRITE_PNG_COMPRESSION, 6])

                # Add DPI back if available
                if dpi:
                    try:
                        with Image.open(temp_path) as pil_img:
                            pil_img.save(temp_path, dpi=dpi, compress_level=6)
                    except Exception:
                        pass

                current_size = get_file_size(temp_path)
                size_diff = abs(current_size - target_bytes)

                if size_diff < best_size_diff:
                    best_size_diff = size_diff
                    if best_path and best_path != temp_path:
                        try:
                            os.remove(best_path)
                        except:
                            pass
                    best_path = temp_path + f"_best"
                    os.rename(temp_path, best_path)
                else:
                    try:
                        os.remove(temp_path)
                    except:
                        pass

            # Get file size of current best attempt
            if best_path and os.path.exists(best_path):
                current_best_size = get_file_size(best_path)

                # If we're close enough (within 5% of target), stop
                if abs(current_best_size - target_bytes) / target_bytes < 0.05:
                    break

                # Adjust scale for next iteration
                if current_best_size > target_bytes:
                    scale_max = scale
                else:
                    scale_min = scale

        except Exception as e:
            # Clean up temp file on error
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass

            # If this is the last iteration, raise the error
            if iteration == max_iterations - 1:
                raise RuntimeError(f"Failed to resize to target size: {str(e)}")

    # Move best result to final output path
    if best_path and os.path.exists(best_path):
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(best_path, output_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save final result: {str(e)}")
    else:
        raise RuntimeError("Failed to create resized image within target size")

    if progress_callback:
        progress_callback(100)

    final_size = get_file_size(output_path)
    print(f"Target: {target_size} ({target_bytes:,} bytes)")
    print(f"Achieved: {final_size:,} bytes ({final_size/1024/1024:.2f} MB)")
    print(
        f"Difference: {abs(final_size - target_bytes):,} bytes ({abs(final_size - target_bytes)/target_bytes*100:.1f}%)"
    )

    return output_path


def resize_image(
    input_path,
    output_path=None,
    width=None,
    height=None,
    target_size=None,
    percent=None,
    progress_callback=None,
):
    """
    Resize an image to the specified dimensions, percentage, or target file size while preserving DPI information.

    Args:
        input_path (str): Path to the input image
        output_path (str, optional): Path to save the resized image. If None, will create one based on input
        width (int, optional): Target width in pixels
        height (int, optional): Target height in pixels
        target_size (str, optional): Target file size (e.g., "5MB", "500KB")
        percent (float, optional): Resize percentage (e.g., 50.0 for 50%)
        progress_callback (callable, optional): Function to call with progress updates

    Returns:
        str: Path to the resized image
    """
    # If target_size is specified, use the target size resizing function
    if target_size:
        return resize_to_target_size(
            input_path, target_size, output_path, progress_callback=progress_callback
        )

    # Validate inputs for dimension-based resizing
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if width is None and height is None and percent is None:
        raise ValueError(
            "Either width, height, percent, or target_size must be specified"
        )

    # First, get the DPI information from the original image using PIL
    try:
        with Image.open(input_path) as pil_img:
            dpi = pil_img.info.get("dpi")
    except Exception:
        dpi = None

    # Read the image with OpenCV for resizing
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")

    # Get original dimensions
    h, w = img.shape[:2]

    # Calculate new dimensions
    if percent is not None:
        # Percentage-based resizing
        new_w = int(w * percent / 100)
        new_h = int(h * percent / 100)
    elif width is not None and height is not None:
        new_w, new_h = width, height
    elif width is not None:
        # Maintain aspect ratio based on width
        ratio = width / w
        new_w = width
        new_h = int(h * ratio)
    else:
        # Maintain aspect ratio based on height
        ratio = height / h
        new_h = height
        new_w = int(w * ratio)

    # Report progress at 33%
    if progress_callback:
        progress_callback(33)

    # Resize the image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create output path if not specified
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        # Include resize information in filename
        if percent is not None:
            output_path = f"{base}_resized_{percent}pct{ext}"
        elif width is not None and height is not None:
            output_path = f"{base}_resized_{new_w}x{new_h}{ext}"
        elif width is not None:
            output_path = f"{base}_resized_w{new_w}{ext}"
        else:
            output_path = f"{base}_resized_h{new_h}{ext}"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Report progress at 66%
    if progress_callback:
        progress_callback(66)

    # Save the resized image with OpenCV
    cv2.imwrite(output_path, resized)

    # If we have DPI information, add it back using PIL
    if dpi:
        try:
            with Image.open(output_path) as pil_img:
                # Create a new image with the same content but with DPI info
                pil_img.save(output_path, dpi=dpi)
        except Exception as e:
            print(f"Warning: Could not preserve DPI information: {e}")

    # Report progress at 100%
    if progress_callback:
        progress_callback(100)

    return output_path
