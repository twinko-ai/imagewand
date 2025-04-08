import os
import cv2
from PIL import Image

MIN_WIDTH = 1500
MAX_WIDTH = 2500
MAX_SIZE_KB = 500
MAX_SIZE_BYTES = MAX_SIZE_KB * 1024


def resize_image(
    input_path, output_path=None, width=None, height=None, progress_callback=None
):
    """
    Resize an image to the specified dimensions while preserving DPI information.

    Args:
        input_path (str): Path to the input image
        output_path (str, optional): Path to save the resized image. If None, will create one based on input
        width (int, optional): Target width in pixels
        height (int, optional): Target height in pixels
        progress_callback (callable, optional): Function to call with progress updates

    Returns:
        str: Path to the resized image
    """
    # Validate inputs
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if width is None and height is None:
        raise ValueError("Either width or height must be specified")

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
    if width is not None and height is not None:
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
        # Include resize dimensions in filename
        if width is not None and height is not None:
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
