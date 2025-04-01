import os
from PIL import Image

MIN_WIDTH = 1500
MAX_WIDTH = 2500
MAX_SIZE_KB = 500
MAX_SIZE_BYTES = MAX_SIZE_KB * 1024


def resize_image(
    input_path, output_path, width=None, height=None, progress_callback=None
):
    """
    Resize an image while maintaining aspect ratio.

    Args:
        input_path (str): Path to the input image
        output_path (str): Path to save the resized image
        width (int): Target width (if None, will be calculated from height)
        height (int): Target height (if None, will be calculated from width)
        progress_callback (callable): Optional callback function for progress updates

    Returns:
        str: Path to the resized image
    """
    # Open the image
    if progress_callback:
        progress_callback(10)  # 10% - Image opened

    img = Image.open(input_path)

    # Get original dimensions
    orig_width, orig_height = img.size

    # Calculate new dimensions
    if width and height:
        new_width, new_height = width, height
    elif width:
        ratio = width / orig_width
        new_width = width
        new_height = int(orig_height * ratio)
    elif height:
        ratio = height / orig_height
        new_height = height
        new_width = int(orig_width * ratio)
    else:
        new_width, new_height = orig_width, orig_height

    if progress_callback:
        progress_callback(50)  # 50% - Dimensions calculated

    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    if progress_callback:
        progress_callback(80)  # 80% - Image resized

    # Save the resized image
    resized_img.save(output_path)

    if progress_callback:
        progress_callback(100)  # 100% - Image saved

    return output_path
