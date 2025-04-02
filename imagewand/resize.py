import os

from PIL import Image

MIN_WIDTH = 1500
MAX_WIDTH = 2500
MAX_SIZE_KB = 500
MAX_SIZE_BYTES = MAX_SIZE_KB * 1024


def resize_image(
    input_path: str, output_path: str, width: int = None, height: int = None
) -> str:
    """Resize an image while maintaining aspect ratio.

    Args:
        input_path: Path to input image
        output_path: Path to output image
        width: Target width in pixels
        height: Target height in pixels

    Returns:
        Path to output image
    """
    if width is None and height is None:
        raise ValueError("Either width or height must be specified")

    img = Image.open(input_path)
    if width and height:
        img = img.resize((width, height))
    elif width:
        ratio = width / img.size[0]
        height = int(img.size[1] * ratio)
        img = img.resize((width, height))
    else:
        ratio = height / img.size[1]
        width = int(img.size[0] * ratio)
        img = img.resize((width, height))

    img.save(output_path)
    return output_path
