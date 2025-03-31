import os
from PIL import Image

MIN_WIDTH = 1500
MAX_WIDTH = 2500
MAX_SIZE_KB = 500
MAX_SIZE_BYTES = MAX_SIZE_KB * 1024

def resize_image(image_path, output_folder, index):
    """Resize and compress an image."""
    with Image.open(image_path) as img:
        if img.mode in ("P", "RGBA"):
            img = img.convert("RGB")

        width, height = img.size
        new_width = max(min(width, MAX_WIDTH), MIN_WIDTH)
        new_height = int((new_width / width) * height)
        img = img.resize((new_width, new_height), Image.LANCZOS)

        os.makedirs(output_folder, exist_ok=True)
        output_filename = os.path.join(output_folder, f"resized_{index:03d}.jpg")

        quality = 95
        while True:
            img.save(output_filename, "JPEG", quality=quality)
            if os.path.getsize(output_filename) <= MAX_SIZE_BYTES or quality <= 20:
                break
            quality -= 5

        return output_filename
