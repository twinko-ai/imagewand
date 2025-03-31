import os
from pdf2image import convert_from_path

def pdf_to_images(pdf_path, output_folder):
    """Convert PDF pages to images."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File '{pdf_path}' not found.")

    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path)

    saved_files = []
    for i, image in enumerate(images, start=1):
        output_filename = os.path.join(output_folder, f"page_{i:03d}.jpg")
        image.save(output_filename, "JPEG", quality=95)
        saved_files.append(output_filename)

    return saved_files
