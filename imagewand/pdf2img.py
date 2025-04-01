import os
import fitz  # PyMuPDF


def pdf_to_images(pdf_path, output_dir, dpi=200, format="jpg", progress_callback=None):
    """
    Convert a PDF file to a series of images.

    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory to save the images
        dpi (int): DPI for the output images
        format (str): Output format (jpg, png, etc.)
        progress_callback (callable): Optional callback for progress updates

    Returns:
        list: List of paths to the generated images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the PDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    image_files = []

    # Calculate zoom factor based on DPI (72 is the base DPI for PDF)
    zoom = dpi / 72

    # Process each page
    for page_num in range(total_pages):
        # Get the page
        page = doc.load_page(page_num)

        # Convert page to an image
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        # Create output filename
        output_file = os.path.join(output_dir, f"page_{page_num + 1}.{format}")

        # Save the image
        pix.save(output_file)
        image_files.append(output_file)

        # Call progress callback if provided
        if progress_callback:
            progress_callback(page_num + 1)

    doc.close()
    return image_files
