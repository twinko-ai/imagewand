#!/usr/bin/env python3
import argparse
import sys
import os
from tqdm import tqdm
from .pdf2img import pdf_to_images
from .resize import resize_image

def main():
    parser = argparse.ArgumentParser(description="ImageWand - Image manipulation tools")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # PDF to images command
    pdf_parser = subparsers.add_parser("pdf2img", help="Convert PDF to images")
    pdf_parser.add_argument("pdf_path", help="Path to the PDF file")
    pdf_parser.add_argument("-o", "--output", help="Output directory", default=None)
    pdf_parser.add_argument("-f", "--format", help="Output format (png, jpg, etc.)", default="jpg")
    pdf_parser.add_argument("-d", "--dpi", help="DPI for conversion", type=int, default=200)
    
    # Resize image command
    resize_parser = subparsers.add_parser("resize", help="Resize images")
    resize_parser.add_argument("image_path", help="Path to the image file")
    resize_parser.add_argument("-o", "--output", help="Output path", default=None)
    resize_parser.add_argument("-w", "--width", help="New width", type=int)
    resize_parser.add_argument("--height", help="New height", type=int)
    
    args = parser.parse_args()
    
    if args.command == "pdf2img":
        # Create default output folder if not specified
        if args.output is None:
            pdf_basename = os.path.splitext(os.path.basename(args.pdf_path))[0]
            output_dir = os.path.join(os.path.dirname(args.pdf_path), f"{pdf_basename}_images")
        else:
            output_dir = args.output
            
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get total number of pages for progress bar
        import fitz
        doc = fitz.open(args.pdf_path)
        total_pages = len(doc)
        doc.close()
        
        # Convert PDF to images with progress bar
        with tqdm(total=total_pages, desc="Converting pages") as pbar:
            def progress_callback(page_num):
                pbar.update(1)
            
            image_files = pdf_to_images(
                args.pdf_path, 
                output_dir, 
                dpi=args.dpi, 
                format=args.format,
                progress_callback=progress_callback
            )
        
        print(f"Converted PDF to {len(image_files)} images in {output_dir}")
    
    elif args.command == "resize":
        # Create default output path if not specified
        if args.output is None:
            image_dir = os.path.dirname(args.image_path)
            image_basename = os.path.splitext(os.path.basename(args.image_path))[0]
            image_ext = os.path.splitext(args.image_path)[1]
            output_path = os.path.join(image_dir, f"{image_basename}_resized{image_ext}")
        else:
            output_path = args.output
            
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Resize the image with progress indication
        with tqdm(total=100, desc="Resizing") as pbar:
            last_progress = 0
            
            def progress_callback(percent):
                nonlocal last_progress
                increment = percent - last_progress
                if increment > 0:
                    pbar.update(increment)
                    last_progress = percent
            
            resize_image(
                args.image_path, 
                output_path, 
                width=args.width, 
                height=args.height, 
                progress_callback=progress_callback
            )
        
        print(f"Resized image saved to {output_path}")
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())