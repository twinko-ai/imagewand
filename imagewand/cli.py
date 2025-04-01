#!/usr/bin/env python3
import argparse
import sys
import os
from tqdm import tqdm
from .pdf2img import pdf_to_images
from .resize import resize_image
from .autofix import autofix
import glob
from .filters import apply_filter, apply_filters, batch_apply_filters, list_filters
import cv2

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
    
    # Autofix image command
    autofix_parser = subparsers.add_parser("autofix", help="Automatically straighten and crop scanned images")
    autofix_parser.add_argument("image_path", help="Path to the image file or directory of images")
    autofix_parser.add_argument("-o", "--output", help="Output path or directory", default=None)
    autofix_parser.add_argument("-b", "--border", help="Percentage of border to keep around content", type=float, default=5)
    
    # Filter command
    filter_parser = subparsers.add_parser("filter", help="Apply filters to images")
    filter_parser.add_argument("image_path", help="Path to the image file or directory of images")
    filter_parser.add_argument("-f", "--filters", help="Comma-separated list of filters to apply", required=True)
    filter_parser.add_argument("-o", "--output", help="Output path or directory", default=None)
    filter_parser.add_argument("-p", "--params", help="JSON string of parameters for filters", default=None)
    filter_parser.add_argument("-r", "--recursive", help="Process directories recursively", action="store_true")
    
    # List filters command
    list_filters_parser = subparsers.add_parser("list-filters", help="List all available filters")
    
    # Add merge command
    merge_parser = subparsers.add_parser("merge", help="Merge multiple scanned images into one")
    merge_parser.add_argument("input_path", help="Path to image files or directory containing images")
    merge_parser.add_argument("-o", "--output", help="Output path (default: merged_<dirname>.jpg)", default=None)
    merge_parser.add_argument("--debug", help="Enable debug mode", action="store_true")
    merge_parser.add_argument("--pattern", help="Image file pattern when using directory (e.g. *.jpg)", default="*.jpg")
    
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
    
    elif args.command == "autofix":
        # Check if input is a directory
        if os.path.isdir(args.image_path):
            # Get all image files
            image_pattern = "*.jpg"
            image_paths = glob.glob(os.path.join(args.image_path, image_pattern))
            
            # Add other common image formats
            for ext in ["*.png", "*.jpeg", "*.bmp", "*.tiff", "*.gif"]:
                image_paths.extend(glob.glob(os.path.join(args.image_path, ext)))
            
            if not image_paths:
                print(f"No images found in {args.image_path}")
                return 1
            
            # Create output directory
            if args.output is None:
                output_dir = os.path.join(os.path.dirname(args.image_path), f"autofix_{os.path.basename(args.image_path)}")
            else:
                output_dir = args.output
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Process all images with progress bar
            print(f"Auto-fixing {len(image_paths)} images...")
            
            processed_files = []
            with tqdm(total=len(image_paths), desc="Processing images") as pbar:
                for image_path in image_paths:
                    # Create output path for this image
                    file_name = os.path.basename(image_path)
                    output_path = os.path.join(output_dir, file_name)
                    
                    # Process with progress callback
                    last_progress = 0
                    
                    def progress_callback(percent):
                        nonlocal last_progress
                        # We don't update the main progress bar here to avoid confusion
                        pass
                    
                    try:
                        result_path = autofix(
                            image_path,
                            output_path,
                            border_percent=args.border,
                            progress_callback=progress_callback
                        )
                        processed_files.append(result_path)
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
                    
                    pbar.update(1)
            
            print(f"Processed {len(processed_files)} images to {output_dir}")
        
        else:
            # Single image processing
            # Create default output path if not specified
            if args.output is None:
                image_dir = os.path.dirname(args.image_path)
                image_basename = os.path.splitext(os.path.basename(args.image_path))[0]
                image_ext = os.path.splitext(args.image_path)[1]
                output_path = os.path.join(image_dir, f"autofix_{image_basename}{image_ext}")
            else:
                output_path = args.output
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Autofix the image with progress indication
            with tqdm(total=100, desc="Auto-fixing") as pbar:
                last_progress = 0
                
                def progress_callback(percent):
                    nonlocal last_progress
                    increment = percent - last_progress
                    if increment > 0:
                        pbar.update(increment)
                        last_progress = percent
                
                output_path = autofix(
                    args.image_path,
                    output_path,
                    border_percent=args.border,
                    progress_callback=progress_callback
                )
            
            print(f"Auto-fixed image saved to {output_path}")
    
    elif args.command == "filter":
        # Parse filters
        filter_names = [f.strip() for f in args.filters.split(",")]
        
        # Parse parameters if provided
        params_list = None
        if args.params:
            import json
            try:
                params = json.loads(args.params)
                # If a single dict is provided, duplicate it for all filters
                if isinstance(params, dict):
                    params_list = [params] * len(filter_names)
                else:
                    params_list = params
            except json.JSONDecodeError:
                print("Error: Invalid JSON parameters")
                return 1
        
        # Check if input is a directory
        if os.path.isdir(args.image_path):
            # Get all image files
            image_pattern = "*.jpg" if not args.recursive else "**/*.jpg"
            image_paths = glob.glob(os.path.join(args.image_path, image_pattern), recursive=args.recursive)
            
            # Add other common image formats
            for ext in ["*.png", "*.jpeg", "*.bmp", "*.tiff", "*.gif"]:
                pattern = ext if not args.recursive else f"**/{ext}"
                image_paths.extend(glob.glob(os.path.join(args.image_path, pattern), recursive=args.recursive))
            
            if not image_paths:
                print(f"No images found in {args.image_path}")
                return 1
            
            # Create output directory
            output_dir = args.output if args.output else os.path.join(args.image_path, "filtered_images")
            os.makedirs(output_dir, exist_ok=True)
            
            # Apply filters to all images with progress bar
            print(f"Applying filters to {len(image_paths)} images...")
            with tqdm(total=100, desc="Processing images") as pbar:
                def progress_callback(percent):
                    pbar.update(percent - pbar.n)
                
                output_paths = batch_apply_filters(
                    image_paths,
                    filter_names,
                    output_dir,
                    params_list,
                    progress_callback
                )
            
            print(f"Filtered {len(output_paths)} images to {output_dir}")
            
        else:
            # Single image processing
            with tqdm(total=100, desc=f"Applying {', '.join(filter_names)}") as pbar:
                last_progress = 0
                
                def progress_callback(percent):
                    nonlocal last_progress
                    increment = percent - last_progress
                    if increment > 0:
                        pbar.update(increment)
                        last_progress = percent
                
                output_path = apply_filters(
                    args.image_path,
                    filter_names,
                    args.output,
                    params_list,
                    progress_callback
                )
            
            print(f"Filtered image saved to {output_path}")
    
    elif args.command == "list-filters":
        filters = list_filters()
        print("Available filters:")
        for f in sorted(filters):
            print(f"  - {f}")
    
    elif args.command == "merge":
        # Create default output path if not specified
        if args.output is None:
            if os.path.isdir(args.input_path):
                # For directory input, use directory name
                dirname = os.path.basename(os.path.normpath(args.input_path))
                output_path = os.path.join(os.path.dirname(args.input_path), f"merged_{dirname}.jpg")
            else:
                # For single file input, use file name
                input_dir = os.path.dirname(args.input_path)
                input_basename = os.path.splitext(os.path.basename(args.input_path))[0]
                output_path = os.path.join(input_dir, f"merged_{input_basename}.jpg")
        else:
            output_path = args.output
            
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get list of images to process
        image_paths = []
        if os.path.isdir(args.input_path):
            # Handle directory input
            patterns = ["*.jpg", "*.jpeg", "*.png", "*.tiff", "*.bmp"]
            for pattern in patterns:
                image_paths.extend(glob.glob(os.path.join(args.input_path, pattern)))
            
            if not image_paths:
                print(f"Error: No images found in directory {args.input_path}")
                return 1
                
            # Sort paths to ensure consistent ordering
            image_paths.sort()
        else:
            # Single file input
            if not os.path.exists(args.input_path):
                print(f"Error: Input path {args.input_path} does not exist")
                return 1
            image_paths = [args.input_path]
            
        # Load images with progress bar
        print(f"Loading {len(image_paths)} images...")
        images = []
        for path in tqdm(image_paths, desc="Loading images"):
            img = cv2.imread(path)
            if img is None:
                print(f"Error: Could not load image {path}")
                return 1
            images.append(img)
            
        if len(images) < 2:
            print("Error: At least two images are required for merging")
            return 1
            
        # Initialize merger
        from .automerge import AutoMerge
        merger = AutoMerge(debug=args.debug)
        
        # Perform merge with progress indication
        print("Merging images...")
        try:
            result = merger.merge_images(images)
        except Exception as e:
            print(f"Error during merge: {str(e)}")
            return 1
        
        # Save result
        print(f"Saving merged result to {output_path}")
        cv2.imwrite(output_path, result)
        print("Done!")
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())