#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import time
from datetime import datetime

import click
import cv2
from tqdm import tqdm

from .align import align_image
from .autocrop import autocrop, crop_framed_photo, crop_with_content_detection
from .config import list_presets, load_presets, save_preset
from .crop import crop_image
from .filters import apply_filter, apply_filters, batch_apply_filters, list_filters
from .imageinfo import print_image_info
from .pdf2img import pdf_to_images
from .resize import resize_image


def print_execution_time(start_time):
    """Print execution time in a human readable format"""
    end_time = time.time()
    duration = end_time - start_time
    if duration < 60:
        time_str = f"{duration:.2f} seconds"
    else:
        minutes = int(duration // 60)
        seconds = duration % 60
        time_str = f"{minutes} minutes {seconds:.2f} seconds"

    print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {time_str}")


@click.group()
def cli():
    """ImageWand - Image processing utilities"""
    pass


@cli.command()
def version():
    """Show version information"""
    click.echo("ImageWand version 0.1.0")


@cli.command()
@click.argument("input_path")
@click.option("-o", "--output", help="Output path")
@click.option(
    "-m", "--mode", default="auto", help="Cropping mode: auto, frame, or border"
)
@click.option(
    "-b",
    "--border-percent",
    default=-1,
    type=int,
    help="Border percentage for border mode",
)
@click.option("--margin", default=-1, type=int, help="Margin in pixels for frame mode")
def autocrop_cmd(input_path, output, mode, border_percent, margin):
    """Auto-crop scanned images"""
    from .autocrop import autocrop

    try:
        result = autocrop(
            input_path, output, mode=mode, border_percent=border_percent, margin=margin
        )
        click.echo(f"Processed image saved to: {result}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


@cli.command()
@click.argument("input_path")
@click.option("-f", "--filters", help="Comma-separated list of filters to apply")
@click.option("-o", "--output", help="Output path")
def filter(input_path, filters, output):
    """Apply filters to images"""
    from .filters import apply_filters

    filters_list = filters.split(",") if filters else []
    result = apply_filters([input_path], filters_list, output)
    click.echo(f"Filtered image saved to: {result}")


@cli.command()
@click.argument("input_path")
@click.option("-o", "--output", help="Output path")
@click.option(
    "-m",
    "--method",
    default="auto",
    type=click.Choice(["auto", "hough", "contour", "center"]),
    help="Alignment method to use",
)
@click.option(
    "-a",
    "--angle-threshold",
    default=1.0,
    type=float,
    help="Minimum angle to correct (degrees)",
)
def align(input_path, output, method, angle_threshold):
    """Automatically align tilted images to be horizontal/vertical"""
    try:
        result = align_image(
            input_path,
            output_path=output,
            method=method,
            angle_threshold=angle_threshold,
        )
        click.echo(f"Aligned image saved to: {result}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


@cli.command()
@click.argument("input_path")
@click.option("-o", "--output", help="Output path")
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["frame", "border", "auto"]),
    default="frame",
    help="Cropping mode: frame, border, or auto",
)
@click.option(
    "-b",
    "--border-percent",
    type=int,
    default=-1,
    help="Border percentage for border mode",
)
@click.option(
    "--margin",
    type=int,
    default=-1,
    help="Margin in pixels for frame mode",
)
@click.option(
    "-t",
    "--threshold",
    type=int,
    default=30,
    help="Brightness threshold for content detection",
)
def crop(input_path, output, mode, border_percent, margin, threshold):
    """Crop images using frame detection or border removal"""
    try:
        start_time = time.time()

        # Handle directory input
        if os.path.isdir(input_path):
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                image_files.extend(glob.glob(os.path.join(input_path, ext)))

            if not image_files:
                click.echo(f"No image files found in {input_path}")
                return 1

            click.echo(f"Processing {len(image_files)} images...")

            # Process each image with progress bar
            with tqdm(total=len(image_files)) as pbar:
                for i, img_path in enumerate(image_files):
                    try:
                        result = crop_image(
                            img_path,
                            None,  # Auto-generate output path for each image
                            mode=mode,
                            margin=margin,
                            border_percent=border_percent,
                            threshold=threshold,
                        )
                        # Update progress
                        pbar.update(1)
                        pbar.set_description(f"Processed: {os.path.basename(img_path)}")
                    except Exception as e:
                        click.echo(f"Error processing {img_path}: {str(e)}", err=True)

            click.echo(f"Finished processing {len(image_files)} images")
        else:
            # Process single image
            result = crop_image(
                input_path,
                output,
                mode=mode,
                margin=margin,
                border_percent=border_percent,
                threshold=threshold,
            )
            click.echo(f"Cropped image saved to: {result}")

        print_execution_time(start_time)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="ImageWand - Image manipulation tools")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # PDF to images command
    pdf_parser = subparsers.add_parser("pdf2img", help="Convert PDF to images")
    pdf_parser.add_argument("pdf_path", help="Path to the PDF file")
    pdf_parser.add_argument("-o", "--output", help="Output directory", default=None)
    pdf_parser.add_argument(
        "-f", "--format", help="Output format (png, jpg, etc.)", default="jpg"
    )
    pdf_parser.add_argument(
        "-d", "--dpi", help="DPI for conversion", type=int, default=200
    )

    # Resize image command
    resize_parser = subparsers.add_parser("resize", help="Resize images")
    resize_parser.add_argument("image_path", help="Path to the image file")
    resize_parser.add_argument("-o", "--output", help="Output path", default=None)
    resize_parser.add_argument("-w", "--width", help="New width", type=int)
    resize_parser.add_argument("--height", help="New height", type=int)

    # Autocrop command (renamed from autofix)
    autocrop_parser = subparsers.add_parser("autocrop", help="Auto-crop scanned images")
    autocrop_parser.add_argument("input_path", help="Path to the image file")
    autocrop_parser.add_argument("-o", "--output", help="Output path", default=None)
    autocrop_parser.add_argument(
        "-m",
        "--mode",
        choices=["auto", "frame", "border"],
        default="auto",
        help="Cropping mode: auto, frame, or border",
    )
    autocrop_parser.add_argument(
        "-b",
        "--border-percent",
        type=int,
        default=-1,
        help="Border percentage for border mode",
    )
    autocrop_parser.add_argument(
        "--margin", type=int, default=-1, help="Margin in pixels for frame mode"
    )

    # Filter command
    filter_parser = subparsers.add_parser("filter", help="Apply filters to images")
    filter_parser.add_argument(
        "image_path", help="Path to the image file or directory of images"
    )
    filter_parser.add_argument(
        "-f", "--filters", help="Comma-separated list of filters to apply"
    )
    filter_parser.add_argument("-p", "--preset", help="Use a saved filter preset")
    filter_parser.add_argument(
        "-o", "--output", help="Output path or directory", default=None
    )
    filter_parser.add_argument(
        "-r", "--recursive", help="Process directories recursively", action="store_true"
    )
    filter_parser.add_argument(
        "--save-preset", help="Save the filter string as a preset"
    )

    # List filters command
    list_filters_parser = subparsers.add_parser(
        "list-filters", help="List all available filters"
    )

    # Add merge command
    add_merge_command(subparsers)

    # Info command
    info_parser = subparsers.add_parser("info", help="Display image information")
    info_parser.add_argument("image_path", help="Path to the image file")
    info_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed information including EXIF data",
    )

    # Align command
    align_parser = subparsers.add_parser(
        "align", help="Automatically align tilted images to be horizontal/vertical"
    )
    align_parser.add_argument("image_path", help="Path to the image file")
    align_parser.add_argument("-o", "--output", help="Output path", default=None)
    align_parser.add_argument(
        "--method",
        choices=["auto", "hough", "contour", "center"],
        default="auto",
        help="Alignment method to use",
    )
    align_parser.add_argument(
        "--angle-threshold",
        type=float,
        default=1.0,
        help="Minimum angle to correct (degrees)",
    )

    # Crop command
    crop_parser = subparsers.add_parser(
        "crop", help="Crop images using frame detection or border removal"
    )
    crop_parser.add_argument("image_path", help="Path to the image file")
    crop_parser.add_argument("-o", "--output", help="Output path", default=None)
    crop_parser.add_argument(
        "-m",
        "--mode",
        choices=["frame", "border", "auto"],
        default="frame",
        help="Cropping mode: frame, border, or auto",
    )
    crop_parser.add_argument(
        "-b",
        "--border-percent",
        type=int,
        default=-1,
        help="Border percentage for border mode",
    )
    crop_parser.add_argument(
        "--margin", type=int, default=-1, help="Margin in pixels for frame mode"
    )
    crop_parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=30,
        help="Brightness threshold for content detection",
    )

    args = parser.parse_args()

    print(f"Starting operation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        if args.command == "pdf2img":
            # Create default output folder if not specified
            if args.output is None:
                pdf_basename = os.path.splitext(os.path.basename(args.pdf_path))[0]
                output_dir = os.path.join(
                    os.path.dirname(args.pdf_path), f"{pdf_basename}_images"
                )
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
                    progress_callback=progress_callback,
                )

            print(f"Converted PDF to {len(image_files)} images in {output_dir}")
            print_execution_time(start_time)

        elif args.command == "resize":
            # Let the resize_image function handle the default output path
            if args.output is None:
                output_path = None
            else:
                output_path = args.output

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path) if output_path else None
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
                    progress_callback=progress_callback,
                )

            print(f"Resized image saved to {output_path}")
            print_execution_time(start_time)

        elif args.command == "autocrop":
            try:
                # Create progress bar
                with tqdm(total=100, desc="Auto-cropping image") as pbar:
                    last_progress = 0

                    def progress_callback(percent):
                        nonlocal last_progress
                        increment = percent - last_progress
                        if increment > 0:
                            pbar.update(increment)
                            last_progress = percent

                    result = autocrop(
                        args.input_path,
                        args.output,
                        mode=args.mode,
                        margin=args.margin,
                        border_percent=args.border_percent,
                    )

                    # Ensure progress bar reaches 100%
                    if last_progress < 100:
                        pbar.update(100 - last_progress)

                print(f"Auto-cropped image saved to: {result}")
                print_execution_time(start_time)
            except Exception as e:
                print(f"Error: {str(e)}")
                sys.exit(1)

        elif args.command == "filter":
            if not args.filters and not args.preset:
                raise click.UsageError("Either --filters or --preset must be specified")

            if args.filters and args.preset:
                raise click.UsageError(
                    "Cannot use both --filters and --preset together"
                )

            # Initialize filters string
            filters = None

            # Get filters either from direct input or preset
            if args.preset:
                presets = load_presets()
                if args.preset not in presets:
                    available = "\n".join(
                        f"  - {name}: {value}" for name, value in presets.items()
                    )
                    raise click.UsageError(
                        f"Preset '{args.preset}' not found. Available presets:\n{available}"
                    )
                filters = presets[args.preset]
            elif args.filters:
                filters = args.filters

            # Save preset if requested
            if args.save_preset and filters:
                save_preset(args.save_preset, filters)
                print(f"Saved filter preset '{args.save_preset}': {filters}")

            # Check if input is a directory
            if os.path.isdir(args.image_path):
                # Get all image files
                image_pattern = "*.jpg" if not args.recursive else "**/*.jpg"
                image_paths = glob.glob(
                    os.path.join(args.image_path, image_pattern),
                    recursive=args.recursive,
                )

                # Add other common image formats
                for ext in ["*.png", "*.jpeg", "*.bmp", "*.tiff", "*.gif"]:
                    pattern = ext if not args.recursive else f"**/{ext}"
                    image_paths.extend(
                        glob.glob(
                            os.path.join(args.image_path, pattern),
                            recursive=args.recursive,
                        )
                    )

                if not image_paths:
                    print(f"No images found in {args.image_path}")
                    return 1

                # Create output directory
                output_dir = (
                    args.output
                    if args.output
                    else os.path.join(args.image_path, "filtered_images")
                )
                os.makedirs(output_dir, exist_ok=True)

                # Apply filters to all images with progress bar
                print(f"Applying filters to {len(image_paths)} images...")
                with tqdm(total=100, desc="Processing images") as pbar:

                    def progress_callback(percent):
                        pbar.update(percent - pbar.n)

                    output_paths = batch_apply_filters(
                        image_paths, filters.split(","), output_dir, progress_callback
                    )

                print(f"Filtered {len(output_paths)} images to {output_dir}")
                print_execution_time(start_time)

            else:
                # Single image processing
                with tqdm(
                    total=100, desc=f"Applying {', '.join(filters.split(','))}"
                ) as pbar:
                    last_progress = 0

                    def progress_callback(percent):
                        nonlocal last_progress
                        increment = percent - last_progress
                        if increment > 0:
                            pbar.update(increment)
                            last_progress = percent

                    output_path = apply_filters(
                        args.image_path,
                        filters.split(","),
                        args.output,
                        progress_callback,
                    )

                print(f"Filtered image saved to {output_path}")
                print_execution_time(start_time)

        elif args.command == "list-filters":
            filters = list_filters()
            print("Available filters:")
            for f in sorted(filters):
                print(f"  - {f}")
            print_execution_time(start_time)

        elif args.command == "merge":
            # Create default output path if not specified
            if args.output is None:
                if os.path.isdir(args.input[0]) or "*" in args.input[0]:
                    # For directory input, use directory name
                    dirname = os.path.basename(os.path.normpath(args.input[0]))
                    output_path = os.path.join(
                        os.path.dirname(args.input[0]), f"merged_{dirname}.jpg"
                    )
                else:
                    # For single file input, use file name
                    input_dir = os.path.dirname(args.input[0])
                    input_basename = os.path.splitext(os.path.basename(args.input[0]))[
                        0
                    ]
                    output_path = os.path.join(
                        input_dir, f"merged_{input_basename}.jpg"
                    )
            else:
                output_path = args.output

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Get list of images to process
            image_paths = []
            for path in args.input:
                if os.path.isdir(path) or "*" in path:
                    # Handle directory input
                    patterns = ["*.jpg", "*.jpeg", "*.png", "*.tiff", "*.bmp"]
                    for pattern in patterns:
                        image_paths.extend(glob.glob(os.path.join(path, pattern)))
                else:
                    # Single file input
                    if not os.path.exists(path):
                        print(f"Error: Input path {path} does not exist")
                        return 1
                    image_paths.append(path)

            if len(image_paths) < 2:
                print("Error: At least two images are required for merging")
                return 1

            # Load images with progress bar
            print(f"Loading {len(image_paths)} images...")
            images = []
            for path in tqdm(image_paths, desc="Loading images"):
                img = cv2.imread(path)
                if img is None:
                    print(f"Error: Could not load image {path}")
                    return 1
                images.append(img)

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
            print_execution_time(start_time)

        elif args.command == "info":
            print_image_info(args.image_path, args.verbose)
            print_execution_time(start_time)

        elif args.command == "align":
            try:
                # Create progress bar
                with tqdm(total=100, desc="Aligning image") as pbar:
                    last_progress = 0

                    def progress_callback(percent):
                        nonlocal last_progress
                        increment = percent - last_progress
                        if increment > 0:
                            pbar.update(increment)
                            last_progress = percent

                    result = align_image(
                        args.image_path,
                        output_path=args.output,
                        method=args.method,
                        angle_threshold=args.angle_threshold,
                    )

                    # Ensure progress bar reaches 100%
                    if last_progress < 100:
                        pbar.update(100 - last_progress)

                print(f"Aligned image saved to: {result}")
                print_execution_time(start_time)
            except Exception as e:
                print(f"Error: {str(e)}")
                sys.exit(1)

        elif args.command == "crop":
            try:
                # Create progress bar
                with tqdm(total=100, desc="Cropping image") as pbar:
                    last_progress = 0

                    def progress_callback(percent):
                        nonlocal last_progress
                        increment = percent - last_progress
                        if increment > 0:
                            pbar.update(increment)
                            last_progress = percent

                    result = crop_image(
                        args.image_path,
                        args.output,
                        mode=args.mode,
                        margin=args.margin,
                        border_percent=args.border_percent,
                        threshold=args.threshold,
                    )

                    # Ensure progress bar reaches 100%
                    if last_progress < 100:
                        pbar.update(100 - last_progress)

                print(f"Cropped image saved to: {result}")
                print_execution_time(start_time)
            except Exception as e:
                print(f"Error: {str(e)}")
                sys.exit(1)

        else:
            parser.print_help()
            return 1

    except Exception as e:
        print(f"\nError: {str(e)}")
        print_execution_time(start_time)
        return 1

    return 0


@click.command()
def list_presets_cmd():
    """List available filter presets"""
    presets = list_presets()
    if not presets:
        print("No saved presets found.")
        return

    print("Available presets:")
    for name, filters in presets.items():
        print(f"  - {name}: {filters}")


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-f", "--filters", help="Comma-separated list of filters to apply")
@click.option("-p", "--preset", help="Use a saved filter preset")
@click.option("-o", "--output", help="Output path")
@click.option("--save-preset", help="Save the filter string as a preset")
def filter_cmd(input_path, filters, preset, output, save_preset):
    """Apply filters to images"""
    start_time = time.time()
    print(
        f"Starting filter operation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    try:
        filter_string = None

        # Get filters either from direct input or preset
        if preset:
            presets = load_presets()
            if preset not in presets:
                available = "\n".join(
                    f"  - {name}: {value}" for name, value in presets.items()
                )
                raise click.UsageError(
                    f"Preset '{preset}' not found. Available presets:\n{available}"
                )
            filter_string = presets[preset]
        elif filters:
            filter_string = filters
        else:
            raise click.UsageError("Either --filters or --preset must be specified")

        # Save preset if requested
        if save_preset:
            if not filter_string:
                raise click.UsageError("No filters specified to save as preset")
            save_preset(save_preset, filter_string)
            print(f"Saved filter preset '{save_preset}': {filter_string}")

        # Apply filters
        result = apply_filters(input_path, filter_string.split(","), output)
        print(f"Filtered image saved to: {result}")

        # Print execution time
        print_execution_time(start_time)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print_execution_time(start_time)
        raise


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", help="Output path")
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["auto", "frame", "border"]),
    default="auto",
    help="Cropping mode",
)
@click.option(
    "-t",
    "--threshold",
    type=int,
    default=30,
    help="Brightness threshold for content detection",
)
def autocrop_cmd(input_path, output, mode, threshold):
    """Auto-crop image content"""
    try:
        result = autocrop(input_path, output, mode, threshold)
        print(f"Auto-cropped image saved to: {result}")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


def add_merge_command(subparsers):
    """Add merge command to the parser"""
    parser = subparsers.add_parser(
        "merge", help="Merge multiple images into a panorama"
    )
    parser.add_argument(
        "input",
        nargs="+",  # Accept multiple arguments
        help="Input directory or image files",
    )
    parser.add_argument("-o", "--output", help="Output file path", default=None)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.set_defaults(func=merge_command)


def merge_command(args):
    """Handle merge command"""
    from imagewand.automerge import automerge

    # If there's only one input and it's a directory or glob pattern, use it directly
    if len(args.input) == 1 and (os.path.isdir(args.input[0]) or "*" in args.input[0]):
        input_path = args.input[0]
    else:
        # Otherwise, treat the inputs as a list of image files
        input_path = args.input

    try:
        result = automerge(input_path, args.output, debug=args.debug)
        print(f"Merged image saved to: {result}")
    except Exception as e:
        print(f"Error merging images: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
