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
from .autocrop import autocrop
from .config import list_presets, load_presets, save_preset
from .filters import apply_filters, batch_apply_filters, list_filters
from .imageinfo import print_image_info
from .pdf2img import pdf_to_images
from .resize import resize_image
from .rmbg import remove_background, remove_background_command
from .workflow import (
    Workflow,
    delete_workflow,
    load_workflows,
    save_workflow,
    workflow_command,
)


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
    click.echo("ImageWand version 0.1.1")


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
def autocrop(input_path, output, mode, border_percent, margin):
    """Auto-crop scanned images"""
    try:
        # Import the function with an alias to avoid name collision
        from .autocrop import autocrop as autocrop_func

        result = autocrop_func(
            input_path, output, mode=mode, border_percent=border_percent, margin=margin
        )
        click.echo(f"Processed image saved to: {result}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


@cli.command()
@click.argument("image_path", type=click.Path(exists=True), required=False)
@click.option("--filters", "-f", help="Comma-separated list of filters to apply")
@click.option("--preset", "-p", help="Use a saved preset")
@click.option("--output", "-o", type=click.Path(), help="Output path")
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="Process all images in directory recursively",
)
@click.option("--save-preset", help="Save filters as a preset with the given name")
@click.option("--list-presets", is_flag=True, help="List available presets")
def filter(
    image_path,
    filters,
    preset,
    output,
    recursive,
    save_preset,
    list_presets,
):
    """Apply filters to images."""
    try:
        from .config import list_presets, load_presets, save_preset
        from .filters import apply_filters, list_filters

        # If --list-presets is specified, just list presets and return
        if list_presets:
            presets = list_presets()
            if not presets:
                click.echo("No presets found.")
                return

            click.echo("Available presets:")
            for name, value in presets.items():
                click.echo(f"  {name}: {value}")
            return

        # Require image_path for other operations
        if not image_path:
            click.echo("Error: image_path is required when not listing presets.")
            click.echo("Usage: imagewand filter [--list-presets] IMAGE_PATH [OPTIONS]")
            return 1

        # Load filters from preset if specified
        if preset:
            presets = load_presets()
            if preset not in presets:
                click.echo(f"Preset '{preset}' not found.")
                return 1
            filters = presets[preset]

        # Get the list of filters to apply
        if not filters:
            avail_filters = list_filters()
            click.echo("Available filters:")
            for filter_name in avail_filters:
                click.echo(f"  {filter_name}")
            return

        # Check if input is a directory
        if os.path.isdir(image_path):
            if not recursive:
                click.echo(
                    "Error: input path is a directory. Use --recursive to process all images."
                )
                return 1

            # Process all images in directory
            image_types = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
            if not output:
                output = os.path.join(image_path, "filtered")
            os.makedirs(output, exist_ok=True)

            files = []
            for pattern in image_types:
                files.extend(glob.glob(os.path.join(image_path, pattern)))

            if not files:
                click.echo("No image files found in directory.")
                return

            with tqdm(total=len(files), desc="Applying filters") as pbar:
                for file in files:
                    out_file = os.path.join(output, os.path.basename(file))
                    try:
                        apply_filters(file, filters.split(","), out_file)
                    except Exception as e:
                        click.echo(f"Error processing {file}: {str(e)}")
                    pbar.update(1)

            click.echo(f"Filtered images saved to: {output}")
        else:
            # Process single image
            if not output:
                output = image_path
                if "." in os.path.basename(output):
                    name, ext = os.path.splitext(output)
                    output = f"{name}_filtered{ext}"
                else:
                    output = f"{output}_filtered"

            apply_filters(image_path, filters.split(","), output)
            click.echo(f"Filtered image saved to: {output}")

        # Save preset if requested
        if save_preset:
            save_preset(save_preset, filters)
            click.echo(f"Saved preset '{save_preset}': {filters}")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


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
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file or directory (if processing multiple files)",
)
@click.option(
    "--batch",
    "-b",
    is_flag=True,
    help="Process all images in the input directory",
)
@click.option(
    "--alpha-matting",
    is_flag=True,
    help="Use alpha matting for improved edges",
)
@click.option(
    "--alpha-matting-foreground-threshold",
    type=int,
    default=240,
    help="Alpha matting foreground threshold (default: 240)",
)
@click.option(
    "--alpha-matting-background-threshold",
    type=int,
    default=10,
    help="Alpha matting background threshold (default: 10)",
)
@click.option(
    "--alpha-matting-erode-size",
    type=int,
    default=10,
    help="Alpha matting erode size (default: 10)",
)
@click.option(
    "--model",
    type=click.Choice(
        ["u2net", "u2netp", "u2net_human_seg", "silueta", "isnet-general-use"]
    ),
    default="u2net",
    help="Model to use for background removal (default: u2net)",
)
def rmbg(
    input_path,
    output,
    batch,
    alpha_matting,
    alpha_matting_foreground_threshold,
    alpha_matting_background_threshold,
    alpha_matting_erode_size,
    model,
):
    """Remove backgrounds from images"""
    try:
        # Reuse the existing implementation
        remove_background_command.callback(
            input_path,
            output,
            batch,
            alpha_matting,
            alpha_matting_foreground_threshold,
            alpha_matting_background_threshold,
            alpha_matting_erode_size,
            model,
        )
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


@cli.command()
@click.argument("input_path", type=click.Path(exists=True), required=False)
@click.option("--output", "-o", type=click.Path(), help="Output path")
@click.option("--workflow", "-w", help="Workflow to use")
@click.option(
    "--list", "-l", "list_workflows", is_flag=True, help="List available workflows"
)
@click.option("--delete", "-d", help="Delete a workflow")
def workflow(
    input_path,
    output,
    workflow,
    list_workflows,
    delete,
):
    """Execute a saved workflow on an image."""
    try:
        from .workflow import workflow_command

        workflow_command.callback(
            input_path=input_path,
            output=output,
            workflow=workflow,
            list_workflows=list_workflows,
            delete=delete,
        )
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


@cli.command()
@click.argument("name")
@click.option("--add-align", "-a", is_flag=True, help="Add align step")
@click.option(
    "--align-method",
    type=click.Choice(["auto", "hough", "contour", "center"]),
    default="auto",
    help="Alignment method",
)
@click.option(
    "--align-threshold", type=float, default=1.0, help="Alignment angle threshold"
)
@click.option("--add-autocrop", "-c", is_flag=True, help="Add autocrop step")
@click.option(
    "--crop-mode",
    type=click.Choice(["auto", "frame", "border"]),
    default="auto",
    help="Cropping mode",
)
@click.option("--crop-margin", type=int, default=-1, help="Crop margin")
@click.option("--crop-border", type=int, default=-1, help="Crop border percent")
@click.option("--add-resize", "-r", is_flag=True, help="Add resize step")
@click.option("--resize-width", type=int, help="Resize width")
@click.option("--resize-height", type=int, help="Resize height")
@click.option("--resize-percent", type=float, help="Resize percentage")
@click.option(
    "--resize-target-size", help="Resize to target file size (e.g., '5MB', '500KB')"
)
@click.option("--add-filter", "-f", help="Add filter step (comma-separated filters)")
@click.option("--add-rmbg", "-b", is_flag=True, help="Add background removal step")
@click.option(
    "--rmbg-model",
    type=click.Choice(
        ["u2net", "u2netp", "u2net_human_seg", "silueta", "isnet-general-use"]
    ),
    default="u2net",
    help="Background removal model",
)
@click.option("--rmbg-alpha-matting", is_flag=True, help="Use alpha matting")
@click.option(
    "--rmbg-foreground-threshold",
    type=int,
    default=240,
    help="Alpha matting foreground threshold",
)
@click.option(
    "--rmbg-background-threshold",
    type=int,
    default=10,
    help="Alpha matting background threshold",
)
@click.option(
    "--rmbg-erode-size", type=int, default=10, help="Alpha matting erode size"
)
def create_workflow(
    name,
    add_align,
    align_method,
    align_threshold,
    add_autocrop,
    crop_mode,
    crop_margin,
    crop_border,
    add_resize,
    resize_width,
    resize_height,
    resize_percent,
    resize_target_size,
    add_filter,
    add_rmbg,
    rmbg_model,
    rmbg_alpha_matting,
    rmbg_foreground_threshold,
    rmbg_background_threshold,
    rmbg_erode_size,
):
    """
    Create a new workflow by specifying steps.

    A workflow is a sequence of operations that are applied to an image in order.
    """
    try:
        from .workflow import Workflow, load_workflows, save_workflow

        # Check if workflow already exists
        workflows = load_workflows()
        if name in workflows:
            if not click.confirm(f"Workflow '{name}' already exists. Overwrite?"):
                return

        # Create new workflow
        workflow = Workflow(name)

        # Add selected steps in order
        if add_align:
            workflow.add_step(
                "align", {"method": align_method, "angle_threshold": align_threshold}
            )
            click.echo(
                f"Added align step (method={align_method}, threshold={align_threshold})"
            )

        if add_autocrop:
            workflow.add_step(
                "autocrop",
                {
                    "mode": crop_mode,
                    "margin": crop_margin,
                    "border_percent": crop_border,
                },
            )
            click.echo(f"Added autocrop step (mode={crop_mode})")

        if add_resize:
            workflow.add_step(
                "resize",
                {
                    "width": resize_width,
                    "height": resize_height,
                    "percent": resize_percent,
                    "target_size": resize_target_size,
                },
            )
            size_info = []
            if resize_width:
                size_info.append(f"width={resize_width}")
            if resize_height:
                size_info.append(f"height={resize_height}")
            if resize_percent:
                size_info.append(f"percent={resize_percent}")
            if resize_target_size:
                size_info.append(f"target_size={resize_target_size}")
            click.echo(f"Added resize step ({', '.join(size_info)})")

        if add_filter:
            workflow.add_step("filter", {"filters": add_filter})
            click.echo(f"Added filter step (filters='{add_filter}')")

        if add_rmbg:
            workflow.add_step(
                "rmbg",
                {
                    "model": rmbg_model,
                    "alpha_matting": rmbg_alpha_matting,
                    "alpha_matting_foreground_threshold": rmbg_foreground_threshold,
                    "alpha_matting_background_threshold": rmbg_background_threshold,
                    "alpha_matting_erode_size": rmbg_erode_size,
                },
            )
            click.echo(f"Added background removal step (model={rmbg_model})")

        # Check if any steps were added
        if not workflow.steps:
            click.echo("No steps added to workflow. Please specify at least one step.")
            return

        # Save workflow
        save_workflow(workflow)
        click.echo(f"Workflow '{name}' saved with {len(workflow.steps)} steps.")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


@cli.command()
@click.argument("input_path")
@click.option("-o", "--output", help="Output path")
@click.option("--width", "-w", type=int, help="Target width in pixels")
@click.option("--height", type=int, help="Target height in pixels")
@click.option(
    "--percent", "-p", type=float, help="Resize percentage (e.g., 50.0 for 50%)"
)
@click.option("--target-size", "-s", help="Target file size (e.g., '5MB', '500KB')")
def resize(input_path, output, width, height, percent, target_size):
    """Resize an image by dimensions, percentage, or target file size"""
    try:
        from .resize import resize_image

        # Validate inputs
        if not any([width, height, percent, target_size]):
            click.echo(
                "Error: Must specify either --width, --height, --percent, or --target-size"
            )
            return 1

        # Check for conflicting options
        resize_options = [width, height, percent, target_size]
        specified_options = [opt for opt in resize_options if opt is not None]

        if target_size and any([width, height, percent]):
            click.echo("Error: Cannot use --target-size with other resize options")
            return 1

        if percent and any([width, height]):
            click.echo("Error: Cannot use --percent with --width or --height")
            return 1

        result = resize_image(
            input_path,
            output_path=output,
            width=width,
            height=height,
            percent=percent,
            target_size=target_size,
        )
        click.echo(f"Resized image saved to: {result}")
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
    resize_parser.add_argument("-p", "--percent", help="Resize percentage", type=float)
    resize_parser.add_argument(
        "-s", "--target-size", help="Target file size (e.g., '5MB', '500KB')"
    )

    # Autocrop command
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
    filter_parser.add_argument("image_path", nargs="?", help="Path to the image file")
    filter_parser.add_argument(
        "-f", "--filters", help="Comma-separated list of filters to apply"
    )
    filter_parser.add_argument("-p", "--preset", help="Use a saved preset")
    filter_parser.add_argument("-o", "--output", help="Output path")
    filter_parser.add_argument(
        "-r", "--recursive", action="store_true", help="Process all images in directory"
    )
    filter_parser.add_argument(
        "--save-preset", help="Save filters as a preset with this name"
    )
    filter_parser.add_argument(
        "--list-presets", action="store_true", help="List available presets"
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

    # Add rmbg command
    rmbg_parser = subparsers.add_parser("rmbg", help="Remove image backgrounds")
    rmbg_parser.add_argument("input_path", help="Path to the image file or directory")
    rmbg_parser.add_argument("-o", "--output", help="Output path", default=None)
    rmbg_parser.add_argument(
        "-b", "--batch", action="store_true", help="Process all images in the directory"
    )
    rmbg_parser.add_argument(
        "--alpha-matting",
        action="store_true",
        help="Use alpha matting for improved edges",
    )
    rmbg_parser.add_argument(
        "--alpha-matting-foreground-threshold",
        type=int,
        default=240,
        help="Alpha matting foreground threshold",
    )
    rmbg_parser.add_argument(
        "--alpha-matting-background-threshold",
        type=int,
        default=10,
        help="Alpha matting background threshold",
    )
    rmbg_parser.add_argument(
        "--alpha-matting-erode-size",
        type=int,
        default=10,
        help="Alpha matting erode size",
    )
    rmbg_parser.add_argument(
        "--model",
        choices=["u2net", "u2netp", "u2net_human_seg", "silueta", "isnet-general-use"],
        default="u2net",
        help="Model to use for background removal",
    )

    # Add workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Execute a workflow")
    workflow_parser.add_argument("input_path", nargs="?", help="Path to input image")
    workflow_parser.add_argument("-o", "--output", help="Output path")
    workflow_parser.add_argument("-w", "--workflow", help="Workflow to use")
    workflow_parser.add_argument(
        "-l", "--list", action="store_true", help="List workflows"
    )
    workflow_parser.add_argument("-d", "--delete", help="Delete a workflow")

    # Create workflow command
    create_workflow_parser = subparsers.add_parser(
        "create-workflow", help="Create a workflow"
    )
    create_workflow_parser.add_argument("name", help="Workflow name")
    create_workflow_parser.add_argument(
        "-a", "--add-align", action="store_true", help="Add align step"
    )
    create_workflow_parser.add_argument(
        "--align-method",
        choices=["auto", "hough", "contour", "center"],
        default="auto",
        help="Alignment method",
    )
    create_workflow_parser.add_argument(
        "--align-threshold", type=float, default=1.0, help="Alignment angle threshold"
    )
    create_workflow_parser.add_argument(
        "-c", "--add-autocrop", action="store_true", help="Add autocrop step"
    )
    create_workflow_parser.add_argument(
        "--crop-mode",
        choices=["auto", "frame", "border"],
        default="auto",
        help="Cropping mode",
    )
    create_workflow_parser.add_argument(
        "--crop-margin", type=int, default=-1, help="Crop margin"
    )
    create_workflow_parser.add_argument(
        "--crop-border", type=int, default=-1, help="Crop border percent"
    )
    create_workflow_parser.add_argument(
        "-r", "--add-resize", action="store_true", help="Add resize step"
    )
    create_workflow_parser.add_argument("--resize-width", type=int, help="Resize width")
    create_workflow_parser.add_argument(
        "--resize-height", type=int, help="Resize height"
    )
    create_workflow_parser.add_argument(
        "--resize-percent", type=float, help="Resize percentage"
    )
    create_workflow_parser.add_argument(
        "--resize-target-size", help="Resize to target file size (e.g., '5MB', '500KB')"
    )
    create_workflow_parser.add_argument(
        "-f", "--add-filter", help="Add filter step (comma-separated filters)"
    )
    create_workflow_parser.add_argument(
        "-b", "--add-rmbg", action="store_true", help="Add background removal step"
    )
    create_workflow_parser.add_argument(
        "--rmbg-model",
        choices=["u2net", "u2netp", "u2net_human_seg", "silueta", "isnet-general-use"],
        default="u2net",
        help="Background removal model",
    )
    create_workflow_parser.add_argument(
        "--rmbg-alpha-matting", action="store_true", help="Use alpha matting"
    )
    create_workflow_parser.add_argument(
        "--rmbg-foreground-threshold",
        type=int,
        default=240,
        help="Alpha matting foreground threshold",
    )
    create_workflow_parser.add_argument(
        "--rmbg-background-threshold",
        type=int,
        default=10,
        help="Alpha matting background threshold",
    )
    create_workflow_parser.add_argument(
        "--rmbg-erode-size", type=int, default=10, help="Alpha matting erode size"
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
            # Validate inputs
            if not any([args.width, args.height, args.percent, args.target_size]):
                print(
                    "Error: Must specify either --width, --height, --percent, or --target-size"
                )
                return 1

            # Check for conflicting options
            if args.target_size and any([args.width, args.height, args.percent]):
                print("Error: Cannot use --target-size with other resize options")
                return 1

            if args.percent and any([args.width, args.height]):
                print("Error: Cannot use --percent with --width or --height")
                return 1

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
                    percent=args.percent,
                    target_size=args.target_size,
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
            try:
                from .config import list_presets, load_presets, save_preset
                from .filters import apply_filters, list_filters

                # If --list-presets is specified, just list presets and return
                if args.list_presets:
                    presets = list_presets()
                    if not presets:
                        print("No presets found.")
                    else:
                        print("Available presets:")
                        for name, value in presets.items():
                            print(f"  {name}: {value}")
                    print_execution_time(start_time)
                    return

                # For other operations, we need image_path
                if not args.image_path:
                    print("Error: image_path is required when not using --list-presets")
                    return 1

                # Process the original filter implementation
                if not args.filters and not args.preset:
                    raise click.UsageError(
                        "Either --filters or --preset must be specified"
                    )

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
                    if not args.recursive:
                        print(
                            "Error: input path is a directory. Use --recursive to process all images."
                        )
                        return 1

                    # Process all images in directory
                    image_types = [
                        "*.jpg",
                        "*.jpeg",
                        "*.png",
                        "*.JPG",
                        "*.JPEG",
                        "*.PNG",
                    ]
                    if not args.output:
                        output = os.path.join(args.image_path, "filtered")
                    else:
                        output = args.output
                    os.makedirs(output, exist_ok=True)

                    files = []
                    for pattern in image_types:
                        files.extend(glob.glob(os.path.join(args.image_path, pattern)))

                    if not files:
                        print("No image files found in directory.")
                        return 1

                    with tqdm(total=len(files), desc="Applying filters") as pbar:
                        for file in files:
                            out_file = os.path.join(output, os.path.basename(file))
                            try:
                                apply_filters(file, filters.split(","), out_file)
                            except Exception as e:
                                print(f"Error processing {file}: {str(e)}")
                            pbar.update(1)

                    print(f"Filtered images saved to: {output}")
                else:
                    # Process single image
                    if not args.output:
                        output = args.image_path
                        if "." in os.path.basename(output):
                            name, ext = os.path.splitext(output)
                            output = f"{name}_filtered{ext}"
                        else:
                            output = f"{output}_filtered"
                    else:
                        output = args.output

                    apply_filters(args.image_path, filters.split(","), output)
                    print(f"Filtered image saved to: {output}")

                print_execution_time(start_time)

            except Exception as e:
                print(f"Error: {str(e)}")
                return 1

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
                        progress_callback=progress_callback,
                    )

                    # Ensure progress bar reaches 100%
                    if last_progress < 100:
                        pbar.update(100 - last_progress)

                print(f"Aligned image saved to: {result}")
                print_execution_time(start_time)
            except Exception as e:
                print(f"Error: {str(e)}")
                return 1

        elif args.command == "rmbg":
            try:
                # Reuse the existing implementation
                remove_background_command.callback(
                    args.input_path,
                    args.output,
                    args.batch,
                    args.alpha_matting,
                    args.alpha_matting_foreground_threshold,
                    args.alpha_matting_background_threshold,
                    args.alpha_matting_erode_size,
                    args.model,
                )
            except Exception as e:
                print(f"Error: {str(e)}")
                return 1

        elif args.command == "workflow":
            try:
                from .workflow import (
                    Workflow,
                    delete_workflow,
                    load_workflows,
                    save_workflow,
                )

                # Execute the workflow
                if args.list:
                    workflows = load_workflows()
                    if not workflows:
                        print("No workflows found.")
                    else:
                        print("Available workflows:")
                        for name, wf in workflows.items():
                            steps_str = " → ".join(
                                step["operation"] for step in wf.steps
                            )
                            print(f"  {name}: {steps_str}")
                elif args.delete:
                    if delete_workflow(args.delete):
                        print(f"Workflow '{args.delete}' deleted.")
                    else:
                        print(f"Workflow '{args.delete}' not found.")
                elif args.workflow:
                    if not os.path.exists(args.input_path):
                        print(f"Error: Input path '{args.input_path}' does not exist.")
                        return 1

                    workflows = load_workflows()
                    if args.workflow not in workflows:
                        print(f"Workflow '{args.workflow}' not found.")
                        return 1

                    result = workflows[args.workflow].execute(
                        args.input_path, args.output
                    )
                    print(f"Workflow completed. Output saved to: {result}")
                else:
                    print(
                        "No workflow specified. Use --workflow to specify a workflow or --list to see available workflows."
                    )
            except Exception as e:
                print(f"Error: {str(e)}")
                return 1

        elif args.command == "create-workflow":
            try:
                from .workflow import Workflow, load_workflows, save_workflow

                # Check if workflow already exists
                workflows = load_workflows()
                if args.name in workflows:
                    answer = input(
                        f"Workflow '{args.name}' already exists. Overwrite? (y/n): "
                    )
                    if answer.lower() != "y":
                        return 0

                # Create new workflow
                workflow = Workflow(args.name)

                # Add selected steps
                if args.add_align:
                    workflow.add_step(
                        "align",
                        {
                            "method": args.align_method,
                            "angle_threshold": args.align_threshold,
                        },
                    )
                    print(
                        f"Added align step (method={args.align_method}, threshold={args.align_threshold})"
                    )

                if args.add_autocrop:
                    workflow.add_step(
                        "autocrop",
                        {
                            "mode": args.crop_mode,
                            "margin": args.crop_margin,
                            "border_percent": args.crop_border,
                        },
                    )
                    print(f"Added autocrop step (mode={args.crop_mode})")

                if args.add_resize:
                    workflow.add_step(
                        "resize",
                        {
                            "width": args.resize_width,
                            "height": args.resize_height,
                            "percent": args.resize_percent,
                            "target_size": args.resize_target_size,
                        },
                    )
                    size_info = []
                    if args.resize_width:
                        size_info.append(f"width={args.resize_width}")
                    if args.resize_height:
                        size_info.append(f"height={args.resize_height}")
                    if args.resize_percent:
                        size_info.append(f"percent={args.resize_percent}")
                    if args.resize_target_size:
                        size_info.append(f"target_size={args.resize_target_size}")
                    print(f"Added resize step ({', '.join(size_info)})")

                if args.add_filter:
                    workflow.add_step("filter", {"filters": args.add_filter})
                    print(f"Added filter step (filters='{args.add_filter}')")

                if args.add_rmbg:
                    workflow.add_step(
                        "rmbg",
                        {
                            "model": args.rmbg_model,
                            "alpha_matting": args.rmbg_alpha_matting,
                            "alpha_matting_foreground_threshold": args.rmbg_foreground_threshold,
                            "alpha_matting_background_threshold": args.rmbg_background_threshold,
                            "alpha_matting_erode_size": args.rmbg_erode_size,
                        },
                    )
                    print(f"Added background removal step (model={args.rmbg_model})")

                # Check if any steps were added
                if not workflow.steps:
                    print(
                        "No steps added to workflow. Please specify at least one step."
                    )
                    return 1

                # Save workflow
                save_workflow(workflow)
                print(f"Workflow '{args.name}' saved with {len(workflow.steps)} steps.")

            except Exception as e:
                print(f"Error: {str(e)}")
                return 1

        else:
            parser.print_help()
            return 1

    except Exception as e:
        print(f"\nError: {str(e)}")
        print_execution_time(start_time)
        return 1

    return 0


def add_merge_command(subparsers):
    """Add the merge command to the subparsers"""
    merge_parser = subparsers.add_parser("merge", help="Merge multiple images")
    merge_parser.add_argument(
        "input", nargs="+", help="Input images or directories of images"
    )
    merge_parser.add_argument("-o", "--output", help="Output path", default=None)
    merge_parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with visualizations"
    )


if __name__ == "__main__":
    sys.exit(main())
