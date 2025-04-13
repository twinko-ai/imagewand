"""
Background removal module for ImageWand using rembg.

This module provides functions to remove backgrounds from images using the
rembg library, which implements U2Net for salient object detection.
"""

import os
from typing import List, Optional

import click
import rembg
from PIL import Image
from tqdm import tqdm

from imagewand.config import (
    list_presets,
    load_rmbg_preset,
    save_rmbg_preset,
)


def remove_background(
    image_path: str,
    output_path: Optional[str] = None,
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    model_name: str = "u2net",
) -> str:
    """
    Remove the background from an image.

    Args:
        image_path: Path to input image
        output_path: Path to save the processed image (optional)
        alpha_matting: Whether to use alpha matting for improved edges
        alpha_matting_foreground_threshold: Alpha matting foreground threshold
        alpha_matting_background_threshold: Alpha matting background threshold
        alpha_matting_erode_size: Alpha matting erode size
        model_name: Model to use (u2net, u2netp, u2net_human_seg, etc.)

    Returns:
        Path to the processed image
    """
    # Create output path if not specified
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        # Build suffix based on parameters
        suffix = "_nobg"

        # Add model name if not default
        if model_name != "u2net":
            suffix += f"_{model_name}"

        # Add alpha matting info if enabled
        if alpha_matting:
            suffix += "_am"
            # Add non-default alpha matting parameters if used
            if (
                alpha_matting_foreground_threshold != 240
                or alpha_matting_background_threshold != 10
                or alpha_matting_erode_size != 10
            ):
                suffix += f"_f{alpha_matting_foreground_threshold}"
                suffix += f"_b{alpha_matting_background_threshold}"
                suffix += f"_e{alpha_matting_erode_size}"

        output_path = f"{base}{suffix}{ext}"

    # Ensure output extension supports transparency
    output_format = os.path.splitext(output_path)[1].lower()
    if output_format not in [".png", ".webp"]:
        base = os.path.splitext(output_path)[0]
        output_path = f"{base}.png"
        click.echo(
            f"Warning: Changed output format to PNG to support transparency: {output_path}"
        )

    # Read input image
    input_img = Image.open(image_path)

    # Apply background removal
    # Create session for better performance with multiple images
    session = rembg.new_session(model_name)

    output_img = rembg.remove(
        input_img,
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
        alpha_matting_background_threshold=alpha_matting_background_threshold,
        alpha_matting_erode_size=alpha_matting_erode_size,
    )

    # Preserve original image metadata
    metadata = input_img.info

    # Save result
    output_img.save(output_path, **metadata)

    return output_path


def batch_remove_background(
    image_paths: List[str],
    output_dir: Optional[str] = None,
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    model_name: str = "u2net",
    show_progress: bool = True,
) -> List[str]:
    """
    Remove backgrounds from multiple images.

    Args:
        image_paths: List of input image paths
        output_dir: Directory to save processed images (optional)
        alpha_matting: Whether to use alpha matting for improved edges
        alpha_matting_foreground_threshold: Alpha matting foreground threshold
        alpha_matting_background_threshold: Alpha matting background threshold
        alpha_matting_erode_size: Alpha matting erode size
        model_name: Model to use (u2net, u2netp, u2net_human_seg, etc.)
        show_progress: Whether to show progress bar

    Returns:
        List of paths to processed images
    """
    output_paths = []

    # Create output directory if specified and doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create a single session for better performance
    session = rembg.new_session(model_name)

    for image_path in tqdm(
        image_paths, desc="Removing backgrounds", disable=not show_progress
    ):
        # Build suffix based on parameters
        suffix = "_nobg"

        # Add model name if not default
        if model_name != "u2net":
            suffix += f"_{model_name}"

        # Add alpha matting info if enabled
        if alpha_matting:
            suffix += "_am"
            # Add non-default alpha matting parameters if used
            if (
                alpha_matting_foreground_threshold != 240
                or alpha_matting_background_threshold != 10
                or alpha_matting_erode_size != 10
            ):
                suffix += f"_f{alpha_matting_foreground_threshold}"
                suffix += f"_b{alpha_matting_background_threshold}"
                suffix += f"_e{alpha_matting_erode_size}"

        if output_dir:
            filename = os.path.basename(image_path)
            base, ext = os.path.splitext(filename)
            out_path = os.path.join(output_dir, f"{base}{suffix}.png")
        else:
            base, ext = os.path.splitext(image_path)
            out_path = f"{base}{suffix}.png"

        # Read input image
        input_img = Image.open(image_path)

        # Apply background removal
        output_img = rembg.remove(
            input_img,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
        )

        # Preserve original image metadata
        metadata = input_img.info

        # Save result
        output_img.save(out_path, **metadata)
        output_paths.append(out_path)

    return output_paths


@click.command()
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
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress progress information and warnings",
)
@click.option(
    "--preset",
    "-p",
    help="Use a saved preset for background removal",
)
@click.option(
    "--save-preset",
    help="Save current parameters as a preset with the given name",
)
@click.option(
    "--list-presets",
    is_flag=True,
    help="List all available background removal presets",
)
def remove_background_command(
    input_path: str,
    output: Optional[str] = None,
    batch: bool = False,
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    model: str = "u2net",
    quiet: bool = False,
    preset: Optional[str] = None,
    save_preset: Optional[str] = None,
    list_presets: bool = False,
):
    """
    Remove backgrounds from images.

    This command removes the background from images using the rembg library,
    which implements U2Net, a state-of-the-art deep learning model for
    salient object detection.

    INPUT_PATH can be an image file or a directory when used with --batch.
    """
    show_progress = not quiet

    # List available presets if requested
    if list_presets:
        presets = list_presets("rmbg_presets")
        if not presets:
            click.echo("No background removal presets found.")
        else:
            click.echo("Available background removal presets:")
            for name, value in presets.items():
                click.echo(f"  {name}: {value}")
        return

    # Load preset if specified
    if preset:
        preset_dict = load_rmbg_preset(preset)
        if not preset_dict:
            raise click.BadParameter(f"Preset '{preset}' not found")

        # Update parameters from preset
        model = preset_dict.get("model", model)
        alpha_matting = preset_dict.get("alpha_matting", alpha_matting)
        if alpha_matting:
            alpha_matting_foreground_threshold = preset_dict.get(
                "foreground_threshold", alpha_matting_foreground_threshold
            )
            alpha_matting_background_threshold = preset_dict.get(
                "background_threshold", alpha_matting_background_threshold
            )
            alpha_matting_erode_size = preset_dict.get(
                "erode_size", alpha_matting_erode_size
            )

    # Save preset if requested
    if save_preset:
        save_rmbg_preset(
            save_preset,
            model,
            alpha_matting,
            alpha_matting_foreground_threshold,
            alpha_matting_background_threshold,
            alpha_matting_erode_size,
        )
        click.echo(f"Preset '{save_preset}' saved successfully")

    if batch:
        # Process directory
        input_path = os.path.abspath(input_path)
        if not os.path.isdir(input_path):
            raise click.BadParameter(
                f"Input path must be a directory when using batch mode: {input_path}"
            )

        # Get all image files
        supported_formats = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        image_paths = []
        for file in os.listdir(input_path):
            if any(file.lower().endswith(ext) for ext in supported_formats):
                image_paths.append(os.path.join(input_path, file))

        if not image_paths:
            raise click.BadParameter(f"No supported image files found in {input_path}")

        # If output is specified and is a directory, use it,
        # otherwise create a "nobg" subfolder
        if output and os.path.isdir(output):
            output_dir = output
        else:
            output_dir = os.path.join(input_path, "nobg")
            os.makedirs(output_dir, exist_ok=True)

        output_paths = batch_remove_background(
            image_paths,
            output_dir,
            alpha_matting,
            alpha_matting_foreground_threshold,
            alpha_matting_background_threshold,
            alpha_matting_erode_size,
            model,
            show_progress,
        )

        if show_progress:
            click.echo(
                f"Processed {len(output_paths)} images. Output saved to {output_dir}"
            )

    else:
        # Process single file
        if not os.path.isfile(input_path):
            raise click.BadParameter(
                f"Input path must be a file when not using batch mode: {input_path}"
            )

        output_path = remove_background(
            input_path,
            output,
            alpha_matting,
            alpha_matting_foreground_threshold,
            alpha_matting_background_threshold,
            alpha_matting_erode_size,
            model,
        )

        if show_progress:
            click.echo(f"Background removed. Output saved to {output_path}")


if __name__ == "__main__":
    remove_background_command()
