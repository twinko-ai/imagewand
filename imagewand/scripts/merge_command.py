from pathlib import Path
from typing import List

import click
import cv2
from tqdm import tqdm

from ..automerge import AutoMerge


@click.command()
@click.argument("input_files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output", "-o", type=click.Path(), required=True, help="Output file path"
)
@click.option("--debug/--no-debug", default=False, help="Enable debug mode")
def merge(input_files: List[str], output: str, debug: bool):
    """
    Merge multiple scanned images into a single image.

    INPUT_FILES: List of input image files to merge
    """
    if len(input_files) < 2:
        raise click.BadParameter("At least two input files are required")

    # Create output directory if it doesn't exist
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load images with progress bar
    click.echo("Loading input images...")
    images = []
    for file in tqdm(input_files, desc="Loading images"):
        img = cv2.imread(str(file))
        if img is None:
            raise click.BadParameter(f"Could not load image: {file}")
        images.append(img)

    # Initialize merger
    merger = AutoMerge(debug=debug)

    # Perform merge
    click.echo("Merging images...")
    result = merger.merge_images(images)

    # Save result
    click.echo(f"Saving merged result to {output}")
    cv2.imwrite(str(output), result)
    click.echo("Done!")


if __name__ == "__main__":
    merge()
