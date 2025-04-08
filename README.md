# ImageWand

[![PyPI version](https://badge.fury.io/py/imagewand.svg)](https://badge.fury.io/py/imagewand)
[![Python Versions](https://img.shields.io/pypi/pyversions/imagewand.svg)](https://pypi.org/project/imagewand)
[![Build Status](https://github.com/twinko-ai/imagewand/actions/workflows/ci.yml/badge.svg)](https://github.com/twinko-ai/imagewand/actions)
[![codecov](https://codecov.io/gh/twinko-ai/imagewand/branch/main/graph/badge.svg)](https://codecov.io/gh/twinko-ai/imagewand)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
[![GitHub last commit](https://img.shields.io/github/last-commit/twinko-ai/imagewand.svg)](https://github.com/twinko-ai/imagewand/commits)
[![GitHub repo size](https://img.shields.io/github/repo-size/twinko-ai/imagewand.svg)](https://github.com/twinko-ai/imagewand)
[![Downloads](https://static.pepy.tech/badge/imagewand)](https://pepy.tech/project/imagewand)

A powerful command-line tool for image manipulation, including PDF to image conversion, image resizing, auto-fixing scanned images, applying filters, and merging scanned images.

## Installation

```bash
pip install imagewand
```

## Commands Overview

ImageWand provides several commands for different image manipulation tasks:

- `pdf2img`: Convert PDF files to images
- `resize`: Resize images
- `align`: Automatically align tilted images to be horizontal/vertical
- `filter`: Apply various filters to images
- `merge`: Merge multiple scanned images into one
- `autocrop`: Automatically crop images using frame detection or border removal
- `info`: Display detailed information about images

## Command Usage

### PDF to Images

Convert PDF files to images with customizable DPI and format:

```bash
# Basic usage
imagewand pdf2img input.pdf

# Specify output directory and format
imagewand pdf2img input.pdf -o output_dir -f png

# Set custom DPI
imagewand pdf2img input.pdf -d 300
```

### Resize Images

Resize images while maintaining aspect ratio:

```bash
# Resize by width
imagewand resize input.jpg -w 800

# Resize by height
imagewand resize input.jpg --height 600

# Resize by percentage
imagewand resize input.jpg -p 50

# Specify output path
imagewand resize input.jpg -o resized.jpg -w 800
```

### Align Tilted Images

Automatically detect and correct the rotation of tilted images:

```bash
# Basic usage - auto-detects the best method
imagewand align tilted_scan.jpg

# Specify output path
imagewand align tilted_scan.jpg -o aligned.jpg

# Choose specific alignment method
imagewand align tilted_scan.jpg -m hough
imagewand align tilted_scan.jpg -m contour
imagewand align tilted_scan.jpg -m center

# Adjust minimum angle threshold for correction (default: 1.0 degrees)
imagewand align tilted_scan.jpg -a 0.5  # More sensitive
imagewand align tilted_scan.jpg -a 2.0  # Less sensitive
```

Alignment methods explained:
- `auto` (default): Tries all methods and uses the one that detects the most significant angle
- `hough`: Uses Hough Line Transform, best for documents with clear straight lines
- `contour`: Uses contour analysis, better for images with distinct shapes
- `center`: Focuses on the central portion of the image, ignoring the background (best for photos with clear subjects)

Output filenames:
- Default: `image_aligned.jpg` (for default parameters)
- With parameters: `image_aligned_hough_a0.5.jpg` (for method=hough, angle_threshold=0.5)

### Apply Filters

Apply various image filters:

```bash
# Apply single filter
imagewand filter input.jpg -f grayscale

# Apply multiple filters
imagewand filter input.jpg -f "grayscale,sharpen"

# Process directory recursively
imagewand filter images/ -f grayscale -r

# List available filters
imagewand list-filters
```

### Filter Parameters

You can customize filter parameters using the format `filtername:param=value`:

```bash
# Adjust contrast level (default is 1.5)
imagewand filter drawing.jpg -f "contrast:factor=1.2"

# Multiple filters with custom parameters
imagewand filter drawing.jpg -f "saturation:factor=1.3,contrast:factor=1.2,sharpen:factor=1.8"

# Mix of default and custom parameters
imagewand filter drawing.jpg -f "saturation:factor=1.3,contrast,sharpen:factor=2.0"
```

Common filter parameters:
- `contrast:factor=1.2` - Lighter contrast (default: 1.5)
- `sharpen:factor=1.8` - Stronger sharpening (default: 2.0)
- `blur:radius=3` - Adjust blur radius (default: 2)

### Merge Images

Merge multiple scanned images into one:

```bash
# Merge all images in a directory
imagewand merge images_folder/ -o merged.jpg

# Merge specific images
imagewand merge image1.jpg image2.jpg image3.jpg -o merged.jpg

# Enable debug mode to see the matching process
imagewand merge images_folder/ -o merged.jpg --debug
```

### Autocrop Images

Automatically crop images using frame detection or border removal:

```bash
# Auto mode - Tries frame detection first, falls back to border removal
imagewand autocrop photo.jpg

# Frame mode - Extract photo from contrasting frame/background
imagewand autocrop photo.jpg -m frame
imagewand autocrop photo.jpg -m frame --margin -5  # More aggressive crop
imagewand autocrop photo.jpg -m frame --margin 10  # Add margin

# Border mode - Remove white margins only
imagewand autocrop photo.jpg -m border
imagewand autocrop photo.jpg -m border -b -5  # More aggressive crop
imagewand autocrop photo.jpg -m border -b 10  # Keep more border
```

Cropping modes explained:
- `auto` (default): Tries frame detection first, falls back to border removal if no frame is detected
- `frame`: Best for photos on contrasting backgrounds (e.g., artwork on black paper)
  - Use `--margin` to adjust cropping (negative for tighter crop, positive for more margin)
  - Default margin is -5 for slightly aggressive crop
  
- `border`: Best for documents with white margins
  - Use `-b` to adjust border percentage (negative for tighter crop, positive to keep more border)
  - Default border is 0 for exact content boundaries

Output filenames:
- Auto mode: `photo_auto.jpg`, `photo_auto_frame.jpg`, or `photo_auto_border.jpg`
- Frame mode: `photo_frame.jpg` or `photo_frame_m10.jpg` (with margin 10)
- Border mode: `photo_border.jpg` or `photo_border_b10.jpg` (with 10% border)

### Image Information

Display comprehensive information about images:

```bash
# Basic usage
imagewand info photo.jpg
```

The info command displays:
- Basic file information: filename, path, size, modification date
- Image properties: dimensions, format, color mode, color depth
- DPI information
- Color statistics: brightness, RGB channel values, sharpness
- EXIF data (when --verbose flag is used)