# ImageWand

[![PyPI version](https://badge.fury.io/py/imagewand.svg)](https://badge.fury.io/py/imagewand)
[![Python Versions](https://img.shields.io/pypi/pyversions/imagewand.svg)](https://pypi.org/project/imagewand)
[![Build Status](https://github.com/twinko-ai/imagewand/actions/workflows/ci.yml/badge.svg)](https://github.com/twinko-ai/imagewand/actions)
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
- `autofix`: Automatically straighten and crop scanned images
- `align`: Automatically align tilted images to be horizontal/vertical
- `filter`: Apply various filters to images
- `merge`: Merge multiple scanned images into one
- `crop`: Crop images using frame detection or border removal

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

# Specify output path
imagewand resize input.jpg -o resized.jpg -w 800
```

### Auto-fix Scanned Images

Automatically crop and clean up scanned images with two specialized modes:

```bash
# Frame mode - Extract photo from contrasting frame/background
imagewand autofix scan.jpg -m frame
imagewand autofix scan.jpg -m frame --margin -2  # More aggressive crop
imagewand autofix scan.jpg -m frame --margin 5   # Add margin

# Border mode - Remove white margins only
imagewand autofix scan.jpg -m border
imagewand autofix scan.jpg -m border -b -2  # More aggressive crop
imagewand autofix scan.jpg -m border -b 5   # Keep more border

# Auto mode (default) - Tries frame detection first, falls back to border removal
imagewand autofix scan.jpg

# Process entire directory
imagewand autofix scans_folder/
```

Cropping modes explained:
- `frame`: Best for photos on contrasting backgrounds (e.g., artwork on black paper)
  - Use `--margin` to adjust cropping (negative for tighter crop, positive for more margin)
  - Default margin is -1 for slightly aggressive crop
  
- `border`: Best for documents with white margins
  - Use `-b` to adjust border percentage (negative for tighter crop, positive to keep more border)
  - Default border is 0 for exact content boundaries
  
- `auto`: Attempts frame detection first, falls back to border removal if no frame is detected

Output filenames reflect the mode and parameters used:
- Frame mode: `photo_frame.jpg` or `photo_frame_m2.jpg` (with margin 2)
- Border mode: `photo_border.jpg` or `photo_border_b5.jpg` (with 5% border)
- Auto mode: `photo_auto_frame.jpg` or `photo_auto_border.jpg`

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

```

### Crop Images

Crop images using frame detection or border removal:

```bash
# Frame mode - Extract photo from contrasting frame/background
imagewand crop photo.jpg -m frame
imagewand crop photo.jpg -m frame --margin -2  # More aggressive crop
imagewand crop photo.jpg -m frame --margin 5   # Add margin

# Border mode - Remove white margins only
imagewand crop photo.jpg -m border
imagewand crop photo.jpg -m border -b -2  # More aggressive crop
imagewand crop photo.jpg -m border -b 5   # Keep more border

# Adjust threshold for content detection (default: 30)
imagewand crop photo.jpg -m border -t 50  # Higher threshold for darker content
```

Cropping modes explained:
- `frame`: Best for photos on contrasting backgrounds (e.g., artwork on black paper)
  - Use `--margin` to adjust cropping (negative for tighter crop, positive for more margin)
  - Default margin is -1 for slightly aggressive crop
  
- `border`: Best for documents with white margins
  - Use `-b` to adjust border percentage (negative for tighter crop, positive to keep more border)
  - Default border is 0 for exact content boundaries

Output filenames:
- Frame mode: `photo_frame.jpg` or `photo_frame_m2.jpg` (with margin 2)
- Border mode: `photo_border.jpg` or `photo_border_b5.jpg` (with 5% border)