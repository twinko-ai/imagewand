# ImageWand

[![PyPI version](https://badge.fury.io/py/imagewand.svg)](https://badge.fury.io/py/imagewand)
[![Python Versions](https://img.shields.io/pypi/pyversions/imagewand.svg)](https://pypi.org/project/imagewand)
[![Build Status](https://github.com/twinko-ai/imagewand/actions/workflows/ci.yml/badge.svg)](https://github.com/twinko-ai/imagewand/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
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
- `filter`: Apply various filters to images
- `merge`: Merge multiple scanned images into one

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
- `blur:radius=3`