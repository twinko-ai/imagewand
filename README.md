# ImageWand

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

Automatically straighten and crop scanned images:

```bash
# Basic usage
imagewand autofix scan.jpg

# Process entire directory
imagewand autofix scans_folder/

# Adjust border percentage
imagewand autofix scan.jpg -b 10
```

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

### Merge Scanned Images

Merge multiple scanned images into a single image:

```bash
# Merge specific images
imagewand merge image1.jpg image2.jpg -o merged.jpg

# Merge all images in a directory
imagewand merge scans_folder/ -o merged.jpg

# Enable debug mode for detailed output
imagewand merge scans_folder/ -o merged.jpg --debug
```

## Common Use Cases

### Converting Multi-page Documents

Convert a multi-page PDF to high-resolution images:
```bash
imagewand pdf2img document.pdf -d 300 -f png -o document_pages/
```

### Processing Scanned Documents

Auto-fix and merge multiple scanned pages:
```bash
# First, auto-fix all scans
imagewand autofix scans/ -o fixed_scans/

# Then merge the fixed scans
imagewand merge fixed_scans/ -o final_document.jpg
```

### Batch Processing

Process multiple images in a directory:
```bash
# Resize all images in a directory
imagewand resize images/ -w 1024

# Apply filters to all images recursively
imagewand filter images/ -f "sharpen,contrast" -r
```

## Tips

1. Use `--debug` flag with any command for detailed output
2. For merging scans:
   - Remove white borders automatically
   - Images should have overlapping regions
   - Order matters when specifying individual files
3. For PDF conversion:
   - Higher DPI values give better quality but larger files
   - PNG format is better for text documents
   - JPG is better for photos

## Common Issues

- If merge fails, ensure images have enough overlapping content
- For better auto-fix results, ensure scans have good contrast
- When resizing, specify either width or height to maintain aspect ratio

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
