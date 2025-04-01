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
- `blur:radius=3` - Control blur strength (default: 2)
- `brightness:factor=1.2` - Adjust brightness (default: 1.5)
- `saturation:factor=1.3` - Control color saturation (default: 1.5)

The output filename will include the parameters used:

### Filter Presets

Save and reuse your favorite filter combinations as presets:

```bash
# Save a filter combination as a preset
imagewand filter drawing.jpg -f "saturation:factor=1.2,contrast:factor=1.2,sharpen:factor=1.2" --save-preset kids_drawing

# Apply the saved preset to an image
imagewand filter drawing.jpg -p kids_drawing

# Apply preset with custom output path
imagewand filter drawing.jpg -p kids_drawing -o enhanced.jpg

# List all available presets
imagewand list-presets
```

The presets are stored in `~/.imagewand_presets` file in INI format:
```ini
[presets]
kids_drawing = saturation:factor=1.2,contrast:factor=1.2,sharpen:factor=1.2
light_pencil = contrast:factor=1.8,contrast:factor=1.5,sharpen:factor=2.0
colorful = saturation:factor=1.5,contrast:factor=1.3,brightness:factor=1.1
```

Common preset examples:
```bash
# For kids' pencil drawings
imagewand filter drawing.jpg -f "contrast:factor=1.2,sharpen:factor=1.8" --save-preset pencil_drawing

# For colorful artwork
imagewand filter painting.jpg -f "saturation:factor=1.3,contrast:factor=1.2" --save-preset vibrant_art

# For old photos
imagewand filter photo.jpg -f "contrast:factor=1.4,brightness:factor=1.1,sharpen:factor=1.5" --save-preset photo_enhance
```

You can also edit the `~/.imagewand_presets` file directly to add or modify presets. The output filename will include the preset name:

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


## Special Use Cases

### Enhancing Kids' Drawings

To make scanned pencil drawings look better, you can try these filter combinations:

```bash
# Basic enhancement for light pencil drawings
imagewand filter drawing.jpg -f "contrast,sharpen"

# For very light drawings, use this sequence
imagewand filter drawing.jpg -f "contrast,contrast,sharpen"

# To make colors pop (for colored pencils)
imagewand filter drawing.jpg -f "saturation,contrast,sharpen"

# To clean up the background and enhance lines
imagewand filter drawing.jpg -f "threshold,sharpen"

# For preserving light details while enhancing contrast
imagewand filter drawing.jpg -f "adaptive_contrast,sharpen"
```

Tips for scanning kids' drawings:
1. Scan in color mode even for pencil drawings
2. Use at least 300 DPI for better detail
3. Make sure the paper is flat on the scanner
4. Clean the scanner glass for best results

Common filter combinations explained:
- `contrast,sharpen`: Best for regular pencil drawings
- `saturation,contrast`: Great for colored pencil art
- `threshold`: Good for line drawings and sketches
- `adaptive_contrast`: Best for drawings with varying pressure
- `denoise,sharpen`: Helpful for rough paper texture

Example workflow:
```bash
# First, auto-fix the scanned drawing to straighten and crop
imagewand autofix drawing.jpg -o fixed.jpg

# Then apply enhancement filters
imagewand filter fixed.jpg -f "contrast,sharpen" -o enhanced.jpg
```

For batch processing multiple drawings:
```bash
# Enhance all drawings in a folder
imagewand filter drawings_folder/ -f "contrast,sharpen" -o enhanced_drawings/
```
```

This section provides:
1. Specific filter combinations for different types of drawings
2. Tips for scanning
3. Explanation of each filter combination
4. Complete workflow examples
5. Batch processing instructions

Let me know if you'd like me to add more filter combinations or specific examples!
## Tips

1. Use `--debug` flag with any command for detailed output
2. For autofix:
   - Use `frame` mode for artwork on contrasting backgrounds
   - Use `border` mode for documents with white margins
   - Try negative margins/borders for more aggressive cropping
   - Use positive margins/borders to keep more content
3. For merging scans:
   - Remove white borders automatically
   - Images should have overlapping regions
   - Order matters when specifying individual files
4. For PDF conversion:
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
