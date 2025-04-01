import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageChops
import cv2
from typing import Callable, List, Dict, Union, Tuple

# Define filter type
FilterFunction = Callable[[Image.Image, Dict], Image.Image]

# Dictionary to store all available filters
FILTERS = {}

def register_filter(name: str):
    """Decorator to register a filter function"""
    def decorator(func: FilterFunction):
        FILTERS[name] = func
        return func
    return decorator

# ===== Built-in Filters =====

@register_filter("grayscale")
def grayscale(img: Image.Image, params: Dict = None) -> Image.Image:
    """Convert image to grayscale"""
    return ImageOps.grayscale(img).convert('RGB')

@register_filter("sepia")
def sepia(img: Image.Image, params: Dict = None) -> Image.Image:
    """Apply sepia tone effect"""
    gray = ImageOps.grayscale(img)
    normalized = gray.convert('RGB')
    
    sepia_tone = (1.2, 1.0, 0.8)  # Sepia RGB multipliers
    
    data = np.array(normalized)
    sepia_data = np.zeros(data.shape, dtype=np.uint8)
    
    # Apply sepia multipliers
    for i in range(3):
        sepia_data[:, :, i] = np.clip(data[:, :, i] * sepia_tone[i], 0, 255).astype(np.uint8)
    
    return Image.fromarray(sepia_data)

@register_filter("blur")
def blur(img: Image.Image, params: Dict = None) -> Image.Image:
    """Apply Gaussian blur"""
    radius = params.get('radius', 2) if params else 2
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

@register_filter("sharpen")
def sharpen(img: Image.Image, params: Dict = None) -> Image.Image:
    """Sharpen the image"""
    factor = params.get('factor', 2.0) if params else 2.0
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)

@register_filter("brightness")
def brightness(img: Image.Image, params: Dict = None) -> Image.Image:
    """Adjust brightness"""
    factor = params.get('factor', 1.5) if params else 1.5
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

@register_filter("contrast")
def contrast(img: Image.Image, params: Dict = None) -> Image.Image:
    """Adjust contrast"""
    factor = params.get('factor', 1.5) if params else 1.5
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

@register_filter("saturation")
def saturation(img: Image.Image, params: Dict = None) -> Image.Image:
    """Adjust color saturation"""
    factor = params.get('factor', 1.5) if params else 1.5
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)

@register_filter("vignette")
def vignette(img: Image.Image, params: Dict = None) -> Image.Image:
    """Apply vignette effect"""
    # Convert to numpy array for OpenCV processing
    img_array = np.array(img)
    
    # Get image dimensions
    height, width = img_array.shape[:2]
    
    # Create a vignette mask
    X_resultant_kernel = cv2.getGaussianKernel(width, width/3)
    Y_resultant_kernel = cv2.getGaussianKernel(height, height/3)
    
    # Generate kernel matrix
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    
    # Normalize kernel
    mask = kernel / kernel.max()
    
    # Apply the mask to each channel
    for i in range(3):
        img_array[:, :, i] = img_array[:, :, i] * mask
    
    return Image.fromarray(img_array.astype(np.uint8))

@register_filter("edge_enhance")
def edge_enhance(img: Image.Image, params: Dict = None) -> Image.Image:
    """Enhance edges in the image"""
    return img.filter(ImageFilter.EDGE_ENHANCE)

@register_filter("emboss")
def emboss(img: Image.Image, params: Dict = None) -> Image.Image:
    """Apply emboss effect"""
    return img.filter(ImageFilter.EMBOSS)

@register_filter("invert")
def invert(img: Image.Image, params: Dict = None) -> Image.Image:
    """Invert the image colors"""
    return ImageOps.invert(img)

@register_filter("posterize")
def posterize(img: Image.Image, params: Dict = None) -> Image.Image:
    """Reduce the number of bits per color channel"""
    bits = params.get('bits', 2) if params else 2
    return ImageOps.posterize(img, bits)

@register_filter("solarize")
def solarize(img: Image.Image, params: Dict = None) -> Image.Image:
    """Invert all pixel values above a threshold"""
    threshold = params.get('threshold', 128) if params else 128
    return ImageOps.solarize(img, threshold)

# ===== Art-Specific Filters =====

@register_filter("clean_sketch")
def clean_sketch(img: Image.Image, params: Dict = None) -> Image.Image:
    """
    Enhance a pencil sketch or drawing by cleaning up the background
    and enhancing the lines
    """
    # Convert to numpy array for OpenCV processing
    img_array = np.array(img)
    
    # Convert to grayscale if it's not already
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    
    # Invert the image
    inverted = 255 - gray
    
    # Apply Gaussian blur
    blur_radius = params.get('blur_radius', 10) if params else 10
    blurred = cv2.GaussianBlur(inverted, (blur_radius, blur_radius), 0)
    
    # Invert the blurred image
    inverted_blurred = 255 - blurred
    
    # Create the pencil sketch effect by dividing the grayscale image by the inverted blurred image
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    
    # Convert back to RGB
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(sketch_rgb)

@register_filter("enhance_drawing")
def enhance_drawing(img: Image.Image, params: Dict = None) -> Image.Image:
    """
    Enhance a drawing by improving contrast and sharpness while
    cleaning up the background
    """
    # First, apply auto levels to improve contrast
    img = auto_levels(img)
    
    # Sharpen the image
    sharpness_factor = params.get('sharpness', 2.0) if params else 2.0
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(sharpness_factor)
    
    # Increase contrast slightly
    contrast_factor = params.get('contrast', 1.2) if params else 1.2
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    
    return img

@register_filter("auto_levels")
def auto_levels(img: Image.Image, params: Dict = None) -> Image.Image:
    """
    Automatically adjust levels to improve contrast and color balance
    (similar to "Auto Levels" in Photoshop)
    """
    # Convert to LAB color space for better results
    img_array = np.array(img)
    
    # Check if the image is grayscale
    if len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1):
        # For grayscale images
        min_val = np.percentile(img_array, 2)
        max_val = np.percentile(img_array, 98)
        
        # Apply contrast stretching
        img_array = np.clip((img_array - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    else:
        # For color images
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge the CLAHE enhanced L channel with the original A and B channels
        merged = cv2.merge((cl, a, b))
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced)

@register_filter("remove_background")
def remove_background(img: Image.Image, params: Dict = None) -> Image.Image:
    """
    Remove or clean up the background of a drawing or painting
    """
    # Convert to numpy array for OpenCV processing
    img_array = np.array(img)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    
    # Apply threshold to separate foreground from background
    threshold = params.get('threshold', 230) if params else 230
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Create a mask from the binary image
    mask = binary.copy()
    
    # Apply the mask to the original image
    result = img_array.copy()
    if len(img_array.shape) == 3:
        for i in range(3):
            result[:, :, i] = cv2.bitwise_and(result[:, :, i], result[:, :, i], mask=255-mask)
            # Set background to white
            result[:, :, i] = cv2.add(result[:, :, i], mask)
    
    return Image.fromarray(result)

@register_filter("enhance_colors")
def enhance_colors(img: Image.Image, params: Dict = None) -> Image.Image:
    """
    Enhance colors in a painting or colored drawing
    """
    # Increase saturation
    saturation_factor = params.get('saturation', 1.3) if params else 1.3
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    
    # Increase contrast slightly
    contrast_factor = params.get('contrast', 1.1) if params else 1.1
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    
    # Apply auto levels
    img = auto_levels(img)
    
    return img

@register_filter("prepare_for_print")
def prepare_for_print(img: Image.Image, params: Dict = None) -> Image.Image:
    """
    Prepare an image for printing by enhancing details and adjusting colors
    """
    # Auto levels first
    img = auto_levels(img)
    
    # Enhance sharpness
    sharpness_factor = params.get('sharpness', 1.5) if params else 1.5
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(sharpness_factor)
    
    # Slightly increase contrast
    contrast_factor = params.get('contrast', 1.1) if params else 1.1
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    
    # Slightly increase saturation
    saturation_factor = params.get('saturation', 1.1) if params else 1.1
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    
    return img

@register_filter("prepare_for_web")
def prepare_for_web(img: Image.Image, params: Dict = None) -> Image.Image:
    """
    Prepare an image for web display by enhancing colors and details
    """
    # Auto levels first
    img = auto_levels(img)
    
    # Enhance sharpness
    sharpness_factor = params.get('sharpness', 1.3) if params else 1.3
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(sharpness_factor)
    
    # Increase saturation for more vibrant web display
    saturation_factor = params.get('saturation', 1.2) if params else 1.2
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    
    # Increase contrast slightly
    contrast_factor = params.get('contrast', 1.15) if params else 1.15
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    
    return img

@register_filter("line_art")
def line_art(img: Image.Image, params: Dict = None) -> Image.Image:
    """
    Convert an image to line art, emphasizing edges and lines
    """
    # Convert to numpy array for OpenCV processing
    img_array = np.array(img)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive threshold to get binary image
    threshold_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if params and params.get('adaptive', True) else cv2.THRESH_BINARY
    block_size = params.get('block_size', 11) if params else 11
    c_value = params.get('c_value', 2) if params else 2
    
    if threshold_type == cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
        binary = cv2.adaptiveThreshold(bilateral, 255, threshold_type, 
                                      cv2.THRESH_BINARY_INV, block_size, c_value)
    else:
        _, binary = cv2.threshold(bilateral, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Convert back to RGB
    result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(result)

@register_filter("watercolor_effect")
def watercolor_effect(img: Image.Image, params: Dict = None) -> Image.Image:
    """
    Apply a watercolor-like effect to an image
    """
    # Convert to numpy array for OpenCV processing
    img_array = np.array(img)
    
    # Apply bilateral filter for edge-preserving smoothing
    bilateral_iterations = params.get('iterations', 2) if params else 2
    filtered = img_array.copy()
    
    for _ in range(bilateral_iterations):
        filtered = cv2.bilateralFilter(filtered, 9, 75, 75)
    
    # Enhance edges
    gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    edges = 255 - edges
    
    # Convert edges to 3 channels
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Blend edges with filtered image
    alpha = params.get('edge_strength', 0.6) if params else 0.6
    result = cv2.addWeighted(filtered, 1.0, edges_rgb, alpha, 0)
    
    # Increase saturation
    result_pil = Image.fromarray(result)
    saturation_factor = params.get('saturation', 1.4) if params else 1.4
    enhancer = ImageEnhance.Color(result_pil)
    result_pil = enhancer.enhance(saturation_factor)
    
    return result_pil

# ===== Filter Application Functions =====

def apply_filter(image_path: str, filter_name: str, output_path: str = None, 
                params: Dict = None, progress_callback: Callable = None) -> str:
    """
    Apply a single filter to an image
    
    Args:
        image_path: Path to the input image
        filter_name: Name of the filter to apply
        output_path: Path to save the filtered image (if None, will use default naming)
        params: Parameters for the filter
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Path to the filtered image
    """
    if progress_callback:
        progress_callback(10)  # 10% - Starting
    
    # Check if filter exists
    if filter_name not in FILTERS:
        raise ValueError(f"Filter '{filter_name}' not found. Available filters: {list(FILTERS.keys())}")
    
    # Create default output path if not provided
    if output_path is None:
        file_dir = os.path.dirname(image_path)
        file_name, file_ext = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(file_dir, f"{file_name}_{filter_name}{file_ext}")
    
    # Open the image
    img = Image.open(image_path)
    
    if progress_callback:
        progress_callback(30)  # 30% - Image loaded
    
    # Apply the filter
    filter_func = FILTERS[filter_name]
    filtered_img = filter_func(img, params)
    
    if progress_callback:
        progress_callback(70)  # 70% - Filter applied
    
    # Save the filtered image
    filtered_img.save(output_path)
    
    if progress_callback:
        progress_callback(100)  # 100% - Image saved
    
    return output_path

def apply_filters(image_path: str, filter_names: List[str], output_path: str = None, 
                 params_list: List[Dict] = None, progress_callback: Callable = None) -> str:
    """
    Apply multiple filters to an image in sequence
    
    Args:
        image_path: Path to the input image
        filter_names: List of filter names to apply in sequence
        output_path: Path to save the filtered image
        params_list: List of parameter dictionaries for each filter
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Path to the filtered image
    """
    if not filter_names:
        return image_path
    
    # Create default output path if not provided
    if output_path is None:
        file_dir = os.path.dirname(image_path)
        file_name, file_ext = os.path.splitext(os.path.basename(image_path))
        filter_suffix = "_".join(filter_names)
        output_path = os.path.join(file_dir, f"{file_name}_{filter_suffix}{file_ext}")
    
    # Open the image
    img = Image.open(image_path)
    
    # Prepare parameters
    if params_list is None:
        params_list = [None] * len(filter_names)
    
    # Apply filters in sequence
    total_filters = len(filter_names)
    for i, (filter_name, params) in enumerate(zip(filter_names, params_list)):
        if filter_name not in FILTERS:
            raise ValueError(f"Filter '{filter_name}' not found. Available filters: {list(FILTERS.keys())}")
        
        # Apply the filter
        filter_func = FILTERS[filter_name]
        img = filter_func(img, params)
        
        # Update progress
        if progress_callback:
            progress = int(((i + 1) / total_filters) * 100)
            progress_callback(progress)
    
    # Save the filtered image
    img.save(output_path)
    
    return output_path

def batch_apply_filters(image_paths: List[str], filter_names: List[str], output_dir: str = None,
                       params_list: List[Dict] = None, progress_callback: Callable = None) -> List[str]:
    """
    Apply filters to multiple images
    
    Args:
        image_paths: List of paths to input images
        filter_names: List of filter names to apply in sequence
        output_dir: Directory to save filtered images
        params_list: List of parameter dictionaries for each filter
        progress_callback: Optional callback function for progress updates
        
    Returns:
        List of paths to filtered images
    """
    if not image_paths:
        return []
    
    # Create output directory if not provided
    if output_dir is None:
        # Use the directory of the first image
        output_dir = os.path.join(os.path.dirname(image_paths[0]), "filtered_images")
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = []
    total_images = len(image_paths)
    
    for i, image_path in enumerate(image_paths):
        # Create output path for this image
        file_name, file_ext = os.path.splitext(os.path.basename(image_path))
        filter_suffix = "_".join(filter_names)
        output_path = os.path.join(output_dir, f"{file_name}_{filter_suffix}{file_ext}")
        
        # Apply filters to this image
        apply_filters(image_path, filter_names, output_path, params_list)
        output_paths.append(output_path)
        
        # Update progress
        if progress_callback:
            progress = int(((i + 1) / total_images) * 100)
            progress_callback(progress)
    
    return output_paths

def list_filters() -> List[str]:
    """Return a list of all available filter names"""
    return list(FILTERS.keys())

def add_custom_filter(name: str, filter_func: FilterFunction) -> None:
    """
    Add a custom filter function
    
    Args:
        name: Name for the custom filter
        filter_func: Function that takes an image and parameters and returns a filtered image
    """
    FILTERS[name] = filter_func 