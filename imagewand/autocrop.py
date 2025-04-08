import os

import cv2
import numpy as np


def autocrop(
    input_path: str,
    output_path: str = None,
    mode: str = "auto",
    margin: int = -1,
    border_percent: int = -1,
) -> str:
    """Auto-crop scanned images.

    Args:
        input_path: Path to input image
        output_path: Path to output image (optional)
        mode: Cropping mode - "auto", "frame", or "border"
        margin: Margin in pixels for frame mode
        border_percent: Border percentage for border mode

    Returns:
        Path to output image
    """
    if mode not in ["auto", "frame", "border"]:
        raise ValueError(f"Invalid mode: {mode}")

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        if mode == "frame":
            suffix = f"_frame_m{margin}" if margin != -1 else "_frame"
        elif mode == "border":
            suffix = f"_border_b{border_percent}" if border_percent != -1 else "_border"
        else:
            suffix = "_auto"
        output_path = f"{base}{suffix}{ext}"

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")

    if mode == "frame":
        result = crop_framed_photo(input_path, output_path, margin=margin)
    elif mode == "border":
        result = crop_with_content_detection(input_path, output_path, mode="border", 
                                           threshold=30, border_percent=border_percent)
    else:  # auto mode
        try:
            result = crop_framed_photo(input_path, output_path, margin=margin)
        except ValueError:
            result = crop_with_content_detection(input_path, output_path, mode="border", 
                                               threshold=30, border_percent=border_percent)

    if result is None:
        raise ValueError("Failed to process image")

    return result


def crop_dark_background(
    image_path: str, output_path: str = None, threshold: int = 30
) -> str:
    """
    Crop out dark/black background, keeping only the main image content.
    Specifically designed for photos on black paper backgrounds.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Simple thresholding to find very dark regions
    _, dark_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find edges in the image
    edges = cv2.Canny(gray, 100, 200)

    # Dilate edges to connect them
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Combine edge detection with threshold
    content_mask = cv2.bitwise_or(dark_mask, edges)

    # Close gaps in the mask
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find contours in the mask
    contours, _ = cv2.findContours(
        content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No content found in image")

    # Find the largest contour
    main_contour = max(contours, key=cv2.contourArea)

    # Get the minimum area rectangle
    rect = cv2.minAreaRect(main_contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.int32)

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(box)

    # Add minimal padding
    pad_x = int(w * 0.005)  # 0.5% padding
    pad_y = int(h * 0.005)

    # Ensure padding doesn't exceed image bounds
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(img.shape[1] - x, w + 2 * pad_x)
    h = min(img.shape[0] - y, h + 2 * pad_y)

    # Crop the image
    cropped = img[y : y + h, x : x + w]

    # Create output path if not specified
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_cropped{ext}"

    # Save result
    cv2.imwrite(output_path, cropped)
    return output_path


def crop_framed_photo(
    image_path: str, output_path: str = None, margin: int = -5
) -> str:
    """
    Detect and crop the main photo from any contrasting background/frame,
    with improved frame detection. Default margin is now -5 for more aggressive cropping.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try multiple edge detection approaches
    # First with Canny
    edges = cv2.Canny(blurred, 30, 100)
    
    # Dilate edges to connect them
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no good contours, try with adaptive thresholding
    if not contours or max(cv2.contourArea(c) for c in contours) < img.shape[0] * img.shape[1] * 0.1:
        # Use adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Clean up with morphology
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours again
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No frame detected in the image")
    
    # Filter contours by area - we want the largest ones that aren't the entire image
    min_area = img.shape[0] * img.shape[1] * 0.05  # Lower threshold: 5% of image
    max_area = img.shape[0] * img.shape[1] * 0.98  # Higher threshold: 98% of image
    
    valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    
    if not valid_contours:
        # If no valid contours, try with the largest one
        largest_contour = max(contours, key=cv2.contourArea)
        valid_contours = [largest_contour]
    
    # Sort contours by area (largest first)
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
    
    # Try to find a rectangular contour first
    best_contour = None
    best_rect_score = float('inf')
    
    for contour in valid_contours[:5]:  # Check top 5 largest contours
        # Approximate the contour to simplify it
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If it has 4 vertices, it's likely a rectangle
        if len(approx) == 4:
            best_contour = contour
            break
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate how rectangular the contour is
        rect_area = w * h
        contour_area = cv2.contourArea(contour)
        rect_score = abs(1 - (contour_area / rect_area))
        
        # Check if it's more rectangular than previous best
        if rect_score < best_rect_score:
            best_rect_score = rect_score
            best_contour = contour
    
    if best_contour is None:
        best_contour = valid_contours[0]  # Use largest if no good rectangle found
    
    # Get bounding rectangle of best contour
    x, y, w, h = cv2.boundingRect(best_contour)
    
    # Apply margin
    if margin >= 0:
        # Add margin (positive value)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
    else:
        # Tighter crop (negative value)
        margin = abs(margin)
        x = min(x + margin, x + w - 1)
        y = min(y + margin, y + h - 1)
        w = max(w - 2 * margin, 1)
        h = max(h - 2 * margin, 1)
    
    # Crop the image
    cropped = img[y:y+h, x:x+w]
    
    # Create output path if not specified
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        if margin != -5:  # Only include margin in filename if it's not the default
            suffix = f"_frame_m{margin}"
        else:
            suffix = "_frame"
        output_path = f"{base}{suffix}{ext}"
    
    # Save result
    cv2.imwrite(output_path, cropped)
    return output_path


def crop_with_content_detection(
    image_path: str,
    output_path: str = None,
    mode: str = "auto",
    threshold: int = 30,
    border_percent: float = -1,
    margin: int = -1,
) -> str:
    """
    Enhanced crop function that supports different cropping modes.

    Args:
        image_path: Path to input image
        output_path: Path to save cropped image (optional)
        mode: Cropping mode:
            - 'frame': Extract photo from contrasting frame/background
            - 'border': Remove white margins only, keep the frame intact
            - 'auto': Try frame detection first, fall back to border removal
        threshold: Brightness threshold (0-255)
        border_percent: Percentage of border to keep (for 'border' mode)
        margin: Margin in pixels (-ve for more aggressive crop, +ve for more margin)
    """
    if mode == "frame":
        return crop_framed_photo(image_path, output_path, margin=margin)
    elif mode == "border":
        # Implement border detection directly
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try adaptive thresholding instead of simple thresholding
        # This works better for images with varying brightness
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations to clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find the bounding box of the content
        points = cv2.findNonZero(binary)
        if points is None:
            # Fall back to simple thresholding if adaptive fails
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            points = cv2.findNonZero(binary)
            if points is None:
                raise ValueError("No content found in the image")
            
        x, y, w, h = cv2.boundingRect(points)
        
        # Apply border percentage if specified
        if border_percent != -1:
            # Positive border_percent means keep more border
            # Negative border_percent means more aggressive crop
            border_x = int(w * border_percent / 100)
            border_y = int(h * border_percent / 100)
            
            x = max(0, x - border_x)
            y = max(0, y - border_y)
            w = min(img.shape[1] - x, w + 2 * border_x)
            h = min(img.shape[0] - y, h + 2 * border_y)
        
        # Crop the image
        cropped = img[y:y+h, x:x+w]
        
        # Create output path if not specified
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            if border_percent != -1:
                suffix = f"_border_b{border_percent}"
            else:
                suffix = "_border"
            output_path = f"{base}{suffix}{ext}"
        
        # Save result
        cv2.imwrite(output_path, cropped)
        return output_path
    else:  # auto mode
        try:
            result = crop_framed_photo(image_path, output_path, margin=margin)
            if output_path is None:
                base, ext = os.path.splitext(image_path)
                new_path = f"{base}_auto_frame{ext}"
                os.rename(result, new_path)
                return new_path
            return result
        except ValueError:
            # Use the border mode implementation directly
            if output_path is None:
                temp_output = None
            else:
                temp_output = output_path
                
            result = crop_with_content_detection(image_path, temp_output, mode="border", 
                                               threshold=threshold, border_percent=border_percent)
                
            if output_path is None:
                base, ext = os.path.splitext(image_path)
                suffix = (
                    f"_auto_border_b{border_percent}"
                    if border_percent != -1
                    else "_auto_border"
                )
                new_path = f"{base}{suffix}{ext}"
                os.rename(result, new_path)
                return new_path
            return result