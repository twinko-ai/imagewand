import cv2
import numpy as np
from PIL import Image
import os
import math

def autofix(input_path, output_path=None, border_percent=0, progress_callback=None):
    """
    Automatically detect, straighten, and crop an image to remove excess white space.
    
    Args:
        input_path (str): Path to the input image
        output_path (str): Path to save the processed image (if None, will use input_path with '_fixed' suffix)
        border_percent (float): Percentage of border to keep around the detected content (0-100)
        progress_callback (callable): Optional callback function for progress updates
        
    Returns:
        str: Path to the processed image
    """
    if progress_callback:
        progress_callback(5)  # 5% - Starting
    
    # Create default output path if not provided
    if output_path is None:
        file_dir = os.path.dirname(input_path)
        file_name, file_ext = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(file_dir, f"autofix_{file_name}{file_ext}")
    
    # Open image with PIL first (for broader format support)
    pil_img = Image.open(input_path)
    
    if progress_callback:
        progress_callback(10)  # 10% - Image loaded
    
    # Convert to numpy array for OpenCV processing
    img = np.array(pil_img)
    
    # Convert to grayscale if it's not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    if progress_callback:
        progress_callback(20)  # 20% - Converted to grayscale
    
    # Apply adaptive thresholding for better edge detection
    # This works better for varying lighting conditions and different types of content
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations to clean up the binary image
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if progress_callback:
        progress_callback(30)  # 30% - Found contours
    
    if not contours:
        # Try a different thresholding approach
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("No content detected in the image.")
            pil_img.save(output_path)
            return output_path
    
    # Filter small contours (noise)
    min_contour_area = img.shape[0] * img.shape[1] * 0.001  # 0.1% of image area
    significant_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    
    if not significant_contours:
        significant_contours = contours
    
    # Combine all significant contours
    all_contours = np.vstack(significant_contours)
    
    # Get the minimum area rectangle that contains all contours
    rect = cv2.minAreaRect(all_contours)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    if progress_callback:
        progress_callback(40)  # 40% - Found bounding box
    
    # Get the angle of the rectangle
    center, (width, height), angle = rect
    
    # OpenCV's angle is different from what we need for rotation
    # If width < height, the angle needs to be adjusted
    if width < height:
        angle = angle - 90
    
    # Adjust angle to be between -45 and 45 degrees
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    
    if progress_callback:
        progress_callback(50)  # 50% - Calculated rotation angle
    
    # Only rotate if the angle is significant (more than 1 degree)
    if abs(angle) > 1:
        # Rotate the image to straighten it
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    else:
        rotated = img.copy()
    
    if progress_callback:
        progress_callback(70)  # 70% - Rotated image
    
    # Convert back to grayscale and threshold again to find the content in the straightened image
    if len(rotated.shape) == 3:
        rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
    else:
        rotated_gray = rotated.copy()
    
    # Use Otsu's thresholding for better content detection
    _, rotated_binary = cv2.threshold(rotated_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the binary image
    rotated_binary = cv2.morphologyEx(rotated_binary, cv2.MORPH_CLOSE, kernel)
    rotated_binary = cv2.morphologyEx(rotated_binary, cv2.MORPH_OPEN, kernel)
    
    rotated_contours, _ = cv2.findContours(rotated_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not rotated_contours:
        print("No content detected after rotation.")
        pil_img.save(output_path)
        return output_path
    
    # Filter small contours again
    significant_rotated_contours = [c for c in rotated_contours if cv2.contourArea(c) > min_contour_area]
    
    if not significant_rotated_contours:
        significant_rotated_contours = rotated_contours
    
    # Find the bounding box of all contours
    x_min, y_min = rotated.shape[1], rotated.shape[0]
    x_max, y_max = 0, 0
    
    for contour in significant_rotated_contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + cw)
        y_max = max(y_max, y + ch)
    
    if progress_callback:
        progress_callback(80)  # 80% - Found crop area
    
    # Add border if requested
    if border_percent > 0:
        border_x = int((x_max - x_min) * border_percent / 100)
        border_y = int((y_max - y_min) * border_percent / 100)
        
        x_min = max(0, x_min - border_x)
        y_min = max(0, y_min - border_y)
        x_max = min(rotated.shape[1], x_max + border_x)
        y_max = min(rotated.shape[0], y_max + border_y)
    
    # Crop the image
    cropped = rotated[y_min:y_max, x_min:x_max]
    
    if progress_callback:
        progress_callback(90)  # 90% - Cropped image
    
    # Convert back to PIL and save
    result_img = Image.fromarray(cropped)
    result_img.save(output_path)
    
    if progress_callback:
        progress_callback(100)  # 100% - Saved image
    
    return output_path 