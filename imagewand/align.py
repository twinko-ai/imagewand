import cv2
import numpy as np
import os


def align_image(
    input_path: str, 
    output_path: str = None, 
    method: str = "auto",
    angle_threshold: float = 1.0
) -> str:
    """
    Align a tilted image to be horizontal/vertical.
    
    Args:
        input_path: Path to input image
        output_path: Path to output image (optional)
        method: Alignment method - "auto", "hough", "contour", or "center"
        angle_threshold: Minimum angle to correct (degrees)
        
    Returns:
        Path to aligned image
    """
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    
    # Create default output path if not specified
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        # Include method and threshold in filename
        if method != "auto" or angle_threshold != 1.0:
            output_path = f"{base}_aligned_{method}_a{angle_threshold:.1f}{ext}"
        else:
            output_path = f"{base}_aligned{ext}"
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect angle based on method
    if method == "hough":
        angle = _detect_angle_hough(gray)
    elif method == "contour":
        angle = _detect_angle_contour(gray)
    elif method == "center":
        angle = _detect_angle_center(gray)
    else:  # auto - try all methods
        angle_hough = _detect_angle_hough(gray)
        angle_contour = _detect_angle_contour(gray)
        angle_center = _detect_angle_center(gray)
        
        # Choose the method that detected a more significant angle
        angles = [
            (abs(angle_hough), angle_hough),
            (abs(angle_contour), angle_contour),
            (abs(angle_center), angle_center)
        ]
        angles.sort(reverse=True)  # Sort by magnitude (descending)
        angle = angles[0][1]  # Take the angle with the largest magnitude
    
    # Only rotate if angle exceeds threshold
    if abs(angle) < angle_threshold:
        # Just copy the image if no significant rotation needed
        cv2.imwrite(output_path, img)
        return output_path
    
    # Rotate the image
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions to avoid cropping
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)
    
    # Adjust the rotation matrix
    rotation_matrix[0, 2] += new_width / 2 - center[0]
    rotation_matrix[1, 2] += new_height / 2 - center[1]
    
    # Perform the rotation
    rotated = cv2.warpAffine(img, rotation_matrix, (new_width, new_height), 
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, 
                            borderValue=(255, 255, 255))
    
    # Save the result
    cv2.imwrite(output_path, rotated)
    return output_path


def _detect_angle_hough(gray_img):
    """Detect rotation angle using Hough Line Transform"""
    # Apply edge detection
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    
    # Apply Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is None or len(lines) == 0:
        return 0.0
    
    # Analyze lines to find dominant angle
    angles = []
    for line in lines:
        rho, theta = line[0]
        # Convert to degrees and normalize to -45 to 45 range
        angle = np.degrees(theta) - 90
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        angles.append(angle)
    
    # Use the median angle for robustness
    return np.median(angles)


def _detect_angle_contour(gray_img):
    """Detect rotation angle using contour analysis"""
    # Threshold the image
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    
    # Get the angle
    angle = rect[2]
    
    # Normalize angle to -45 to 45 range
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    
    return angle


def _detect_angle_center(gray_img):
    """
    Detect rotation angle focusing on the center portion of the image.
    This method is useful for images with a clear central subject.
    """
    # Get image dimensions
    height, width = gray_img.shape
    
    # Define the center region (middle 60% of the image)
    center_width = int(width * 0.6)
    center_height = int(height * 0.6)
    start_x = (width - center_width) // 2
    start_y = (height - center_height) // 2
    
    # Extract the center region
    center_img = gray_img[start_y:start_y+center_height, start_x:start_x+center_width]
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(center_img, (5, 5), 0)
    
    # Apply adaptive thresholding to handle varying lighting conditions
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours in the center region
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # Filter contours by area to ignore very small ones
    min_area = (center_width * center_height) * 0.01  # 1% of center area
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if not significant_contours:
        return 0.0
    
    # Analyze angles of all significant contours
    angles = []
    for contour in significant_contours:
        # Get the minimum area rectangle
        rect = cv2.minAreaRect(contour)
        
        # Get width and height of the rectangle
        (_, _), (w, h), angle = rect
        
        # Only consider rectangles that are clearly not square
        if min(w, h) > 0 and max(w, h) / min(w, h) > 1.2:
            # Adjust angle based on rectangle orientation
            if w < h:
                angle += 90
                
            # Normalize angle to -45 to 45 range
            while angle < -45:
                angle += 90
            while angle > 45:
                angle -= 90
                
            angles.append(angle)
    
    if not angles:
        return 0.0
    
    # Use the median angle for robustness
    return np.median(angles) 