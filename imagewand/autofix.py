import cv2
import numpy as np
from PIL import Image
import os
import math


def autofix(input_path, output_path=None, border_percent=-1, progress_callback=None):
    """
    Remove white margins only, keeping the actual content and frame intact.

    Args:
        input_path (str): Path to the input image
        output_path (str): Path to save the processed image
        border_percent: Percentage of border (-5 to crop more aggressively,
                       positive to keep more border, default: -1)
        progress_callback (callable): Optional callback function for progress updates
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not load image: {input_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Even more aggressive threshold for white margins
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # Lowered to 240

    # More aggressive noise cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Increased kernel size
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_DILATE, kernel
    )  # Dilate to connect content
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel)  # Erode to remove noise

    # Find contours of non-white regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No content detected in the image.")
        if output_path is None:
            output_path = input_path.replace(".", "_fixed.")
        cv2.imwrite(output_path, img)
        return output_path

    # Find exact bounding box of all content
    x_min, y_min = img.shape[1], img.shape[0]
    x_max, y_max = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # Apply border percentage (now handles negative values)
    border_x = int((x_max - x_min) * abs(border_percent) / 100)
    border_y = int((y_max - y_min) * abs(border_percent) / 100)

    if border_percent >= 0:
        # Add border
        x_min = max(0, x_min - border_x)
        y_min = max(0, y_min - border_y)
        x_max = min(img.shape[1], x_max + border_x)
        y_max = min(img.shape[0], y_max + border_y)
    else:
        # Crop more aggressively
        x_min = min(x_min + border_x, x_max)
        y_min = min(y_min + border_y, y_max)
        x_max = max(x_max - border_x, x_min)
        y_max = max(y_max - border_y, y_min)

    # Crop exactly at the bounds
    cropped = img[y_min:y_max, x_min:x_max]

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        suffix = f"_border_b{border_percent}" if border_percent > 0 else "_border"
        output_path = f"{base}{suffix}{ext}"

    cv2.imwrite(output_path, cropped)
    return output_path


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
    image_path: str, output_path: str = None, margin: int = -1
) -> str:
    """
    Detect and crop the main photo from any contrasting background/frame,
    with more lenient frame detection.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # More lenient edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduced blur kernel
    edges = cv2.Canny(blurred, 30, 120)  # Lower thresholds for more edges

    # Analyze content distribution
    rows = np.sum(edges, axis=1)
    cols = np.sum(edges, axis=0)

    # More lenient gradient thresholds
    row_gradient = np.gradient(rows)
    col_gradient = np.gradient(cols)

    row_thresh = np.max(np.abs(row_gradient)) * 0.15  # Reduced from 0.2
    col_thresh = np.max(np.abs(col_gradient)) * 0.15  # Reduced from 0.2

    # Find content boundaries
    top_indices = np.where(np.abs(row_gradient) > row_thresh)[0]
    left_indices = np.where(np.abs(col_gradient) > col_thresh)[0]

    if len(top_indices) == 0 or len(left_indices) == 0:
        raise ValueError("No clear content boundaries found")

    top = top_indices[0]
    bottom = top_indices[-1]
    left = left_indices[0]
    right = left_indices[-1]

    # More lenient edge connection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # Reduced from 11x11
    dilated = cv2.dilate(edges, kernel, iterations=2)  # Reduced iterations

    # Find contours with more lenient parameters
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No content found in image")

    # More lenient contour filtering
    img_area = img.shape[0] * img.shape[1]
    filtered_contours = []

    content_width = right - left
    content_height = bottom - top
    min_area = (content_width * content_height) * 0.6  # Reduced from 0.8
    max_area = img_area * 0.99  # Increased from 0.98

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour, 0.04 * peri, True
        )  # More lenient approximation
        if len(approx) >= 4:  # Removed upper bound check
            filtered_contours.append(contour)

    if not filtered_contours:
        # Fall back to content boundaries if no suitable contours found
        x, y = left, top
        w, h = right - left, bottom - top
        # Create a rectangular contour
        best_contour = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        ).reshape(-1, 1, 2)
    else:
        # Find best contour with more weight on content coverage
        best_contour = None
        best_score = float("inf")

        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Adjust scoring to favor content coverage
            boundary_score = (
                abs(x - left)
                + abs((x + w) - right)
                + abs(y - top)
                + abs((y + h) - bottom)
            ) / (w + h)

            area = cv2.contourArea(contour)
            rect_area = w * h
            coverage_score = 1 - (area / rect_area)

            # More emphasis on boundary matching
            score = boundary_score * 0.8 + coverage_score * 0.2

            if score < best_score:
                best_score = score
                best_contour = contour

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(best_contour)

    # Apply margin with more aggressive content-aware adjustments
    if margin >= 0:
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
    else:
        margin = abs(margin)
        # Analyze wider strips for better density estimation
        analyze_width = max(margin * 2, 20)

        left_density = np.mean(gray[y : y + h, x : x + analyze_width])
        right_density = np.mean(gray[y : y + h, x + w - analyze_width : x + w])
        top_density = np.mean(gray[y : y + analyze_width, x : x + w])
        bottom_density = np.mean(gray[y + h - analyze_width : y + h, x : x + w])

        # Dynamic cropping factors based on density differences
        left_factor = 1.5 if left_density > 200 else 0.8
        right_factor = 1.5 if right_density > 200 else 0.8
        top_factor = 1.5 if top_density > 200 else 0.8
        bottom_factor = 1.5 if bottom_density > 200 else 0.8

        # Apply asymmetric cropping
        x = min(x + int(margin * left_factor), x + w - 1)
        w = max(w - int(margin * (left_factor + right_factor)), 1)
        y = min(y + int(margin * top_factor), y + h - 1)
        h = max(h - int(margin * (top_factor + bottom_factor)), 1)

    # Crop the image
    cropped = img[y : y + h, x : x + w]

    # Create output path if not specified
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        if margin != -1:
            suffix = f"_frame_m{margin}"
        else:
            suffix = "_frame"
        output_path = f"{base}{suffix}{ext}"

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
        return autofix(image_path, output_path, border_percent=border_percent)
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
            result = autofix(image_path, output_path, border_percent=border_percent)
            if output_path is None:
                base, ext = os.path.splitext(image_path)
                suffix = (
                    f"_auto_border_b{border_percent}"
                    if border_percent != 0
                    else "_auto_border"
                )
                new_path = f"{base}{suffix}{ext}"
                os.rename(result, new_path)
                return new_path
            return result
