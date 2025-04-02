import cv2
import numpy as np

from imagewand import add_custom_filter, apply_filter


def my_custom_filter(img: np.ndarray, params: dict = None) -> np.ndarray:
    """Apply a custom filter to the image.
    
    Args:
        img: Input image as numpy array
        params: Optional parameters for the filter
        
    Returns:
        Filtered image as numpy array
    """
    if params is None:
        params = {}
    
    intensity = params.get("intensity", 1.0)
    
    # Apply some basic image processing
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    enhanced = cv2.addWeighted(img, intensity, blurred, 1 - intensity, 0)
    
    return enhanced


# Register the custom filter
add_custom_filter("my_noir_look", my_custom_filter)

# Now use it
if __name__ == "__main__":
    apply_filter("input.jpg", "my_noir_look", "output.jpg")
