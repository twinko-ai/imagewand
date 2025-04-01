from PIL import Image, ImageOps

from imagewand import add_custom_filter, apply_filter


def my_custom_filter(img, params=None):
    # Example: Create a custom filter that combines grayscale and inversion
    gray_img = ImageOps.grayscale(img).convert("RGB")
    inverted_img = ImageOps.invert(gray_img)
    return inverted_img


# Register the custom filter
add_custom_filter("my_noir_look", my_custom_filter)

# Now use it
if __name__ == "__main__":
    apply_filter("input.jpg", "my_noir_look", "output.jpg")
