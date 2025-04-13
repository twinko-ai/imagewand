"""
ImageWand - Image processing toolkit
"""

import warnings

# Suppress Numba warnings about FNV hashing
warnings.filterwarnings("ignore", message="FNV hashing is not implemented in Numba")

from .autocrop import autocrop
from .filters import (
    add_custom_filter,
    apply_filter,
    apply_filters,
    batch_apply_filters,
    list_filters,
)
from .pdf2img import pdf_to_images
from .resize import resize_image
from .rmbg import batch_remove_background, remove_background

__all__ = [
    "resize_image",
    "pdf_to_images",
    "autocrop",
    "apply_filter",
    "apply_filters",
    "batch_apply_filters",
    "list_filters",
    "add_custom_filter",
    "remove_background",
    "batch_remove_background",
]
