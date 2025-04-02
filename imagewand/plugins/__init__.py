"""Custom filter plugins for ImageWand.

This module provides custom filter plugins that can be used with ImageWand.
All filters in this module will be automatically registered with the filter system.
"""

# Import any filters you want to expose directly
from .custom_filters import my_custom_filter, my_noir_look

# List of filters that will be automatically registered
__all__ = ["my_custom_filter", "my_noir_look"]
