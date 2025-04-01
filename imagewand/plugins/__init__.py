"""
ImageWand Plugins

This module contains example plugins and custom filters that can be used with ImageWand.
Users can create their own custom filters by following these examples.
"""

# Import any filters you want to expose directly
from .custom_filters import my_noir_look

# List of filters that will be automatically registered
__all__ = ["my_noir_look"]
