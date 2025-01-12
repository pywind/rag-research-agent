"""Root package for the project."""

# Import shared modules to make them accessible without the src prefix
from .shared import retrieval  # Import shared modules here
from .shared import configuration  # Add any other shared modules you need

# You can also define __all__ if you want to control what gets imported with 'from src import *'
__all__ = ["retrieval", "configuration"] 