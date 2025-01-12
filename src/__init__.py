"""Root package for the project."""

# Import shared modules to make them accessible without the src prefix
from .shared import (
    configuration,  # Add any other shared modules you need
    retrieval,  # Import shared modules here
)

__all__ = ["retrieval", "configuration"] 