"""Preprocessing and image characterization package for Chandrayaan-2 lunar imagery.

CHUNK 5: Non-destructive intensity normalization, OpenCV CLAHE representation,
directional Sobel gradients, shadow/dark-region diagnostics, and cross-sensor
resolution scale analysis.
"""

from .characterization import (
    ImageStatistics,
    compute_image_statistics,
    compute_streaming_image_statistics,
)
from .enhancement import (
    apply_clahe,
    compute_dark_region_diagnostics,
    compute_gradients,
)
from .extractor import (
    OverlapDataExtractor,
    SourceWindowMetadata,
)
from .normalization import percentile_normalize
from .scale import compute_scale_diagnostic

__all__ = [
    "ImageStatistics",
    "compute_image_statistics",
    "compute_streaming_image_statistics",
    "percentile_normalize",
    "apply_clahe",
    "compute_gradients",
    "compute_dark_region_diagnostics",
    "compute_scale_diagnostic",
    "OverlapDataExtractor",
    "SourceWindowMetadata",
]
