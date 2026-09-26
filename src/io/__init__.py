"""I/O package for memory-efficient and region-based loading of lunar orbital imagery."""

from .geospatial import (
    CornerCoordinates,
    GeographicMetadata,
    GroundGridReader,
    OverlapMetrics,
    calculate_spherical_polygon_area_m2,
    compute_footprint_overlap,
    parse_geographic_metadata,
)
from .ground_grid import (
    GridSummary,
    GroundGrid,
    OverlapResult,
    ValidationMetrics,
    compute_sensor_overlap,
    geo_to_pixel,
    pixel_to_geo,
)
from .metadata import (
    PDS4ImageMetadata,
    check_metadata_discrepancies,
    parse_pds4_xml,
    validate_file_size,
)
from .raster_reader import BaseRasterReader, OHRCRasterReader, TMC2RasterReader
from .region_loader import DatasetLoader, get_default_loader, read_region

__all__ = [
    "BaseRasterReader",
    "OHRCRasterReader",
    "TMC2RasterReader",
    "PDS4ImageMetadata",
    "DatasetLoader",
    "get_default_loader",
    "read_region",
    "parse_pds4_xml",
    "validate_file_size",
    "check_metadata_discrepancies",
    "CornerCoordinates",
    "GeographicMetadata",
    "OverlapMetrics",
    "GroundGridReader",
    "calculate_spherical_polygon_area_m2",
    "compute_footprint_overlap",
    "parse_geographic_metadata",
    "GroundGrid",
    "GridSummary",
    "ValidationMetrics",
    "OverlapResult",
    "compute_sensor_overlap",
    "geo_to_pixel",
    "pixel_to_geo",
]
