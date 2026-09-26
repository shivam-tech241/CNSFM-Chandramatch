"""Geographic metadata, lunar geodesic calculations, footprint analysis, and ground grid readers."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

# Mean volumetric radius of the Moon in meters (IAU standard)
MOON_RADIUS_METERS: float = 1737400.0


def calculate_spherical_polygon_area_m2(
    poly: Polygon, radius: float = MOON_RADIUS_METERS
) -> float:
    """Calculate the exact geodesic area of a polygon on a spherical body in square meters.

    Uses spherical excess / line integral over latitude and longitude.

    Args:
        poly: Shapely Polygon with (longitude, latitude) coordinates in degrees.
        radius: Body radius in meters (default: Moon radius = 1,737,400 m).

    Returns:
        Area in square meters.
    """
    if poly.is_empty:
        return 0.0

    coords = np.array(poly.exterior.coords)
    if len(coords) < 3:
        return 0.0

    rad_lons = np.radians(coords[:, 0])
    rad_lats = np.radians(coords[:, 1])

    # Spherical polygon line integral formula
    area = 0.0
    for i in range(len(rad_lons) - 1):
        dlon = rad_lons[i + 1] - rad_lons[i]
        area += dlon * (np.sin(rad_lats[i]) + np.sin(rad_lats[i + 1])) / 2.0

    return float(abs(area) * (radius**2))


@dataclass(frozen=True)
class CornerCoordinates:
    """Four-corner bounding coordinates in degrees."""

    upper_left_lat: float
    upper_left_lon: float
    upper_right_lat: float
    upper_right_lon: float
    lower_left_lat: float
    lower_left_lon: float
    lower_right_lat: float
    lower_right_lon: float

    def to_polygon(self) -> Polygon:
        """Create a closed Shapely Polygon in (lon, lat) order: UL -> UR -> LR -> LL -> UL."""
        coords = [
            (self.upper_left_lon, self.upper_left_lat),
            (self.upper_right_lon, self.upper_right_lat),
            (self.lower_right_lon, self.lower_right_lat),
            (self.lower_left_lon, self.lower_left_lat),
            (self.upper_left_lon, self.upper_left_lat),
        ]
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly


@dataclass
class GeographicMetadata:
    """Comprehensive geographic and illumination metadata for a lunar orbiter product."""

    sensor_name: str
    xml_path: Path
    system_corners: CornerCoordinates
    refined_corners: Optional[CornerCoordinates]
    active_corners: CornerCoordinates
    projection: str
    pixel_resolution_m: float
    solar_incidence_deg: float
    sun_azimuth_deg: float
    sun_elevation_deg: float
    lines: int
    samples: int
    footprint_area_deg2: float
    footprint_area_km2: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_name": self.sensor_name,
            "xml_path": str(self.xml_path),
            "projection": self.projection,
            "pixel_resolution_m": self.pixel_resolution_m,
            "solar_incidence_deg": self.solar_incidence_deg,
            "sun_azimuth_deg": self.sun_azimuth_deg,
            "sun_elevation_deg": self.sun_elevation_deg,
            "lines": self.lines,
            "samples": self.samples,
            "footprint_area_deg2": self.footprint_area_deg2,
            "footprint_area_km2": self.footprint_area_km2,
            "corners": asdict(self.active_corners),
        }


def _extract_corners_from_parent(parent_elem: ET.Element) -> Optional[CornerCoordinates]:
    """Extract corner coordinates from an XML element if all 8 coordinates exist."""
    required_tags = [
        "upper_left_latitude",
        "upper_left_longitude",
        "upper_right_latitude",
        "upper_right_longitude",
        "lower_left_latitude",
        "lower_left_longitude",
        "lower_right_latitude",
        "lower_right_longitude",
    ]
    vals: Dict[str, float] = {}
    for tag in required_tags:
        elem = parent_elem.find(f".//{tag}")
        if elem is None or not elem.text or not elem.text.strip():
            return None
        try:
            vals[tag] = float(elem.text.strip())
        except ValueError:
            return None

    return CornerCoordinates(
        upper_left_lat=vals["upper_left_latitude"],
        upper_left_lon=vals["upper_left_longitude"],
        upper_right_lat=vals["upper_right_latitude"],
        upper_right_lon=vals["upper_right_longitude"],
        lower_left_lat=vals["lower_left_latitude"],
        lower_left_lon=vals["lower_left_longitude"],
        lower_right_lat=vals["lower_right_latitude"],
        lower_right_lon=vals["lower_right_longitude"],
    )


def parse_geographic_metadata(
    xml_path: Path | str, sensor_name: str = "GENERIC"
) -> GeographicMetadata:
    """Parse PDS4 XML label to extract geometric coordinates, illumination, and footprint.

    Args:
        xml_path: Path to the PDS4 XML file.
        sensor_name: Sensor identifier string.

    Returns:
        Populated GeographicMetadata instance.

    Raises:
        FileNotFoundError: If XML file does not exist.
        ValueError: If essential coordinate fields are missing.
    """
    path = Path(xml_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDS4 XML file not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    # Strip XML namespaces for uniform querying
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    # Corner coordinates
    system_elem = root.find(".//System_Level_Coordinates")
    system_corners = _extract_corners_from_parent(system_elem) if system_elem is not None else None

    refined_elem = root.find(".//Refined_Corner_Coordinates")
    refined_corners = _extract_corners_from_parent(refined_elem) if refined_elem is not None else None

    # Preferred active corners: refined if available, else system level
    active_corners = refined_corners or system_corners
    if active_corners is None:
        raise ValueError(f"No valid corner coordinates found in PDS4 XML: {path}")

    # Illumination
    sun_azimuth_elem = root.find(".//sun_azimuth")
    sun_azimuth = float(sun_azimuth_elem.text.strip()) if sun_azimuth_elem is not None and sun_azimuth_elem.text else 0.0

    sun_elev_elem = root.find(".//sun_elevation")
    sun_elevation = float(sun_elev_elem.text.strip()) if sun_elev_elem is not None and sun_elev_elem.text else 0.0

    solar_inc_elem = root.find(".//solar_incidence")
    solar_incidence = float(solar_inc_elem.text.strip()) if solar_inc_elem is not None and solar_inc_elem.text else 0.0

    # Projection
    proj_elem = root.find(".//projection")
    projection = proj_elem.text.strip() if proj_elem is not None and proj_elem.text else "Selenographic"

    # Pixel resolution
    res_elem = root.find(".//pixel_resolution")
    pixel_res = float(res_elem.text.strip()) if res_elem is not None and res_elem.text else 0.0

    # Dimensions
    lines: int = 0
    samples: int = 0
    for axis in root.findall(".//Axis_Array"):
        name_elem = axis.find("axis_name")
        elems_elem = axis.find("elements")
        if name_elem is not None and elems_elem is not None and name_elem.text and elems_elem.text:
            axis_name = name_elem.text.strip().lower()
            if axis_name == "line":
                lines = int(elems_elem.text.strip())
            elif axis_name == "sample":
                samples = int(elems_elem.text.strip())

    # Footprint polygon and areas
    poly = active_corners.to_polygon()
    area_deg2 = float(poly.area)
    area_km2 = calculate_spherical_polygon_area_m2(poly) / 1e6

    return GeographicMetadata(
        sensor_name=sensor_name,
        xml_path=path,
        system_corners=system_corners if system_corners is not None else active_corners,
        refined_corners=refined_corners,
        active_corners=active_corners,
        projection=projection,
        pixel_resolution_m=pixel_res,
        solar_incidence_deg=solar_incidence,
        sun_azimuth_deg=sun_azimuth,
        sun_elevation_deg=sun_elevation,
        lines=lines,
        samples=samples,
        footprint_area_deg2=area_deg2,
        footprint_area_km2=area_km2,
    )


@dataclass
class OverlapMetrics:
    """Quantitative geographic overlap metrics between two footprints."""

    sensor1_name: str
    sensor2_name: str
    sensor1_area_km2: float
    sensor2_area_km2: float
    intersection_area_km2: float
    overlap_ratio_sensor1: float
    overlap_ratio_sensor2: float
    is_overlapping: bool
    sensor1_contained_in_sensor2: bool
    sensor2_contained_in_sensor1: bool
    intersection_bounds_lon_lat: Tuple[float, float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor1": self.sensor1_name,
            "sensor2": self.sensor2_name,
            "sensor1_area_km2": self.sensor1_area_km2,
            "sensor2_area_km2": self.sensor2_area_km2,
            "intersection_area_km2": self.intersection_area_km2,
            "overlap_ratio_sensor1": self.overlap_ratio_sensor1,
            "overlap_ratio_sensor2": self.overlap_ratio_sensor2,
            "is_overlapping": self.is_overlapping,
            "sensor1_contained_in_sensor2": self.sensor1_contained_in_sensor2,
            "sensor2_contained_in_sensor1": self.sensor2_contained_in_sensor1,
            "intersection_bounds_lon_lat": list(self.intersection_bounds_lon_lat),
        }


def compute_footprint_overlap(
    meta1: GeographicMetadata, meta2: GeographicMetadata
) -> Tuple[OverlapMetrics, Polygon]:
    """Compute geographic intersection and relative overlap metrics between two products.

    Args:
        meta1: First product metadata (e.g. OHRC).
        meta2: Second product metadata (e.g. TMC-2).

    Returns:
        Tuple of (OverlapMetrics, Shapely intersection Polygon).
    """
    poly1 = meta1.active_corners.to_polygon()
    poly2 = meta2.active_corners.to_polygon()

    if not poly1.intersects(poly2):
        metrics = OverlapMetrics(
            sensor1_name=meta1.sensor_name,
            sensor2_name=meta2.sensor_name,
            sensor1_area_km2=meta1.footprint_area_km2,
            sensor2_area_km2=meta2.footprint_area_km2,
            intersection_area_km2=0.0,
            overlap_ratio_sensor1=0.0,
            overlap_ratio_sensor2=0.0,
            is_overlapping=False,
            sensor1_contained_in_sensor2=False,
            sensor2_contained_in_sensor1=False,
            intersection_bounds_lon_lat=(0.0, 0.0, 0.0, 0.0),
        )
        return metrics, Polygon()

    intersection_geom = poly1.intersection(poly2)
    if isinstance(intersection_geom, Polygon):
        inter_poly = intersection_geom
    else:
        # MultiPolygon or boundary geometry
        inter_poly = Polygon(intersection_geom.convex_hull.exterior.coords)

    inter_area_km2 = calculate_spherical_polygon_area_m2(inter_poly) / 1e6

    ratio1 = inter_area_km2 / meta1.footprint_area_km2 if meta1.footprint_area_km2 > 0 else 0.0
    ratio2 = inter_area_km2 / meta2.footprint_area_km2 if meta2.footprint_area_km2 > 0 else 0.0

    # Containment check
    contained1_in_2 = bool(poly2.contains(poly1)) or ratio1 >= 0.999
    contained2_in_1 = bool(poly1.contains(poly2)) or ratio2 >= 0.999

    bounds = inter_poly.bounds  # (minx, miny, maxx, maxy) -> (min_lon, min_lat, max_lon, max_lat)

    metrics = OverlapMetrics(
        sensor1_name=meta1.sensor_name,
        sensor2_name=meta2.sensor_name,
        sensor1_area_km2=meta1.footprint_area_km2,
        sensor2_area_km2=meta2.footprint_area_km2,
        intersection_area_km2=inter_area_km2,
        overlap_ratio_sensor1=ratio1,
        overlap_ratio_sensor2=ratio2,
        is_overlapping=True,
        sensor1_contained_in_sensor2=contained1_in_2,
        sensor2_contained_in_sensor1=contained2_in_1,
        intersection_bounds_lon_lat=bounds,
    )
    return metrics, inter_poly


class GroundGridReader:
    """Reader and inspector for ISRO ground coordinate grid files (_g_grd_d18.csv).

    Maps between (Longitude, Latitude) and image coordinates (Pixel / Scan).
    """

    def __init__(self, csv_path: Path | str, sensor_name: str = "GENERIC"):
        self.csv_path = Path(csv_path)
        self.sensor_name = sensor_name
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Geometry grid CSV not found: {self.csv_path}")

        self._df: Optional[pd.DataFrame] = None
        self._summary: Optional[Dict[str, Any]] = None

    def load(self) -> pd.DataFrame:
        """Load grid records into memory."""
        if self._df is None:
            self._df = pd.read_csv(self.csv_path)
            expected_cols = {"Longitude", "Latitude", "Pixel", "Scan"}
            actual_cols = set(self._df.columns)
            if not expected_cols.issubset(actual_cols):
                raise ValueError(
                    f"Geometry CSV {self.csv_path} missing required columns: {expected_cols - actual_cols}"
                )
        return self._df

    def get_summary(self) -> Dict[str, Any]:
        """Compute statistical summary of grid spacing and geographic coverage."""
        if self._summary is not None:
            return self._summary

        df = self.load()
        unique_pixels = sorted(df["Pixel"].unique())
        unique_scans = sorted(df["Scan"].unique())

        pixel_steps = sorted(list(set(int(b - a) for a, b in zip(unique_pixels[:-1], unique_pixels[1:]))))
        scan_steps = sorted(list(set(int(b - a) for a, b in zip(unique_scans[:-1], unique_scans[1:]))))

        self._summary = {
            "sensor": self.sensor_name,
            "csv_path": str(self.csv_path),
            "record_count": int(len(df)),
            "pixel_min": int(df["Pixel"].min()),
            "pixel_max": int(df["Pixel"].max()),
            "pixel_unique_count": int(len(unique_pixels)),
            "pixel_step_sizes": pixel_steps,
            "scan_min": int(df["Scan"].min()),
            "scan_max": int(df["Scan"].max()),
            "scan_unique_count": int(len(unique_scans)),
            "scan_step_sizes": scan_steps,
            "lon_min": float(df["Longitude"].min()),
            "lon_max": float(df["Longitude"].max()),
            "lat_min": float(df["Latitude"].min()),
            "lat_max": float(df["Latitude"].max()),
        }
        return self._summary

    def find_pixel_bounding_box(
        self, min_lon: float, max_lon: float, min_lat: float, max_lat: float
    ) -> Optional[Dict[str, int]]:
        """Find the enclosing (Scan, Pixel) bounding box for a given geographic bounding box.

        Returns:
            Dict with 'scan_start', 'scan_end', 'pixel_start', 'pixel_end' or None if no overlap.
        """
        df = self.load()
        mask = (
            (df["Longitude"] >= min_lon)
            & (df["Longitude"] <= max_lon)
            & (df["Latitude"] >= min_lat)
            & (df["Latitude"] <= max_lat)
        )
        subset = df[mask]
        if len(subset) == 0:
            return None

        return {
            "scan_start": int(subset["Scan"].min()),
            "scan_end": int(subset["Scan"].max()),
            "pixel_start": int(subset["Pixel"].min()),
            "pixel_end": int(subset["Pixel"].max()),
            "matching_grid_points": int(len(subset)),
        }
