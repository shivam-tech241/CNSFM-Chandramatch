"""Geometric data model and coordinate handling for Crater Neighborhood Structure Features (CNSF).

Chunk 8 Implementation:
- CraterFeature dataclass with explicit coordinate separation (pixel x/y vs lon/lat vs row/col).
- Spherical and planar distance functions on the lunar surface.
- AngularStructure dataclass parameterizing the triangle formed by CC and two NCs (Section 2.2.3).
- Coordinate conversion utilities strictly preserving sensor coordinate conventions.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# IAU standard mean lunar radius in meters
LUNAR_RADIUS_METERS: float = 1737400.0


@dataclass
class CraterFeature:
    """Explicit data representation of an individual detected impact crater.

    Coordinate conventions:
    - image_x: horizontal pixel coordinate (column in image raster, 0-indexed, float).
    - image_y: vertical pixel coordinate (row/line in image raster, 0-indexed, float).
    - longitude: selenographic East longitude in degrees [-180, 180] or [0, 360].
    - latitude: selenographic North latitude in degrees [-90, 90].
    - diameter_px: measured crater diameter in image pixels.
    - diameter_m: physical crater diameter in meters (diameter_px * gsd_m).
    - radius_px: measured crater radius in pixels (diameter_px / 2.0).
    - confidence: detection confidence score in [0.0, 1.0].
    - sensor: sensor name string ("TMC-2", "OHRC", "SYNTHETIC").
    - detection_id: unique integer or string identifier within the sensor feature set.
    - scale: representation scale string (e.g. "native", "coarse", "5.40m/px").
    """

    detection_id: Union[int, str]
    sensor: str
    image_x: float
    image_y: float
    longitude: float
    latitude: float
    diameter_px: float
    diameter_m: float
    radius_px: float
    confidence: float
    scale: str = "native"

    def __post_init__(self):
        self.image_x = float(self.image_x)
        self.image_y = float(self.image_y)
        self.longitude = float(self.longitude)
        self.latitude = float(self.latitude)
        self.diameter_px = float(self.diameter_px)
        self.diameter_m = float(self.diameter_m)
        self.radius_px = float(self.radius_px)
        self.confidence = float(self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CraterFeature":
        return cls(**d)


@dataclass(frozen=True)
class AngularStructure:
    """Geometric parameterization of an angular structure within a CNSF.

    Formed by the central crater (CC) and two nearby craters (NC1, NC2) in a clockwise direction.
    Corresponds directly to Section 2.2.3 (Figure 4 and Equations 7–8) of Xie et al. (2025).

    Parameters:
    - cc_id: Central crater ID
    - nc1_id: First nearby crater ID (starting edge)
    - nc2_id: Second nearby crater ID (terminal edge)
    - angle_deg: Angle beta in degrees [0, 360) formed at CC from CC->NC1 to CC->NC2 clockwise
    - dist1_m: Side length S1 (distance from CC to NC1 in meters)
    - dist2_m: Side length S2 (distance from CC to NC2 in meters)
    - diam0_m: Central crater diameter phi_0 in meters
    - diam1_m: NC1 diameter phi_1 in meters
    - diam2_m: NC2 diameter phi_2 in meters
    - dist_ratio: S2 / S1
    - diam_ratio1: phi_1 / phi_0
    - diam_ratio2: phi_2 / phi_0
    """

    cc_id: Union[int, str]
    nc1_id: Union[int, str]
    nc2_id: Union[int, str]
    angle_deg: float
    dist1_m: float
    dist2_m: float
    diam0_m: float
    diam1_m: float
    diam2_m: float
    dist_ratio: float
    diam_ratio1: float
    diam_ratio2: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_euclidean_distance_m(
    c1: CraterFeature,
    c2: CraterFeature,
    gsd_m: Optional[float] = None,
) -> float:
    """Compute physical Euclidean distance between two craters in meters.

    If gsd_m is provided, computes planar Euclidean distance from image_x and image_y.
    If gsd_m is None and selenographic coordinates are valid and non-identical, computes great-circle distance.
    Otherwise falls back to pixel distance.

    Args:
        c1: First CraterFeature.
        c2: Second CraterFeature.
        gsd_m: Optional GSD in meters per pixel.

    Returns:
        Physical ground distance in meters.
    """
    if gsd_m is not None:
        dx = (c1.image_x - c2.image_x) * gsd_m
        dy = (c1.image_y - c2.image_y) * gsd_m
        return float(math.hypot(dx, dy))

    has_geo = not (np.isnan(c1.longitude) or np.isnan(c1.latitude) or np.isnan(c2.longitude) or np.isnan(c2.latitude))
    if has_geo and (c1.longitude != c2.longitude or c1.latitude != c2.latitude):
        return compute_selenographic_distance_m(c1.longitude, c1.latitude, c2.longitude, c2.latitude)

    dx = c1.image_x - c2.image_x
    dy = c1.image_y - c2.image_y
    return float(math.hypot(dx, dy))


def compute_selenographic_distance_m(
    lon1_deg: float,
    lat1_deg: float,
    lon2_deg: float,
    lat2_deg: float,
    lunar_radius_m: float = LUNAR_RADIUS_METERS,
) -> float:
    """Compute great-circle distance on the Moon between two selenographic coordinates via Haversine.

    Args:
        lon1_deg: Longitude of point 1 in degrees.
        lat1_deg: Latitude of point 1 in degrees.
        lon2_deg: Longitude of point 2 in degrees.
        lat2_deg: Latitude of point 2 in degrees.
        lunar_radius_m: Mean lunar radius in meters (default: 1737400.0 m).

    Returns:
        Great-circle distance in meters.
    """
    phi1 = math.radians(lat1_deg)
    phi2 = math.radians(lat2_deg)
    dphi = math.radians(lat2_deg - lat1_deg)
    dlambda = math.radians(lon2_deg - lon1_deg)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(lunar_radius_m * c)


def compute_clockwise_angle_deg(
    x_c: float,
    y_c: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Compute the clockwise angle beta from vector (CC -> NC1) to vector (CC -> NC2) in degrees.

    Clockwise angle beta in [0, 360) deg per Section 2.2.3 and Figure 4.
    In standard image coordinates (x right, y down), the clockwise direction corresponds
    to increasing standard mathematical angle in image space.

    Args:
        x_c, y_c: Central crater center coordinates.
        x1, y1: First neighbor (starting edge) coordinates.
        x2, y2: Second neighbor (terminal edge) coordinates.

    Returns:
        Angle beta in degrees in [0.0, 360.0).
    """
    v1_x = x1 - x_c
    v1_y = y1 - y_c
    v2_x = x2 - x_c
    v2_y = y2 - y_c

    theta1 = math.atan2(v1_y, v1_x)
    theta2 = math.atan2(v2_y, v2_x)

    diff_rad = (theta2 - theta1) % (2.0 * math.pi)
    diff_deg = math.degrees(diff_rad) % 360.0
    return float(diff_deg)
