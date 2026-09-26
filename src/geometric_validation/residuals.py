"""Residual computation, displacement analysis, and physical metric coordinate conversions (Chunk 9).

Provides:
- lonlat_to_local_planar_m: Converts selenographic (lon, lat) to local tangent plane (X, Y) in meters.
- compute_candidate_displacements: Analyzes initial cross-sensor horizontal separation before modeling.
- compute_residual_statistics: Computes comprehensive statistical distribution metrics on residuals.
"""

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

MOON_RADIUS_M = 1737400.0
BENCHMARK_CENTER_LON = 336.536
BENCHMARK_CENTER_LAT = -3.000


def lonlat_to_local_planar_m(
    lons: np.ndarray,
    lats: np.ndarray,
    center_lon: float = BENCHMARK_CENTER_LON,
    center_lat: float = BENCHMARK_CENTER_LAT,
    radius_m: float = MOON_RADIUS_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert selenographic (lon, lat) coordinates into local metric planar tangent coordinates (X, Y).

    Uses an Equirectangular / local tangent projection centered at the benchmark reference point.
    In this coordinate space, 1.0 unit = 1.0 physical meter on the lunar surface.

    Args:
        lons: Array or list of longitudes in degrees [0, 360).
        lats: Array or list of latitudes in degrees [-90, +90].
        center_lon: Projection center longitude.
        center_lat: Projection center latitude.
        radius_m: Volumetric mean lunar radius in meters.

    Returns:
        Tuple of (X_m, Y_m) numpy arrays in meters.
    """
    lons_arr = np.asarray(lons, dtype=np.float64)
    lats_arr = np.asarray(lats, dtype=np.float64)

    phi_mid_rad = math.radians(center_lat)
    d_lon_rad = np.radians(lons_arr - center_lon)
    d_lat_rad = np.radians(lats_arr - center_lat)

    x_m = radius_m * d_lon_rad * math.cos(phi_mid_rad)
    y_m = radius_m * d_lat_rad

    return x_m, y_m


def compute_candidate_displacements(
    df_candidates: pd.DataFrame,
    center_lon: float = BENCHMARK_CENTER_LON,
    center_lat: float = BENCHMARK_CENTER_LAT,
    radius_m: float = MOON_RADIUS_M,
) -> pd.DataFrame:
    """Compute physical metric horizontal displacement for each candidate crater pair.

    Args:
        df_candidates: DataFrame containing source and target coordinates.
        center_lon: Reference longitude.
        center_lat: Reference latitude.
        radius_m: Mean lunar radius.

    Returns:
        Updated DataFrame with local planar coordinates (src_X_m, src_Y_m, tgt_X_m, tgt_Y_m)
        and initial displacement_m.
    """
    df = df_candidates.copy()

    # Source local planar coordinates (meters)
    src_x_m, src_y_m = lonlat_to_local_planar_m(
        df["source_lon"].values,
        df["source_lat"].values,
        center_lon=center_lon,
        center_lat=center_lat,
        radius_m=radius_m,
    )
    # Target local planar coordinates (meters)
    tgt_x_m, tgt_y_m = lonlat_to_local_planar_m(
        df["target_lon"].values,
        df["target_lat"].values,
        center_lon=center_lon,
        center_lat=center_lat,
        radius_m=radius_m,
    )

    dx_m = tgt_x_m - src_x_m
    dy_m = tgt_y_m - src_y_m
    disp_m = np.hypot(dx_m, dy_m)

    df["src_X_m"] = np.round(src_x_m, 2)
    df["src_Y_m"] = np.round(src_y_m, 2)
    df["tgt_X_m"] = np.round(tgt_x_m, 2)
    df["tgt_Y_m"] = np.round(tgt_y_m, 2)
    df["displacement_m"] = np.round(disp_m, 2)

    return df


def compute_residual_statistics(residuals_m: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive statistical metrics for residual distributions.

    Args:
        residuals_m: 1D array of residuals in meters.

    Returns:
        Dictionary of summary statistics.
    """
    arr = np.asarray(residuals_m, dtype=np.float64)
    if len(arr) == 0:
        return {
            "count": 0,
            "mean_m": 0.0,
            "median_m": 0.0,
            "rmse_m": 0.0,
            "std_m": 0.0,
            "p25_m": 0.0,
            "p75_m": 0.0,
            "p95_m": 0.0,
            "max_m": 0.0,
            "min_m": 0.0,
        }

    return {
        "count": int(len(arr)),
        "mean_m": round(float(np.mean(arr)), 3),
        "median_m": round(float(np.median(arr)), 3),
        "rmse_m": round(float(np.sqrt(np.mean(arr**2))), 3),
        "std_m": round(float(np.std(arr)), 3),
        "p25_m": round(float(np.percentile(arr, 25)), 3),
        "p75_m": round(float(np.percentile(arr, 75)), 3),
        "p95_m": round(float(np.percentile(arr, 95)), 3),
        "max_m": round(float(np.max(arr)), 3),
        "min_m": round(float(np.min(arr)), 3),
    }
