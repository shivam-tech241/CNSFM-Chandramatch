"""Spatial coverage, distribution, and geometric dispersion metrics (Chunk 9).

Provides:
- SpatialCoverageMetrics: Dataclass recording bounding box, hull area, cell occupancy, and spacing.
- compute_spatial_coverage: Evaluates whether matches are broadly distributed or tightly clustered.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import ConvexHull, KDTree


@dataclass
class SpatialCoverageMetrics:
    """Comprehensive spatial coverage and distribution metrics for a point set."""

    num_points: int
    bbox_x_min_m: float
    bbox_x_max_m: float
    bbox_y_min_m: float
    bbox_y_max_m: float
    bbox_width_m: float
    bbox_height_m: float
    bbox_area_m2: float
    convex_hull_area_m2: float
    occupied_cells: int
    total_grid_cells: int
    cell_occupancy_ratio: float
    mean_nearest_neighbor_dist_m: float
    median_nearest_neighbor_dist_m: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_points": self.num_points,
            "bbox_x_min_m": round(self.bbox_x_min_m, 2),
            "bbox_x_max_m": round(self.bbox_x_max_m, 2),
            "bbox_y_min_m": round(self.bbox_y_min_m, 2),
            "bbox_y_max_m": round(self.bbox_y_max_m, 2),
            "bbox_width_m": round(self.bbox_width_m, 2),
            "bbox_height_m": round(self.bbox_height_m, 2),
            "bbox_area_m2": round(self.bbox_area_m2, 1),
            "convex_hull_area_m2": round(self.convex_hull_area_m2, 1),
            "occupied_cells": self.occupied_cells,
            "total_grid_cells": self.total_grid_cells,
            "cell_occupancy_ratio": round(self.cell_occupancy_ratio, 4),
            "mean_nearest_neighbor_dist_m": round(self.mean_nearest_neighbor_dist_m, 2),
            "median_nearest_neighbor_dist_m": round(self.median_nearest_neighbor_dist_m, 2),
        }


def compute_spatial_coverage(
    points: np.ndarray,
    scene_width_m: float = 1080.0,
    scene_height_m: float = 1080.0,
    grid_cells_per_axis: int = 4,
) -> SpatialCoverageMetrics:
    """Compute spatial coverage and dispersion metrics on a set of points (e.g. geometric inliers).

    Args:
        points: (N, 2) array of coordinates in meters.
        scene_width_m: Extent of the benchmark scene width in meters (default: 1080m).
        scene_height_m: Extent of the benchmark scene height in meters (default: 1080m).
        grid_cells_per_axis: Number of spatial grid subdivisions along each axis (default: 4 -> 16 cells).

    Returns:
        Populated SpatialCoverageMetrics instance.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    total_cells = grid_cells_per_axis * grid_cells_per_axis

    if n == 0:
        return SpatialCoverageMetrics(
            num_points=0,
            bbox_x_min_m=0.0,
            bbox_x_max_m=0.0,
            bbox_y_min_m=0.0,
            bbox_y_max_m=0.0,
            bbox_width_m=0.0,
            bbox_height_m=0.0,
            bbox_area_m2=0.0,
            convex_hull_area_m2=0.0,
            occupied_cells=0,
            total_grid_cells=total_cells,
            cell_occupancy_ratio=0.0,
            mean_nearest_neighbor_dist_m=0.0,
            median_nearest_neighbor_dist_m=0.0,
        )

    # 1. Bounding box
    x_min = float(np.min(pts[:, 0]))
    x_max = float(np.max(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    y_max = float(np.max(pts[:, 1]))
    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)
    bbox_area = w * h

    # 2. Convex Hull Area
    if n >= 3:
        try:
            hull = ConvexHull(pts)
            hull_area = float(hull.volume)  # For 2D, hull.volume is the 2D area
        except Exception:
            hull_area = 0.0
    else:
        hull_area = 0.0

    # 3. Grid Cell Occupancy
    # Translate points so minimum bounds start at 0 if coordinates are centered
    pts_offset_x = pts[:, 0] - x_min
    pts_offset_y = pts[:, 1] - y_min
    span_x = max(scene_width_m, w, 1.0)
    span_y = max(scene_height_m, h, 1.0)

    cell_w = span_x / grid_cells_per_axis
    cell_h = span_y / grid_cells_per_axis

    occupied_set = set()
    for pt in pts_offset_x:
        pass  # Just iterate pairs
    for i in range(n):
        col = int(min(grid_cells_per_axis - 1, max(0, math.floor(pts_offset_x[i] / cell_w))))
        row = int(min(grid_cells_per_axis - 1, max(0, math.floor(pts_offset_y[i] / cell_h))))
        occupied_set.add((row, col))

    occupied_cells = len(occupied_set)
    cell_occupancy_ratio = float(occupied_cells / total_cells)

    # 4. Nearest Neighbor Distances
    if n >= 2:
        kdtree = KDTree(pts)
        dists, _ = kdtree.query(pts, k=2)  # k=2 returns distance to self (0.0) and nearest neighbor
        nn_dists = dists[:, 1]
        mean_nn = float(np.mean(nn_dists))
        median_nn = float(np.median(nn_dists))
    else:
        mean_nn = 0.0
        median_nn = 0.0

    return SpatialCoverageMetrics(
        num_points=n,
        bbox_x_min_m=x_min,
        bbox_x_max_m=x_max,
        bbox_y_min_m=y_min,
        bbox_y_max_m=y_max,
        bbox_width_m=w,
        bbox_height_m=h,
        bbox_area_m2=bbox_area,
        convex_hull_area_m2=hull_area,
        occupied_cells=occupied_cells,
        total_grid_cells=total_cells,
        cell_occupancy_ratio=cell_occupancy_ratio,
        mean_nearest_neighbor_dist_m=mean_nn,
        median_nearest_neighbor_dist_m=median_nn,
    )
