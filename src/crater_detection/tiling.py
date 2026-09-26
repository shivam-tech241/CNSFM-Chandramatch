"""Tiling strategy and duplicate suppression for crater detection on large lunar images.

Features:
- Fixed-size overlapping tile grid generation.
- Mapping of tile-relative coordinates to global image coordinates.
- Fast spatial-grid duplicate suppression (Non-Maximum Suppression) across tile seams.
- Tiled inference driver for any BaseCraterDetector.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.crater_detection.base import BaseCraterDetector, CraterDetection


@dataclass
class Tile:
    """Represents a spatial tile within a larger raster."""

    tile_id: str
    row_start: int
    row_end: int
    col_start: int
    col_end: int

    @property
    def height(self) -> int:
        return self.row_end - self.row_start

    @property
    def width(self) -> int:
        return self.col_end - self.col_start


def generate_tiles(
    image_shape: Tuple[int, int],
    tile_size: int = 640,
    overlap: int = 64,
) -> List[Tile]:
    """Generate a regular grid of overlapping tiles covering an image.

    Args:
        image_shape: (height, width) of the image.
        tile_size: Size of square tiles in pixels.
        overlap: Overlap in pixels between adjacent tiles.

    Returns:
        List of Tile objects covering the image.
    """
    height, width = image_shape
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError(f"overlap must be in [0, tile_size), got {overlap}")

    stride = tile_size - overlap
    tiles: List[Tile] = []

    if height <= tile_size and width <= tile_size:
        return [Tile(tile_id="tile_r0_c0", row_start=0, row_end=height, col_start=0, col_end=width)]

    row_starts = list(range(0, height, stride))
    if height > tile_size and (height - row_starts[-1]) < tile_size:
        row_starts[-1] = max(0, height - tile_size)
    row_starts = sorted(list(set(row_starts)))

    col_starts = list(range(0, width, stride))
    if width > tile_size and (width - col_starts[-1]) < tile_size:
        col_starts[-1] = max(0, width - tile_size)
    col_starts = sorted(list(set(col_starts)))

    for r_idx, r_start in enumerate(row_starts):
        r_end = min(height, r_start + tile_size)
        for c_idx, c_start in enumerate(col_starts):
            c_end = min(width, c_start + tile_size)
            tile_id = f"tile_r{r_idx}_c{c_idx}"
            tiles.append(
                Tile(
                    tile_id=tile_id,
                    row_start=r_start,
                    row_end=r_end,
                    col_start=c_start,
                    col_end=c_end,
                )
            )

    return tiles


def map_tile_detection_to_global(
    detection: CraterDetection,
    row_start: int,
    col_start: int,
    tile_id: str,
) -> CraterDetection:
    """Translate a detection from tile-local coordinates to global image coordinates."""
    return CraterDetection(
        center_x=float(detection.center_x + col_start),
        center_y=float(detection.center_y + row_start),
        radius_px=float(detection.radius_px),
        diameter_px=float(detection.diameter_px),
        diameter_m=float(detection.diameter_m),
        confidence=float(detection.confidence),
        source_sensor=detection.source_sensor,
        image_region=detection.image_region,
        detector_name=detection.detector_name,
        detector_version=detection.detector_version,
        preprocessing_representation=detection.preprocessing_representation,
        inference_parameters=detection.inference_parameters,
        tile_id=tile_id,
        crater_id=detection.crater_id,
    )


def compute_circle_iou(
    x1: float, y1: float, r1: float,
    x2: float, y2: float, r2: float,
) -> float:
    """Compute the Intersection over Union (IoU) of two circles."""
    d = math.hypot(x1 - x2, y1 - y2)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        r_min = min(r1, r2)
        r_max = max(r1, r2)
        area_min = math.pi * r_min * r_min
        area_max = math.pi * r_max * r_max
        return area_min / area_max

    alpha = 2.0 * math.acos((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1))
    beta = 2.0 * math.acos((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2))
    area_intersect = 0.5 * (
        r1 * r1 * (alpha - math.sin(alpha)) +
        r2 * r2 * (beta - math.sin(beta))
    )
    area_union = math.pi * r1 * r1 + math.pi * r2 * r2 - area_intersect
    return area_intersect / max(area_union, 1e-8)


def suppress_duplicate_detections(
    detections: List[CraterDetection],
    iou_threshold: float = 0.35,
    distance_threshold_ratio: float = 0.40,
    cell_size: float = 128.0,
) -> List[CraterDetection]:
    """Perform fast spatial-grid Non-Maximum Suppression (NMS) on crater detections.

    Suppresses duplicate detections of the same physical crater across tile seams.

    Args:
        detections: List of global CraterDetection objects.
        iou_threshold: Circle IoU above which the lower-confidence detection is suppressed.
        distance_threshold_ratio: Center distance as a fraction of smaller radius below which
                                 detections are considered identical if sizes are similar.
        cell_size: Spatial hash grid cell size in pixels.

    Returns:
        Deduplicated list of CraterDetection objects, sorted by confidence descending.
    """
    if not detections:
        return []

    # Sort descending by confidence
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep: List[CraterDetection] = []
    grid: Dict[Tuple[int, int], List[CraterDetection]] = {}

    for cand in sorted_dets:
        cx1, cy1, r1 = cand.center_x, cand.center_y, cand.radius_px

        # Spatial grid bounds for candidate
        min_gx = int((cx1 - r1) / cell_size)
        max_gx = int((cx1 + r1) / cell_size)
        min_gy = int((cy1 - r1) / cell_size)
        max_gy = int((cy1 + r1) / cell_size)

        is_duplicate = False

        for gx in range(min_gx, max_gx + 1):
            if is_duplicate:
                break
            for gy in range(min_gy, max_gy + 1):
                cell_items = grid.get((gx, gy))
                if not cell_items:
                    continue

                for kept in cell_items:
                    cx2, cy2, r2 = kept.center_x, kept.center_y, kept.radius_px

                    # Rapid bounding box rejection
                    dx = abs(cx1 - cx2)
                    dy = abs(cy1 - cy2)
                    r_sum = r1 + r2
                    if dx >= r_sum or dy >= r_sum:
                        continue

                    d_sq = dx * dx + dy * dy
                    if d_sq >= r_sum * r_sum:
                        continue

                    min_r = min(r1, r2)
                    max_r = max(r1, r2)

                    # Center proximity check
                    if d_sq < (min_r * distance_threshold_ratio) ** 2 and (max_r / max(min_r, 1e-4)) < 2.0:
                        is_duplicate = True
                        break

                    # IoU check
                    iou = compute_circle_iou(cx1, cy1, r1, cx2, cy2, r2)
                    if iou > iou_threshold:
                        is_duplicate = True
                        break

                if is_duplicate:
                    break

        if not is_duplicate:
            keep.append(cand)
            # Register in spatial grid
            c_gx = int(cx1 / cell_size)
            c_gy = int(cy1 / cell_size)
            grid.setdefault((c_gx, c_gy), []).append(cand)

    # Re-assign sequential crater IDs
    for idx, det in enumerate(keep):
        det.crater_id = idx

    return keep


def run_tiled_detection(
    detector: BaseCraterDetector,
    image: np.ndarray,
    gsd_m: float,
    source_sensor: str,
    image_region: str,
    preprocessing_representation: str = "clahe",
    tile_size: int = 640,
    overlap: int = 64,
    iou_threshold: float = 0.35,
    distance_threshold_ratio: float = 0.40,
) -> Tuple[List[CraterDetection], List[CraterDetection], Dict[str, Any]]:
    """Run crater detection over an image using overlapping tiles.

    Args:
        detector: An instance of BaseCraterDetector.
        image: 2D numpy array (grayscale image).
        gsd_m: Ground sample distance in meters.
        source_sensor: Sensor name (e.g. "OHRC" or "TMC-2").
        image_region: Region label.
        preprocessing_representation: Preprocessing method applied.
        tile_size: Width/height of square tile in pixels.
        overlap: Overlap in pixels.
        iou_threshold: NMS IoU threshold.
        distance_threshold_ratio: NMS center distance ratio threshold.

    Returns:
        (raw_detections, deduplicated_detections, tiling_metadata)
    """
    tiles = generate_tiles(image.shape, tile_size=tile_size, overlap=overlap)
    raw_detections: List[CraterDetection] = []
    tile_stats = []

    for tile in tiles:
        tile_patch = image[tile.row_start:tile.row_end, tile.col_start:tile.col_end]
        tile_dets = detector.detect(
            image=tile_patch,
            gsd_m=gsd_m,
            source_sensor=source_sensor,
            image_region=image_region,
            preprocessing_representation=preprocessing_representation,
            tile_id=tile.tile_id,
        )

        global_dets = [
            map_tile_detection_to_global(d, tile.row_start, tile.col_start, tile.tile_id)
            for d in tile_dets
        ]
        raw_detections.extend(global_dets)

        tile_stats.append({
            "tile_id": tile.tile_id,
            "row_start": tile.row_start,
            "row_end": tile.row_end,
            "col_start": tile.col_start,
            "col_end": tile.col_end,
            "num_detections": len(global_dets),
        })

    # Apply fast spatial-grid duplicate suppression (NMS)
    post_nms_detections = suppress_duplicate_detections(
        raw_detections,
        iou_threshold=iou_threshold,
        distance_threshold_ratio=distance_threshold_ratio,
        cell_size=float(max(64.0, detector.max_radius_px * 2.0)),
    )

    tiling_metadata = {
        "num_tiles": len(tiles),
        "tile_size": tile_size,
        "overlap": overlap,
        "raw_detection_count": len(raw_detections),
        "post_nms_count": len(post_nms_detections),
        "duplicates_removed": len(raw_detections) - len(post_nms_detections),
        "tile_details": tile_stats,
    }

    return raw_detections, post_nms_detections, tiling_metadata
