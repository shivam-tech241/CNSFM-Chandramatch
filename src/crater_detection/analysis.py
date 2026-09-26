"""Analysis routines for crater detections: statistics, scaling, sectors, and dark-region diagnostics."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.crater_detection.base import CraterDetection


def compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of numerical values."""
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p5": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
        }
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def compute_detection_statistics(
    detections: List[CraterDetection],
    image_shape: Tuple[int, int],
    gsd_m: float,
    sensor_name: str,
) -> Dict[str, Any]:
    """Compute comprehensive detection statistics for a set of detections."""
    height, width = image_shape
    area_km2 = (height * gsd_m / 1000.0) * (width * gsd_m / 1000.0)

    confs = [d.confidence for d in detections]
    radii_px = [d.radius_px for d in detections]
    diameters_px = [d.diameter_px for d in detections]
    diameters_m = [d.diameter_m for d in detections]

    conf_stats = compute_distribution_stats(confs)
    radius_stats = compute_distribution_stats(radii_px)
    diameter_px_stats = compute_distribution_stats(diameters_px)
    diameter_m_stats = compute_distribution_stats(diameters_m)

    density_per_km2 = len(detections) / max(area_km2, 1e-6)
    density_per_megapixel = (len(detections) / (height * width)) * 1e6

    return {
        "sensor_name": sensor_name,
        "total_detections": len(detections),
        "image_shape": {"height": height, "width": width},
        "gsd_m": gsd_m,
        "ground_area_km2": float(round(area_km2, 4)),
        "spatial_density": {
            "craters_per_km2": float(round(density_per_km2, 2)),
            "craters_per_megapixel": float(round(density_per_megapixel, 2)),
        },
        "confidence_distribution": conf_stats,
        "radius_px_distribution": radius_stats,
        "diameter_px_distribution": diameter_px_stats,
        "diameter_m_distribution": diameter_m_stats,
    }


def compute_sector_statistics(
    detections: List[CraterDetection],
    image_height: int,
    num_sectors: int = 3,
) -> Dict[str, Any]:
    """Divide the raster along the Y axis into sectors (e.g. Top, Center, Bottom) and report statistics."""
    sector_names = ["Top", "Center", "Bottom"] if num_sectors == 3 else [f"Sector_{i}" for i in range(num_sectors)]
    sector_height = image_height / num_sectors

    sectors_data = {}
    for i, name in enumerate(sector_names):
        y_min = i * sector_height
        y_max = (i + 1) * sector_height
        sec_dets = [d for d in detections if y_min <= d.center_y < y_max]

        confs = [d.confidence for d in sec_dets]
        diams_m = [d.diameter_m for d in sec_dets]

        sectors_data[name] = {
            "y_range": [float(round(y_min, 1)), float(round(y_max, 1))],
            "count": len(sec_dets),
            "percentage": float(round((len(sec_dets) / max(len(detections), 1)) * 100.0, 2)),
            "mean_confidence": float(round(np.mean(confs) if confs else 0.0, 4)),
            "mean_diameter_m": float(round(np.mean(diams_m) if diams_m else 0.0, 2)),
            "median_diameter_m": float(round(np.median(diams_m) if diams_m else 0.0, 2)),
        }

    return sectors_data


def compute_dark_region_breakdown(
    detections: List[CraterDetection],
    dark_mask: np.ndarray,
) -> Dict[str, Any]:
    """Compare crater detections inside vs outside the dark-region statistical proxy mask."""
    h, w = dark_mask.shape
    inside_dets = []
    outside_dets = []

    for d in detections:
        ix = int(np.clip(np.round(d.center_x), 0, w - 1))
        iy = int(np.clip(np.round(d.center_y), 0, h - 1))

        # Check center pixel and 3x3 local patch
        y0, y1 = max(0, iy - 1), min(h, iy + 2)
        x0, x1 = max(0, ix - 1), min(w, ix + 2)
        local_mask = dark_mask[y0:y1, x0:x1]

        # If more than half the local patch is marked dark
        if np.mean(local_mask > 0) >= 0.5:
            inside_dets.append(d)
        else:
            outside_dets.append(d)

    total = max(len(detections), 1)

    inside_confs = [d.confidence for d in inside_dets]
    outside_confs = [d.confidence for d in outside_dets]

    inside_diams = [d.diameter_m for d in inside_dets]
    outside_diams = [d.diameter_m for d in outside_dets]

    return {
        "inside_dark_proxy": {
            "count": len(inside_dets),
            "percentage": float(round(len(inside_dets) / total * 100.0, 2)),
            "confidence": compute_distribution_stats(inside_confs),
            "diameter_m": compute_distribution_stats(inside_diams),
        },
        "outside_dark_proxy": {
            "count": len(outside_dets),
            "percentage": float(round(len(outside_dets) / total * 100.0, 2)),
            "confidence": compute_distribution_stats(outside_confs),
            "diameter_m": compute_distribution_stats(outside_diams),
        },
        "scientific_note": (
            "The dark-region mask is a statistical intensity proxy (Chunk-5 5th percentile cutoff), "
            "not a ground-truth physical shadow segmentation."
        ),
    }


def compute_scale_comparison(
    ohrc_detections: List[CraterDetection],
    tmc2_detections: List[CraterDetection],
    ohrc_gsd: float = 0.25,
    tmc2_gsd: float = 5.40,
) -> Dict[str, Any]:
    """Compute physical vs pixel scale relationship between OHRC and TMC-2 detections."""
    ratio = tmc2_gsd / ohrc_gsd

    ohrc_diams_m = [d.diameter_m for d in ohrc_detections]
    tmc2_diams_m = [d.diameter_m for d in tmc2_detections]

    min_shared = max(min(ohrc_diams_m) if ohrc_diams_m else 0, min(tmc2_diams_m) if tmc2_diams_m else 0) if (ohrc_diams_m and tmc2_diams_m) else 0
    max_shared = min(max(ohrc_diams_m) if ohrc_diams_m else 0, max(tmc2_diams_m) if tmc2_diams_m else 0) if (ohrc_diams_m and tmc2_diams_m) else 0

    has_shared_window = bool(min_shared <= max_shared and min_shared > 0 and max_shared > 0)
    shared_window = (
        [float(round(min_shared, 2)), float(round(max_shared, 2))]
        if has_shared_window
        else None
    )
    physical_gap_m = (
        float(round(min_shared - max_shared, 2))
        if not has_shared_window
        else 0.0
    )

    # Count detections in shared window if one exists
    ohrc_in_tmc_range = [d for d in ohrc_detections if d.diameter_m >= min_shared] if has_shared_window else []
    tmc_in_ohrc_range = [d for d in tmc2_detections if d.diameter_m <= max_shared] if has_shared_window else []

    return {
        "resolution_ratio": float(round(ratio, 2)),
        "ohrc_gsd_m": ohrc_gsd,
        "tmc2_gsd_m": tmc2_gsd,
        "physical_size_range_m": {
            "ohrc_observed": [
                float(round(min(ohrc_diams_m), 2)) if ohrc_diams_m else 0,
                float(round(max(ohrc_diams_m), 2)) if ohrc_diams_m else 0,
            ],
            "tmc2_observed": [
                float(round(min(tmc2_diams_m), 2)) if tmc2_diams_m else 0,
                float(round(max(tmc2_diams_m), 2)) if tmc2_diams_m else 0,
            ],
            "has_shared_overlap_window": has_shared_window,
            "shared_physical_overlap_window_m": shared_window,
            "physical_scale_gap_m": physical_gap_m,
        },
        "detectable_overlap_analysis": {
            "ohrc_detections_in_shared_range": len(ohrc_in_tmc_range),
            "ohrc_fraction_in_shared_range": float(
                round(len(ohrc_in_tmc_range) / max(len(ohrc_detections), 1), 4)
            ),
            "tmc2_detections_in_shared_range": len(tmc_in_ohrc_range),
            "tmc2_fraction_in_shared_range": float(
                round(len(tmc_in_ohrc_range) / max(len(tmc2_detections), 1), 4)
            ),
        },
        "critical_scale_observation": (
            f"Because TMC-2 GSD (5.40 m) is {ratio:.1f}× coarser than OHRC (0.25 m), a 100-pixel crater "
            f"in OHRC (25.0 m diameter) occupies only {25.0 / tmc2_gsd:.2f} pixels in TMC-2, below the reliable "
            f"detection threshold. Reliable cross-sensor matching can only occur on larger craters (>50 m diameter), "
            f"which represent the upper tail of the OHRC size distribution and the lower-to-middle tail of the TMC-2 distribution."
        ),
    }
