"""Image enhancement, gradient representations, and dark-region diagnostics.

Provides baseline OpenCV CLAHE contrast enhancement, spatial Sobel gradients,
and quantitative dark-region proxy segmentation for lunar orbital imagery.
"""

from typing import Any, Dict, Tuple
import cv2
import numpy as np


def apply_clahe(
    arr_uint8: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) using OpenCV.

    Conservative baseline representation without sensor-specific overfitting.

    Args:
        arr_uint8: Input 2D uint8 image array.
        clip_limit: Contrast clipping limit (default: 2.0).
        tile_grid_size: Tile grid size for localized equalization (default: (8, 8)).

    Returns:
        Tuple of:
            - Enhanced uint8 array.
            - Dict recording CLAHE parameters and characterization.
    """
    if arr_uint8.dtype != np.uint8:
        raise ValueError(f"CLAHE input array must be uint8, got {arr_uint8.dtype}")

    if arr_uint8.ndim != 2:
        raise ValueError(f"CLAHE input array must be 2D, got shape {arr_uint8.shape}")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(arr_uint8)

    params = {
        "method": "OpenCV_CLAHE",
        "clip_limit": float(clip_limit),
        "tile_grid_size": list(tile_grid_size),
        "is_baseline": True,
        "note": "Conservative baseline parameters; not claimed to be globally optimal for crater detection.",
    }

    return enhanced, params


def compute_gradients(
    arr_uint8: np.ndarray,
    ksize: int = 3,
) -> Dict[str, Any]:
    """Compute directional Sobel gradients and gradient magnitude.

    Calculates Sobel X (horizontal), Sobel Y (vertical), and gradient magnitude:
        G_mag = sqrt(Gx^2 + Gy^2)

    Args:
        arr_uint8: Input 2D uint8 image array.
        ksize: Aperture size for Sobel filter (must be 1, 3, 5, or 7; default: 3).

    Returns:
        Dict containing raw float64 gradient arrays, normalized uint8 visual representations,
        and statistical summary metrics.
    """
    if arr_uint8.dtype != np.uint8:
        raise ValueError(f"Gradient input array must be uint8, got {arr_uint8.dtype}")

    if arr_uint8.ndim != 2:
        raise ValueError(f"Gradient input array must be 2D, got shape {arr_uint8.shape}")

    gx = cv2.Sobel(arr_uint8, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(arr_uint8, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(gx**2 + gy**2)

    # Visual representations (uint8)
    # Centered at 128 for directional gradients (gray = 0 gradient)
    sobel_x_vis = np.clip(128.0 + 0.5 * gx, 0.0, 255.0).astype(np.uint8)
    sobel_y_vis = np.clip(128.0 + 0.5 * gy, 0.0, 255.0).astype(np.uint8)

    # 99th percentile scaling for magnitude to suppress outlier speckle
    mag_p99 = float(np.percentile(magnitude, 99.0))
    if mag_p99 > 1e-4:
        magnitude_vis = np.clip((magnitude / mag_p99) * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        magnitude_vis = np.zeros_like(arr_uint8, dtype=np.uint8)

    # Gradient statistical metrics
    mean_mag = float(np.mean(magnitude))
    median_mag = float(np.median(magnitude))
    p95_mag = float(np.percentile(magnitude, 95.0))
    p99_mag = float(np.percentile(magnitude, 99.0))
    max_mag = float(np.max(magnitude))
    frac_near_zero = float(np.mean(magnitude < 1.0))

    stats = {
        "mean_gradient_magnitude": mean_mag,
        "median_gradient_magnitude": median_mag,
        "p95_gradient_magnitude": p95_mag,
        "p99_gradient_magnitude": p99_mag,
        "max_gradient_magnitude": max_mag,
        "fraction_near_zero_gradient": frac_near_zero,
        "sobel_kernel_size": ksize,
    }

    return {
        "sobel_x": gx,
        "sobel_y": gy,
        "magnitude": magnitude,
        "sobel_x_vis": sobel_x_vis,
        "sobel_y_vis": sobel_y_vis,
        "magnitude_vis": magnitude_vis,
        "statistics": stats,
    }


def compute_dark_region_diagnostics(
    arr: np.ndarray,
    percentile_threshold: float = 5.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute shadow/dark-region diagnostic mask and spatial morphology statistics.

    IMPORTANT: This is a photometric DARK-REGION proxy (identifying low-albedo / deep-shadow
    candidates below a statistical percentile), not a certified physical shadow segmentation.

    Args:
        arr: Input numpy array (raw or normalized).
        percentile_threshold: Percentile cutoff for dark regions (default: 5.0).

    Returns:
        Tuple of:
            - binary uint8 mask (0 = illuminated, 255 = dark/shadow proxy).
            - dict recording threshold value, dark fraction, and connected-component statistics.
    """
    if arr.size == 0:
        raise ValueError("Cannot compute dark region diagnostics on an empty array.")

    data = arr.astype(np.float32)
    threshold_val = float(np.percentile(data, percentile_threshold))

    binary_mask = (data <= threshold_val).astype(np.uint8) * 255

    total_pixels = binary_mask.size
    dark_pixels = int(np.count_nonzero(binary_mask == 255))
    dark_fraction = float(dark_pixels / total_pixels)

    # Connected-component analysis (8-connectivity)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    # stats columns: [LEFT, TOP, WIDTH, HEIGHT, AREA]
    # Label 0 is background (illuminated pixels)
    if num_labels > 1:
        dark_areas = stats[1:, cv2.CC_STAT_AREA]
        num_components = len(dark_areas)
        mean_area = float(np.mean(dark_areas))
        median_area = float(np.median(dark_areas))
        max_area = int(np.max(dark_areas))
        p95_area = float(np.percentile(dark_areas, 95.0))
    else:
        num_components = 0
        mean_area = 0.0
        median_area = 0.0
        max_area = 0
        p95_area = 0.0

    diag_stats = {
        "proxy_definition": "Statistical low-intensity thresholding (DARK-REGION proxy, not ground-truth shadow segmentation)",
        "percentile_threshold": float(percentile_threshold),
        "intensity_threshold_value": threshold_val,
        "dark_pixel_count": dark_pixels,
        "total_pixel_count": total_pixels,
        "dark_fraction": dark_fraction,
        "connected_component_count": num_components,
        "mean_component_area_px": mean_area,
        "median_component_area_px": median_area,
        "max_component_area_px": max_area,
        "p95_component_area_px": p95_area,
    }

    return binary_mask, diag_stats
