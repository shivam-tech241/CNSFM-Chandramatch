"""Intensity normalization module for cross-sensor lunar orbital imagery.

Provides robust, percentile-based radiometric normalization that expands
usable contrast while remaining strictly non-destructive to raw source arrays.
"""

from typing import Any, Dict, Tuple
import numpy as np


def percentile_normalize(
    arr: np.ndarray,
    p_min: float = 1.0,
    p_max: float = 99.0,
    target_range: Tuple[float, float] = (0.0, 255.0),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Perform robust percentile-clipped contrast normalization on an image array.

    Does not modify or mutate the input array.

    Formula:
        vmin = percentile(arr, p_min)
        vmax = percentile(arr, p_max)
        clipped = clip(arr, vmin, vmax)
        normalized = (clipped - vmin) / (vmax - vmin) * (out_max - out_min) + out_min

    Args:
        arr: Input numpy array (uint8, uint16, float).
        p_min: Lower percentile for contrast floor (default: 1.0).
        p_max: Upper percentile for contrast ceiling (default: 99.0).
        target_range: Output intensity range (default: (0.0, 255.0)).

    Returns:
        Tuple of:
            - normalized uint8 array scaled to target_range.
            - dict recording exact parameters (p_min, p_max, vmin, vmax, input_dtype).
    """
    if arr.size == 0:
        raise ValueError("Cannot normalize an empty array.")

    if p_min >= p_max:
        raise ValueError(f"p_min ({p_min}) must be strictly less than p_max ({p_max}).")

    data = arr.astype(np.float32, copy=True)
    vmin = float(np.percentile(data, p_min))
    vmax = float(np.percentile(data, p_max))

    # Fallback if percentiles collapse (e.g. constant or near-constant patch)
    if vmax <= vmin:
        vmin = float(np.min(data))
        vmax = float(np.max(data))

    out_min, out_max = target_range

    if vmax > vmin:
        clipped = np.clip(data, vmin, vmax)
        normalized = (clipped - vmin) / (vmax - vmin) * (out_max - out_min) + out_min
    else:
        # Uniform image: output center of target range
        normalized = np.full_like(data, (out_min + out_max) / 2.0)

    # Cast to uint8 if default 0-255 range
    if out_min == 0.0 and out_max == 255.0:
        out_arr = np.clip(np.round(normalized), 0, 255).astype(np.uint8)
    else:
        out_arr = normalized.astype(np.float32)

    params = {
        "method": "percentile_clipping_minmax",
        "p_min_percentile": float(p_min),
        "p_max_percentile": float(p_max),
        "vmin_intensity": float(vmin),
        "vmax_intensity": float(vmax),
        "input_dtype": str(arr.dtype),
        "output_dtype": str(out_arr.dtype),
        "target_range": list(target_range),
    }

    return out_arr, params
