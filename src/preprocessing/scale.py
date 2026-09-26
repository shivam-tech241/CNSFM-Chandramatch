"""Resolution scale diagnostics for cross-sensor lunar imagery.

Quantifies the 21.6x spatial GSD disparity between OHRC (0.25 m/pixel)
and TMC-2 (5.40 m/pixel) and evaluates downsampled multi-scale representation.
"""

from typing import Any, Dict, Tuple
import cv2
import numpy as np


def compute_scale_diagnostic(
    ohrc_patch_native: np.ndarray,
    tmc2_patch: np.ndarray,
    ohrc_gsd_m: float = 0.25,
    tmc2_gsd_m: float = 5.40,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute scale diagnostic by downsampling native OHRC to match TMC-2 spatial resolution.

    Args:
        ohrc_patch_native: High-resolution native OHRC patch (e.g. 4320x4320 pixels).
        tmc2_patch: Co-located TMC-2 patch (e.g. 200x200 pixels).
        ohrc_gsd_m: Native ground sample distance of OHRC in meters (0.25 m/px).
        tmc2_gsd_m: Native ground sample distance of TMC-2 in meters (5.40 m/px).

    Returns:
        Tuple of:
            - Downsampled OHRC patch matching TMC-2 pixel dimensions.
            - Dict recording scale ratio, dimensions, and statistical comparison.
    """
    scale_ratio = float(tmc2_gsd_m / ohrc_gsd_m)  # 5.40 / 0.25 = 21.6

    target_h, target_w = tmc2_patch.shape[:2]

    # Downsample native OHRC to target dimensions using area averaging
    downsampled_ohrc = cv2.resize(
        ohrc_patch_native,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )

    metrics = {
        "ohrc_native_gsd_m": ohrc_gsd_m,
        "tmc2_native_gsd_m": tmc2_gsd_m,
        "resolution_scale_ratio": scale_ratio,
        "ohrc_native_shape": list(ohrc_patch_native.shape),
        "ohrc_downsampled_shape": list(downsampled_ohrc.shape),
        "tmc2_shape": list(tmc2_patch.shape),
        "ohrc_native_mean": float(np.mean(ohrc_patch_native)),
        "ohrc_native_std": float(np.std(ohrc_patch_native)),
        "ohrc_downsampled_mean": float(np.mean(downsampled_ohrc)),
        "ohrc_downsampled_std": float(np.std(downsampled_ohrc)),
        "tmc2_mean": float(np.mean(tmc2_patch)),
        "tmc2_std": float(np.std(tmc2_patch)),
        "downsampling_method": "cv2.INTER_AREA (pixel-area relation decimation)",
        "diagnostic_note": (
            "Diagnostic comparison only: demonstrates impact of 21.6x resolution difference. "
            "Downsampling OHRC preserves major crater rim morphology while eliminating sub-meter ejecta texture."
        ),
    }

    return downsampled_ohrc, metrics
