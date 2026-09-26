"""Diagnostic script: Read tiny crops from OHRC and TMC-2, perform safe normalization, and save PNG previews.

Chunk 2 Data Loading Validation.
"""

import json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import DatasetLoader
from src.utils.config import load_config
from src.utils.logging import setup_logger


def safe_normalize_to_uint8(arr: np.ndarray, p_min: float = 1.0, p_max: float = 99.0) -> np.ndarray:
    """Normalize array values to 0-255 uint8 using safe percentile stretching.

    Args:
        arr: Input numpy array (uint8 or uint16).
        p_min: Lower percentile for contrast clipping.
        p_max: Upper percentile for contrast clipping.

    Returns:
        uint8 numpy array suitable for PNG preview visualization.
    """
    data = arr.astype(np.float32)
    vmin = np.percentile(data, p_min)
    vmax = np.percentile(data, p_max)

    if vmax <= vmin:
        vmin = float(np.min(data))
        vmax = float(np.max(data))

    if vmax > vmin:
        clipped = np.clip(data, vmin, vmax)
        normalized = ((clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    else:
        normalized = np.zeros_like(data, dtype=np.uint8)

    return normalized


def run_diagnostic():
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    config = load_config(config_path)

    logger = setup_logger(
        name="DiagnosticCrop",
        level="INFO",
    )

    output_dir = PROJECT_ROOT / "results" / "data_loading"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DatasetLoader(config=config)

    logger.info("Initializing diagnostic crop reads...")

    # 1. Read small crop from OHRC (500 x 500)
    # Sampling from middle line/sample region to ensure illuminated surface content
    ohrc_r0, ohrc_r1 = 10000, 10500
    ohrc_c0, ohrc_c1 = 5000, 5500
    logger.info(f"Reading OHRC region [{ohrc_r0}:{ohrc_r1}, {ohrc_c0}:{ohrc_c1}]...")
    ohrc_crop = loader.read_region("OHRC", ohrc_r0, ohrc_r1, ohrc_c0, ohrc_c1)

    # 2. Read small crop from TMC-2 (500 x 500)
    tmc2_r0, tmc2_r1 = 50000, 50500
    tmc2_c0, tmc2_c1 = 1500, 2000
    logger.info(f"Reading TMC-2 region [{tmc2_r0}:{tmc2_r1}, {tmc2_c0}:{tmc2_c1}]...")
    tmc2_crop = loader.read_region("TMC2", tmc2_r0, tmc2_r1, tmc2_c0, tmc2_c1)

    # Calculate statistics
    stats = {
        "ohrc": {
            "requested_region": [ohrc_r0, ohrc_r1, ohrc_c0, ohrc_c1],
            "shape": list(ohrc_crop.shape),
            "dtype": str(ohrc_crop.dtype),
            "min": float(np.min(ohrc_crop)),
            "max": float(np.max(ohrc_crop)),
            "mean": float(np.mean(ohrc_crop)),
            "std": float(np.std(ohrc_crop)),
            "is_finite": bool(np.all(np.isfinite(ohrc_crop))),
        },
        "tmc2": {
            "requested_region": [tmc2_r0, tmc2_r1, tmc2_c0, tmc2_c1],
            "shape": list(tmc2_crop.shape),
            "dtype": str(tmc2_crop.dtype),
            "min": float(np.min(tmc2_crop)),
            "max": float(np.max(tmc2_crop)),
            "mean": float(np.mean(tmc2_crop)),
            "std": float(np.std(tmc2_crop)),
            "is_finite": bool(np.all(np.isfinite(tmc2_crop))),
        },
    }

    # 3. Perform safe normalization for preview
    ohrc_norm = safe_normalize_to_uint8(ohrc_crop)
    tmc2_norm = safe_normalize_to_uint8(tmc2_crop)

    # 4. Save PNG previews
    ohrc_png_path = output_dir / "ohrc_crop_preview.png"
    tmc2_png_path = output_dir / "tmc2_crop_preview.png"
    plt.imsave(ohrc_png_path, ohrc_norm, cmap="gray")
    plt.imsave(tmc2_png_path, tmc2_norm, cmap="gray")

    # Save stats JSON
    stats_json_path = output_dir / "crop_stats.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved OHRC preview: {ohrc_png_path}")
    logger.info(f"Saved TMC-2 preview: {tmc2_png_path}")
    logger.info(f"Saved crop stats: {stats_json_path}")
    logger.info("Diagnostic completed successfully.")

    loader.close()
    return stats


if __name__ == "__main__":
    run_diagnostic()
