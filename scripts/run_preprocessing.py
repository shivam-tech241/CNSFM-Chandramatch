"""Script to execute Chunk 5: Preprocessing and Image Characterization.

Orchestrates raw image statistical characterization, percentile intensity normalization,
OpenCV CLAHE representations, directional Sobel gradients, dark-region morphological diagnostics,
resolution scale analysis, cross-sensor visual comparisons, and machine-readable metadata export.
"""

import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import DatasetLoader
from src.preprocessing import (
    ImageStatistics,
    OverlapDataExtractor,
    apply_clahe,
    compute_dark_region_diagnostics,
    compute_gradients,
    compute_image_statistics,
    compute_scale_diagnostic,
    compute_streaming_image_statistics,
    percentile_normalize,
)
from src.utils.logging import setup_logger


def save_image_preview(arr_uint8: np.ndarray, output_path: Path, title: str, cmap: str = "gray") -> None:
    """Save an in-memory 2D array as a PNG preview with title and labeled axes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    im = ax.imshow(arr_uint8, cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Sample (Pixel)", fontsize=9)
    ax.set_ylabel("Line (Scan)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def save_gradients_preview(
    grads: dict,
    output_path: Path,
    sensor_name: str,
) -> None:
    """Save a 3-panel figure showing Sobel X, Sobel Y, and Gradient Magnitude."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    # Sobel X
    im0 = axes[0].imshow(grads["sobel_x_vis"], cmap="gray", aspect="auto")
    axes[0].set_title(f"{sensor_name} — Sobel X (Horizontal)", fontsize=10, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Sobel Y
    im1 = axes[1].imshow(grads["sobel_y_vis"], cmap="gray", aspect="auto")
    axes[1].set_title(f"{sensor_name} — Sobel Y (Vertical)", fontsize=10, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Gradient Magnitude
    im2 = axes[2].imshow(grads["magnitude_vis"], cmap="inferno", aspect="auto")
    stats = grads["statistics"]
    axes[2].set_title(
        f"{sensor_name} — Gradient Magnitude\n"
        f"Mean: {stats['mean_gradient_magnitude']:.1f} | Med: {stats['median_gradient_magnitude']:.1f} | "
        f"P95: {stats['p95_gradient_magnitude']:.1f}",
        fontsize=10,
        fontweight="bold",
    )
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def save_scale_diagnostic_plot(
    ohrc_native: np.ndarray,
    ohrc_downsampled: np.ndarray,
    tmc2_patch: np.ndarray,
    metrics: dict,
    output_path: Path,
) -> None:
    """Save multi-panel resolution scale diagnostic plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    # 1. Native OHRC (zoomed central sub-region to reveal high-frequency ejecta & small craters)
    h, w = ohrc_native.shape[:2]
    # Crop central 800x800 for visual clarity of 0.25m detail
    c_y, c_x = h // 2, w // 2
    sub_ohrc = ohrc_native[c_y - 400 : c_y + 400, c_x - 400 : c_x + 400]
    sub_norm, _ = percentile_normalize(sub_ohrc)

    axes[0].imshow(sub_norm, cmap="gray")
    axes[0].set_title(
        f"Native OHRC (High-Res Detail)\n"
        f"800×800 px sub-window | GSD: 0.25 m/px\n"
        f"Resolves small boulder ejecta & meter-scale crater rims",
        fontsize=10,
        fontweight="bold",
    )
    axes[0].axis("off")

    # 2. Downsampled OHRC (full patch decimated by 21.6x)
    ohrc_down_norm, _ = percentile_normalize(ohrc_downsampled)
    axes[1].imshow(ohrc_down_norm, cmap="gray")
    axes[1].set_title(
        f"Downsampled OHRC (Decimated 21.6×)\n"
        f"{ohrc_downsampled.shape[0]}×{ohrc_downsampled.shape[1]} px | Effective GSD: ~5.40 m/px\n"
        f"Area-averaged: suppresses sub-meter texture, preserves major rims",
        fontsize=10,
        fontweight="bold",
    )
    axes[1].axis("off")

    # 3. Native TMC-2 Patch
    tmc_norm, _ = percentile_normalize(tmc2_patch)
    axes[2].imshow(tmc_norm, cmap="gray")
    axes[2].set_title(
        f"Native TMC-2 Co-Located Patch\n"
        f"{tmc2_patch.shape[0]}×{tmc2_patch.shape[1]} px | GSD: 5.40 m/px\n"
        f"Physical ground extent: ~1.08 km × 1.08 km",
        fontsize=10,
        fontweight="bold",
    )
    axes[2].axis("off")

    plt.suptitle(
        f"Resolution Scale Diagnostic: OHRC (0.25 m/px) vs TMC-2 (5.40 m/px) — Scale Ratio {metrics['resolution_scale_ratio']:.1f}×",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def save_cross_sensor_comparison(
    ohrc_raw: np.ndarray,
    tmc_raw: np.ndarray,
    ohrc_norm: np.ndarray,
    tmc_norm: np.ndarray,
    ohrc_clahe: np.ndarray,
    tmc_clahe: np.ndarray,
    ohrc_grad_mag: np.ndarray,
    tmc_grad_mag: np.ndarray,
    ohrc_dark_mask: np.ndarray,
    tmc_dark_mask: np.ndarray,
    output_path: Path,
) -> None:
    """Save comprehensive 5-row cross-sensor visual comparison figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 2, figsize=(12, 22), dpi=150)

    # Helper for raw display (direct linear min-max stretch to show raw digital contrast)
    def to_display_raw(a):
        vmin, vmax = float(np.min(a)), float(np.max(a))
        if vmax > vmin:
            return ((a.astype(np.float32) - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
        return np.zeros_like(a, dtype=np.uint8)

    rows = [
        ("A. Raw Unprocessed (Linear Min-Max)", to_display_raw(ohrc_raw), to_display_raw(tmc_raw), "gray"),
        ("B. Percentile Normalized (P1–P99)", ohrc_norm, tmc_norm, "gray"),
        ("C. Baseline CLAHE Enhancement", ohrc_clahe, tmc_clahe, "gray"),
        ("D. Gradient Magnitude (Sobel)", ohrc_grad_mag, tmc_grad_mag, "inferno"),
        ("E. Dark-Region Diagnostic Proxy (P5 Cutoff)", ohrc_dark_mask, tmc_dark_mask, "magma"),
    ]

    for row_idx, (row_label, ohrc_img, tmc_img, cmap) in enumerate(rows):
        # OHRC column
        ax_o = axes[row_idx, 0]
        ax_o.imshow(ohrc_img, cmap=cmap)
        ax_o.set_title(f"OHRC (0.25 m/px) — {row_label}", fontsize=9, fontweight="bold")
        ax_o.axis("off")

        # TMC-2 column
        ax_t = axes[row_idx, 1]
        ax_t.imshow(tmc_img, cmap=cmap)
        ax_t.set_title(f"TMC-2 (5.40 m/px) — {row_label}", fontsize=9, fontweight="bold")
        ax_t.axis("off")

    plt.suptitle(
        "Cross-Sensor Representation Comparison: OHRC vs TMC-2 (Triplet 1 Overlap)",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def main():
    logger = setup_logger(name="Preprocessing", level="INFO")
    logger.info("Starting Preprocessing and Image Characterization (Chunk 5)...")

    # Output directories
    out_base = PROJECT_ROOT / "results" / "preprocessing"
    dir_stats = out_base / "statistics"
    dir_norm = out_base / "normalized"
    dir_clahe = out_base / "clahe"
    dir_grads = out_base / "gradients"
    dir_dark = out_base / "dark_regions"
    dir_scale = out_base / "scale_diagnostic"
    dir_overview = out_base / "overview"

    for d in [dir_stats, dir_norm, dir_clahe, dir_grads, dir_dark, dir_scale, dir_overview]:
        d.mkdir(parents=True, exist_ok=True)

    loader = DatasetLoader()
    extractor = OverlapDataExtractor(loader=loader)

    # 1. OVERLAP DATA EXTRACTION
    logger.info("Extracting calculated TMC-2 overlap region (Chunk-4 bounds)...")
    tmc_overlap_raw, tmc_meta = extractor.extract_tmc2_overlap()
    logger.info(
        f"Loaded TMC-2 overlap: shape={tmc_overlap_raw.shape}, dtype={tmc_overlap_raw.dtype}, "
        f"ground size: {tmc_meta.ground_width_km:.2f} km × {tmc_meta.ground_height_km:.2f} km"
    )

    logger.info("Extracting co-located benchmark patch pair around overlap center (lat -3.000°, lon 336.536°)...")
    tmc_bench_raw, ohrc_bench_raw, bench_meta = extractor.extract_benchmark_pair(
        center_lon=336.536, center_lat=-3.000, tmc_size_px=200
    )
    logger.info(f"Loaded Benchmark Pair: TMC-2 shape={tmc_bench_raw.shape}, OHRC shape={ohrc_bench_raw.shape}")

    # For detailed visual comparison, also extract an 800x800 sub-window of native OHRC at the same ground point
    h_b, w_b = ohrc_bench_raw.shape
    c_y, c_x = h_b // 2, w_b // 2
    ohrc_sub_raw = ohrc_bench_raw[c_y - 400 : c_y + 400, c_x - 400 : c_x + 400]

    # 2. RAW IMAGE CHARACTERIZATION
    logger.info("Computing exact raw image statistics for TMC-2 overlap region...")
    tmc_overlap_stats = compute_image_statistics(tmc_overlap_raw, sensor_name="TMC-2_Overlap")
    logger.info(
        f"TMC-2 Overlap Stats: min={tmc_overlap_stats.min}, max={tmc_overlap_stats.max}, "
        f"mean={tmc_overlap_stats.mean:.2f}, median={tmc_overlap_stats.median:.1f}, std={tmc_overlap_stats.std:.2f}, "
        f"P1={tmc_overlap_stats.p1:.1f}, P99={tmc_overlap_stats.p99:.1f}"
    )

    logger.info("Computing exact raw image statistics for OHRC benchmark region...")
    ohrc_bench_stats = compute_image_statistics(ohrc_bench_raw, sensor_name="OHRC_Benchmark")
    logger.info(
        f"OHRC Benchmark Stats: min={ohrc_bench_stats.min}, max={ohrc_bench_stats.max}, "
        f"mean={ohrc_bench_stats.mean:.2f}, median={ohrc_bench_stats.median:.1f}, std={ohrc_bench_stats.std:.2f}, "
        f"P1={ohrc_bench_stats.p1:.1f}, P99={ohrc_bench_stats.p99:.1f}"
    )

    logger.info("Streaming full OHRC raster statistics across all 93,693 lines...")
    ohrc_reader = loader.get_reader("OHRC")
    ohrc_full_stats = compute_streaming_image_statistics(ohrc_reader, sensor_name="OHRC_FullSwath", chunk_lines=10000)
    logger.info(
        f"OHRC Full Swath Stats (1.12B pixels): min={ohrc_full_stats.min}, max={ohrc_full_stats.max}, "
        f"mean={ohrc_full_stats.mean:.2f}, median={ohrc_full_stats.median:.1f}, std={ohrc_full_stats.std:.2f}, "
        f"P1={ohrc_full_stats.p1:.1f}, P99={ohrc_full_stats.p99:.1f}"
    )

    # Save raw statistics JSON
    raw_stats_payload = {
        "tmc2_overlap_crop": tmc_overlap_stats.to_dict(),
        "ohrc_benchmark_crop": ohrc_bench_stats.to_dict(),
        "ohrc_full_swath": ohrc_full_stats.to_dict(),
        "tmc2_benchmark_crop": compute_image_statistics(tmc_bench_raw, sensor_name="TMC-2_Benchmark").to_dict(),
    }
    raw_stats_path = dir_stats / "raw_statistics.json"
    with open(raw_stats_path, "w", encoding="utf-8") as f:
        json.dump(raw_stats_payload, f, indent=2)
    logger.info(f"Saved raw statistics to: {raw_stats_path}")

    # 3. INTENSITY NORMALIZATION (Percentile-based non-destructive scaling)
    logger.info("Generating percentile-normalized representations (P1–P99)...")
    tmc_norm, tmc_norm_meta = percentile_normalize(tmc_overlap_raw, p_min=1.0, p_max=99.0)
    tmc_bench_norm, _ = percentile_normalize(tmc_bench_raw, p_min=1.0, p_max=99.0)
    ohrc_bench_norm, ohrc_norm_meta = percentile_normalize(ohrc_bench_raw, p_min=1.0, p_max=99.0)
    ohrc_sub_norm, _ = percentile_normalize(ohrc_sub_raw, p_min=1.0, p_max=99.0)

    tmc_norm_preview_path = dir_norm / "tmc2_overlap_normalized.png"
    ohrc_norm_preview_path = dir_norm / "ohrc_benchmark_normalized.png"
    save_image_preview(
        tmc_norm,
        tmc_norm_preview_path,
        f"TMC-2 Overlap Normalized (P1={tmc_norm_meta['vmin_intensity']:.1f}, P99={tmc_norm_meta['vmax_intensity']:.1f})",
    )
    save_image_preview(
        ohrc_sub_norm,
        ohrc_norm_preview_path,
        f"OHRC Benchmark Normalized (P1={ohrc_norm_meta['vmin_intensity']:.1f}, P99={ohrc_norm_meta['vmax_intensity']:.1f})",
    )
    logger.info(f"Saved normalized previews to {dir_norm}")

    # 4. CLAHE REPRESENTATION (Conservative OpenCV baseline)
    logger.info("Generating OpenCV CLAHE representations (clipLimit=2.0, tileGridSize=(8, 8))...")
    tmc_clahe, tmc_clahe_meta = apply_clahe(tmc_norm, clip_limit=2.0, tile_grid_size=(8, 8))
    tmc_bench_clahe, _ = apply_clahe(tmc_bench_norm, clip_limit=2.0, tile_grid_size=(8, 8))
    ohrc_bench_clahe, ohrc_clahe_meta = apply_clahe(ohrc_bench_norm, clip_limit=2.0, tile_grid_size=(8, 8))
    ohrc_sub_clahe, _ = apply_clahe(ohrc_sub_norm, clip_limit=2.0, tile_grid_size=(8, 8))

    tmc_clahe_preview_path = dir_clahe / "tmc2_overlap_clahe.png"
    ohrc_clahe_preview_path = dir_clahe / "ohrc_benchmark_clahe.png"
    save_image_preview(tmc_clahe, tmc_clahe_preview_path, "TMC-2 Overlap — CLAHE Enhanced (clipLimit=2.0)")
    save_image_preview(ohrc_sub_clahe, ohrc_clahe_preview_path, "OHRC Benchmark — CLAHE Enhanced (clipLimit=2.0)")
    logger.info(f"Saved CLAHE previews to {dir_clahe}")

    # 5. GRADIENT REPRESENTATIONS (Sobel X, Y, Magnitude)
    logger.info("Computing spatial Sobel gradients and gradient magnitude distributions...")
    tmc_grads = compute_gradients(tmc_norm, ksize=3)
    tmc_bench_grads = compute_gradients(tmc_bench_norm, ksize=3)
    ohrc_sub_grads = compute_gradients(ohrc_sub_norm, ksize=3)

    tmc_grad_preview_path = dir_grads / "tmc2_gradients.png"
    ohrc_grad_preview_path = dir_grads / "ohrc_gradients.png"
    save_gradients_preview(tmc_grads, tmc_grad_preview_path, "TMC-2")
    save_gradients_preview(ohrc_sub_grads, ohrc_grad_preview_path, "OHRC")
    logger.info(f"Saved gradient previews to {dir_grads}")

    # 6. SHADOW / DARK-REGION DIAGNOSTICS (Statistical proxy)
    logger.info("Computing shadow/dark-region morphological diagnostic masks (5th percentile cutoff)...")
    tmc_dark_mask, tmc_dark_stats = compute_dark_region_diagnostics(tmc_overlap_raw, percentile_threshold=5.0)
    tmc_bench_dark_mask, _ = compute_dark_region_diagnostics(tmc_bench_raw, percentile_threshold=5.0)
    ohrc_bench_dark_mask, ohrc_dark_stats = compute_dark_region_diagnostics(ohrc_bench_raw, percentile_threshold=5.0)
    ohrc_sub_dark_mask, _ = compute_dark_region_diagnostics(ohrc_sub_raw, percentile_threshold=5.0)

    tmc_dark_preview_path = dir_dark / "tmc2_dark_region_mask.png"
    ohrc_dark_preview_path = dir_dark / "ohrc_dark_region_mask.png"
    save_image_preview(
        tmc_dark_mask,
        tmc_dark_preview_path,
        f"TMC-2 Dark-Region Proxy Mask (Fraction: {tmc_dark_stats['dark_fraction']*100:.2f}%, Components: {tmc_dark_stats['connected_component_count']})",
        cmap="magma",
    )
    save_image_preview(
        ohrc_sub_dark_mask,
        ohrc_dark_preview_path,
        f"OHRC Dark-Region Proxy Mask (Fraction: {ohrc_dark_stats['dark_fraction']*100:.2f}%, Components: {ohrc_dark_stats['connected_component_count']})",
        cmap="magma",
    )
    logger.info(f"Saved dark-region diagnostic previews to {dir_dark}")

    # 7. CROSS-SENSOR VISUAL COMPARISON (Multi-panel overview)
    logger.info("Generating multi-panel cross-sensor comparison figure...")
    overview_path = dir_overview / "cross_sensor_comparison.png"
    save_cross_sensor_comparison(
        ohrc_raw=ohrc_sub_raw,
        tmc_raw=tmc_bench_raw,
        ohrc_norm=ohrc_sub_norm,
        tmc_norm=tmc_bench_norm,
        ohrc_clahe=ohrc_sub_clahe,
        tmc_clahe=tmc_bench_clahe,
        ohrc_grad_mag=ohrc_sub_grads["magnitude_vis"],
        tmc_grad_mag=tmc_bench_grads["magnitude_vis"],
        ohrc_dark_mask=ohrc_sub_dark_mask,
        tmc_dark_mask=tmc_bench_dark_mask,
        output_path=overview_path,
    )
    logger.info(f"Saved cross-sensor overview figure to: {overview_path}")

    # 8. SCALE / RESOLUTION DIAGNOSTIC (21.6x Downsampling analysis)
    logger.info("Performing resolution scale diagnostic (downsampling OHRC by 21.6x toward TMC-2)...")
    downsampled_ohrc, scale_metrics = compute_scale_diagnostic(
        ohrc_patch_native=ohrc_bench_raw,
        tmc2_patch=tmc_bench_raw,
        ohrc_gsd_m=0.25,
        tmc2_gsd_m=5.40,
    )
    scale_plot_path = dir_scale / "scale_diagnostic.png"
    save_scale_diagnostic_plot(
        ohrc_native=ohrc_bench_raw,
        ohrc_downsampled=downsampled_ohrc,
        tmc2_patch=tmc_bench_raw,
        metrics=scale_metrics,
        output_path=scale_plot_path,
    )
    logger.info(f"Saved scale diagnostic plot to: {scale_plot_path}")

    # 9. OUTPUT MACHINE-READABLE REPORT (preprocessing_report.json)
    logger.info("Compiling machine-readable preprocessing report...")
    report_payload = {
        "timestamp_generated": "2026-09-25T10:15:00Z",
        "chunk": "CHUNK 5 — PREPROCESSING AND IMAGE CHARACTERIZATION",
        "source_imagery": {
            "tmc2": tmc_meta.to_dict(),
            "ohrc_benchmark": {
                "sensor_name": "OHRC",
                "row_start": bench_meta["ohrc"]["row_start"],
                "row_end": bench_meta["ohrc"]["row_end"],
                "col_start": bench_meta["ohrc"]["col_start"],
                "col_end": bench_meta["ohrc"]["col_end"],
                "shape": bench_meta["ohrc"]["shape"],
                "dtype": bench_meta["ohrc"]["dtype"],
                "gsd_m": 0.25,
            },
            "ohrc_full_swath_dimensions": {
                "lines": ohrc_reader.lines,
                "samples": ohrc_reader.samples,
                "dtype": "uint8",
                "gsd_m": 0.25,
            },
            "correspondence_note": bench_meta["correspondence_note"],
        },
        "raw_statistics": raw_stats_payload,
        "normalization_parameters": {
            "tmc2": tmc_norm_meta,
            "ohrc": ohrc_norm_meta,
        },
        "clahe_parameters": {
            "tmc2": tmc_clahe_meta,
            "ohrc": ohrc_clahe_meta,
        },
        "gradient_statistics": {
            "tmc2_overlap": tmc_grads["statistics"],
            "tmc2_benchmark": tmc_bench_grads["statistics"],
            "ohrc_sub_window": ohrc_sub_grads["statistics"],
        },
        "dark_region_diagnostics": {
            "tmc2_overlap": tmc_dark_stats,
            "ohrc_benchmark": ohrc_dark_stats,
        },
        "resolution_scale_diagnostic": scale_metrics,
        "artifact_paths": {
            "raw_statistics_json": str(raw_stats_path),
            "tmc2_normalized_preview": str(tmc_norm_preview_path),
            "ohrc_normalized_preview": str(ohrc_norm_preview_path),
            "tmc2_clahe_preview": str(tmc_clahe_preview_path),
            "ohrc_clahe_preview": str(ohrc_clahe_preview_path),
            "tmc2_gradients_preview": str(tmc_grad_preview_path),
            "ohrc_gradients_preview": str(ohrc_grad_preview_path),
            "tmc2_dark_region_mask": str(tmc_dark_preview_path),
            "ohrc_dark_region_mask": str(ohrc_dark_preview_path),
            "scale_diagnostic_plot": str(scale_plot_path),
            "cross_sensor_overview": str(overview_path),
        },
        "limitations_and_assumptions": [
            "Raw source rasters remain unmodified and read-only.",
            "Percentile normalization (P1-P99) clips extreme 1% outlier pixels to expand active dynamic range.",
            "OpenCV CLAHE uses conservative baseline parameters (clipLimit=2.0, tileGrid=(8,8)); not claimed optimal.",
            "Dark-region proxy uses statistical 5th percentile cutoff and is not certified physical shadow segmentation.",
            "Scale diagnostic uses pixel area averaging (cv2.INTER_AREA) to illustrate the 21.6x GSD disparity; final multi-scale detector strategy is deferred to Chunk 6.",
        ],
    }

    report_path = out_base / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    logger.info(f"Successfully exported preprocessing report to: {report_path}")
    logger.info("Chunk 5 execution complete.")


if __name__ == "__main__":
    main()
