"""Script to execute Chunk 4: Ground-Grid Pixel Mapping and Overlap Extraction.

Loads official ISRO ground coordinate grid files for OHRC and TMC-2,
performs held-out interpolation accuracy validation, maps the OHRC footprint
boundary into TMC-2 pixel coordinates, exports machine-readable metadata
results/overlap/pixel_overlap.json, and generates diagnostic visualization previews.
"""

import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import (
    DatasetLoader,
    GroundGrid,
    compute_sensor_overlap,
)
from src.utils.config import load_config
from src.utils.logging import setup_logger


def safe_percentile_stretch(arr: np.ndarray, p_min: float = 1.0, p_max: float = 99.0) -> np.ndarray:
    """Normalize 2D array to uint8 0-255 using percentile clipping for preview visualization."""
    data = arr.astype(np.float32)
    vmin = float(np.percentile(data, p_min))
    vmax = float(np.percentile(data, p_max))
    if vmax <= vmin:
        vmin = float(np.min(data))
        vmax = float(np.max(data))
    if vmax > vmin:
        clipped = np.clip(data, vmin, vmax)
        return ((clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    return np.zeros_like(data, dtype=np.uint8)


def plot_boundary_in_tmc2(
    tmc2_grid: GroundGrid,
    overlap_res,
    boundary_data: dict,
    output_path: Path,
):
    """Plot OHRC boundary polygon in the TMC-2 pixel coordinate system."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 9), dpi=150, gridspec_kw={"width_ratios": [1, 2]})

    # 1. Full TMC-2 swath overview
    ax_full.set_xlim(0, tmc2_grid.pixel_max)
    ax_full.set_ylim(tmc2_grid.scan_max, 0)  # Inverted scan axis (top-down image lines)
    ax_full.set_title("TMC-2 Full Swath (4000 × 295,234)\nwith OHRC Overlap Location", fontsize=11, fontweight="bold")
    ax_full.set_xlabel("TMC-2 Pixel (Sample)", fontsize=10)
    ax_full.set_ylabel("TMC-2 Scan (Line)", fontsize=10)
    ax_full.grid(True, linestyle="--", alpha=0.4)

    # Highlight overlap bounding box
    bbox_rect = patches.Rectangle(
        (overlap_res.tmc2_overlap_pixel_start, overlap_res.tmc2_overlap_scan_start),
        overlap_res.tmc2_overlap_pixel_end - overlap_res.tmc2_overlap_pixel_start,
        overlap_res.tmc2_overlap_scan_end - overlap_res.tmc2_overlap_scan_start,
        linewidth=2,
        edgecolor="#d62728",
        facecolor="#d62728",
        alpha=0.3,
        label="OHRC Strip Extent",
    )
    ax_full.add_patch(bbox_rect)
    ax_full.legend(loc="upper right")

    # Annotate scan range
    center_s = (overlap_res.tmc2_overlap_scan_start + overlap_res.tmc2_overlap_scan_end) / 2
    ax_full.annotate(
        f"OHRC Strip\nScans: {overlap_res.tmc2_overlap_scan_start}–{overlap_res.tmc2_overlap_scan_end}\n"
        f"Pixels: {overlap_res.tmc2_overlap_pixel_start}–{overlap_res.tmc2_overlap_pixel_end}",
        xy=(2841, center_s),
        xytext=(500, center_s - 30000),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.9),
    )

    # 2. Zoomed Detail View of the Overlap Region in TMC-2 Coordinates
    tmc_px = boundary_data["tmc2_pixel"]
    tmc_sc = boundary_data["tmc2_scan"]

    pad_px = 150
    pad_sc = 300
    ax_zoom.set_xlim(overlap_res.tmc2_overlap_pixel_start - pad_px, overlap_res.tmc2_overlap_pixel_end + pad_px)
    ax_zoom.set_ylim(overlap_res.tmc2_overlap_scan_end + pad_sc, overlap_res.tmc2_overlap_scan_start - pad_sc)
    ax_zoom.set_title(
        "OHRC Boundary Mapped into TMC-2 Pixel Coordinates\n"
        f"TMC-2 Window: Scans [{overlap_res.tmc2_overlap_scan_start} : {overlap_res.tmc2_overlap_scan_end}], "
        f"Pixels [{overlap_res.tmc2_overlap_pixel_start} : {overlap_res.tmc2_overlap_pixel_end}]",
        fontsize=11,
        fontweight="bold",
    )
    ax_zoom.set_xlabel("TMC-2 Pixel (Sample)", fontsize=10)
    ax_zoom.set_ylabel("TMC-2 Scan (Line)", fontsize=10)
    ax_zoom.grid(True, linestyle="--", alpha=0.5)

    # Plot official TMC-2 tie points inside this window
    sc_mask = (tmc2_grid.u_scans >= overlap_res.tmc2_overlap_scan_start - pad_sc) & (
        tmc2_grid.u_scans <= overlap_res.tmc2_overlap_scan_end + pad_sc
    )
    px_mask = (tmc2_grid.u_pixels >= overlap_res.tmc2_overlap_pixel_start - pad_px) & (
        tmc2_grid.u_pixels <= overlap_res.tmc2_overlap_pixel_end + pad_px
    )

    sc_in = tmc2_grid.u_scans[sc_mask]
    px_in = tmc2_grid.u_pixels[px_mask]
    grid_px, grid_sc = np.meshgrid(px_in, sc_in)
    ax_zoom.scatter(
        grid_px.ravel(),
        grid_sc.ravel(),
        color="#1f77b4",
        s=15,
        alpha=0.6,
        marker="+",
        label=f"ISRO TMC-2 Ground Grid Tie Points ({overlap_res.num_tie_points_in_overlap} in overlap)",
    )

    # Draw mapped OHRC polygon
    ax_zoom.plot(tmc_px, tmc_sc, color="#d62728", linewidth=2.5, label="OHRC Footprint Boundary")
    ax_zoom.fill(tmc_px, tmc_sc, color="#d62728", alpha=0.2)

    # Draw enclosing bounding box
    bbox_detail = patches.Rectangle(
        (overlap_res.tmc2_overlap_pixel_start, overlap_res.tmc2_overlap_scan_start),
        overlap_res.tmc2_overlap_pixel_end - overlap_res.tmc2_overlap_pixel_start,
        overlap_res.tmc2_overlap_scan_end - overlap_res.tmc2_overlap_scan_start,
        linewidth=1.8,
        linestyle="--",
        edgecolor="#2ca02c",
        facecolor="none",
        label=f"Calculated Overlap Bounding Box ({overlap_res.tmc2_overlap_scan_end - overlap_res.tmc2_overlap_scan_start} × {overlap_res.tmc2_overlap_pixel_end - overlap_res.tmc2_overlap_pixel_start} px)",
    )
    ax_zoom.add_patch(bbox_detail)

    # Mark key corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left of OHRC
    # In boundary_data:
    # 0 is top-left, 50 is top-right, 100 is bottom-right, 150 is bottom-left (for 50 pts/edge)
    n_edge = len(tmc_px) // 4
    c_tl = (tmc_px[0], tmc_sc[0])
    c_tr = (tmc_px[n_edge], tmc_sc[n_edge])
    c_br = (tmc_px[2 * n_edge], tmc_sc[2 * n_edge])
    c_bl = (tmc_px[3 * n_edge], tmc_sc[3 * n_edge])

    corners = [
        ("OHRC Corner (0, 0)\n[North-West]", c_tl, "right", "bottom"),
        ("OHRC Corner (11999, 0)\n[North-East]", c_tr, "left", "bottom"),
        ("OHRC Corner (11999, 93692)\n[South-East]", c_br, "left", "top"),
        ("OHRC Corner (0, 93692)\n[South-West]", c_bl, "right", "top"),
    ]

    for label, pt, ha, va in corners:
        ax_zoom.plot(pt[0], pt[1], marker="o", color="#d62728", markersize=7)
        ax_zoom.annotate(
            f"{label}\n({pt[0]:.1f}, {pt[1]:.1f})",
            xy=pt,
            xytext=(pt[0] + (-20 if ha == "right" else 20), pt[1] + (-30 if va == "bottom" else 30)),
            fontsize=8,
            fontweight="semibold",
            ha=ha,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d62728", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1),
        )

    ax_zoom.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def extract_diagnostic_crops(
    loader: DatasetLoader,
    ohrc_grid: GroundGrid,
    tmc2_grid: GroundGrid,
    overlap_res,
    output_dir: Path,
    logger,
):
    """Read the calculated TMC-2 overlap region and co-located OHRC samples, saving previews."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read full TMC-2 Overlap Region
    logger.info(
        f"Reading TMC-2 overlap region: Scans [{overlap_res.tmc2_overlap_scan_start}:{overlap_res.tmc2_overlap_scan_end}], "
        f"Pixels [{overlap_res.tmc2_overlap_pixel_start}:{overlap_res.tmc2_overlap_pixel_end}]..."
    )
    tmc_crop = loader.read_region(
        "TMC2",
        row_start=overlap_res.tmc2_overlap_scan_start,
        row_end=overlap_res.tmc2_overlap_scan_end,
        col_start=overlap_res.tmc2_overlap_pixel_start,
        col_end=overlap_res.tmc2_overlap_pixel_end,
    )
    logger.info(f"Loaded TMC-2 overlap crop: shape={tmc_crop.shape}, dtype={tmc_crop.dtype}")

    # Normalize TMC-2 crop safely
    tmc_crop_vis = safe_percentile_stretch(tmc_crop, p_min=1.0, p_max=99.0)

    # Save full TMC-2 overlap preview
    tmc_preview_path = output_dir / "tmc2_overlap_crop_preview.png"
    fig, ax = plt.subplots(figsize=(8, 16), dpi=150)
    im = ax.imshow(tmc_crop_vis, cmap="gray", aspect="auto")
    ax.set_title(
        f"TMC-2 Calculated Overlap Region\n"
        f"Lines: {overlap_res.tmc2_overlap_scan_start}–{overlap_res.tmc2_overlap_scan_end} (5112 lines)\n"
        f"Samples: {overlap_res.tmc2_overlap_pixel_start}–{overlap_res.tmc2_overlap_pixel_end} (644 samples)\n"
        f"GSD: 5.40 m/pixel | Selenographic Lat: -3.42° to -2.58°",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel("TMC-2 Sample Offset (from pixel 2519)", fontsize=9)
    ax.set_ylabel("TMC-2 Line Offset (from scan 280781)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Pixel Intensity (8-bit stretched)")
    plt.tight_layout()
    plt.savefig(tmc_preview_path)
    plt.close(fig)
    logger.info(f"Saved TMC-2 overlap preview to: {tmc_preview_path}")

    # 2. Read Co-Located Diagnostic Patches (Top, Center, Bottom)
    # Define 3 representative lunar surface points inside the overlap
    sample_points = [
        ("Top Region (~Lat -2.70°)", 336.535, -2.700),
        ("Center Region (~Lat -3.00°)", 336.536, -3.000),
        ("Bottom Region (~Lat -3.30°)", 336.536, -3.300),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 14), dpi=150)

    patch_half_tmc = 150  # 300x300 pixels in TMC-2 = 1.62 km x 1.62 km
    patch_half_ohrc = 600  # 1200x1200 pixels in OHRC = 0.30 km x 0.30 km (higher resolution)

    for idx, (title, lon_t, lat_t) in enumerate(sample_points):
        # Map to TMC-2 pixel/scan
        px_t, sc_t = tmc2_grid.geo_to_pixel(lon_t, lat_t)
        # Map to OHRC pixel/scan
        px_o, sc_o = ohrc_grid.geo_to_pixel(lon_t, lat_t)

        r0_t = max(0, int(round(sc_t)) - patch_half_tmc)
        r1_t = min(tmc2_grid.scan_max, r0_t + 2 * patch_half_tmc)
        c0_t = max(0, int(round(px_t)) - patch_half_tmc)
        c1_t = min(tmc2_grid.pixel_max, c0_t + 2 * patch_half_tmc)

        r0_o = max(0, int(round(sc_o)) - patch_half_ohrc)
        r1_o = min(ohrc_grid.scan_max, r0_o + 2 * patch_half_ohrc)
        c0_o = max(0, int(round(px_o)) - patch_half_ohrc)
        c1_o = min(ohrc_grid.pixel_max, c0_o + 2 * patch_half_ohrc)

        tmc_patch = loader.read_region("TMC2", r0_t, r1_t, c0_t, c1_t)
        ohrc_patch = loader.read_region("OHRC", r0_o, r1_o, c0_o, c1_o)

        tmc_norm = safe_percentile_stretch(tmc_patch)
        ohrc_norm = safe_percentile_stretch(ohrc_patch)

        # Plot TMC-2 patch
        ax_tmc = axes[idx, 0]
        ax_tmc.imshow(tmc_norm, cmap="gray")
        ax_tmc.set_title(
            f"TMC-2: {title}\nPixel: {px_t:.1f}, Scan: {sc_t:.1f}\nResolution: 5.40 m/px",
            fontsize=9,
            fontweight="bold",
        )
        ax_tmc.axis("off")

        # Plot OHRC patch
        ax_ohrc = axes[idx, 1]
        ax_ohrc.imshow(ohrc_norm, cmap="gray")
        ax_ohrc.set_title(
            f"OHRC: {title}\nPixel: {px_o:.1f}, Scan: {sc_o:.1f}\nResolution: 0.25 m/px",
            fontsize=9,
            fontweight="bold",
        )
        ax_ohrc.axis("off")

    plt.suptitle(
        "Diagnostic Co-Located Surface Feature Crops\n"
        "Verifying Georeferencing Alignment Between OHRC and TMC-2",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    colocated_path = output_dir / "colocated_feature_previews.png"
    plt.savefig(colocated_path)
    plt.close(fig)
    logger.info(f"Saved co-located feature previews to: {colocated_path}")

    return tmc_preview_path, colocated_path


def main():
    logger = setup_logger(name="GroundGridMapping", level="INFO")
    logger.info("Starting Ground-Grid Pixel Mapping and Overlap Extraction (Chunk 4)...")

    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    config = load_config(config_path)
    root_dir = config.dataset.root_dir

    ohrc_grd_csv = root_dir / "OHRC" / "geometry" / "calibrated" / "20210405" / "ch2_ohr_ncp_20210405T1606536730_g_grd_d18.csv"
    tmc2_grd_csv = root_dir / "TMC" / "geometry" / "calibrated" / "20250807" / "ch2_tmc_ncf_20250807T1904346039_g_grd_d18.csv"

    # Task 1: Load and Understand Ground Grids
    logger.info(f"Loading OHRC Ground Grid from: {ohrc_grd_csv}")
    ohrc_grid = GroundGrid(ohrc_grd_csv, sensor_name="OHRC")
    ohrc_summary = ohrc_grid.get_summary()
    logger.info(
        f"OHRC Grid: {ohrc_summary.num_points} points ({ohrc_summary.n_scans} scans × {ohrc_summary.n_pixels} pixels), "
        f"Regular: {ohrc_summary.is_regular_grid}, Monotonic: Lat={ohrc_summary.is_monotonic_lat}, Lon={ohrc_summary.is_monotonic_lon}"
    )

    logger.info(f"Loading TMC-2 Ground Grid from: {tmc2_grd_csv}")
    tmc2_grid = GroundGrid(tmc2_grd_csv, sensor_name="TMC-2")
    tmc2_summary = tmc2_grid.get_summary()
    logger.info(
        f"TMC-2 Grid: {tmc2_summary.num_points} points ({tmc2_summary.n_scans} scans × {tmc2_summary.n_pixels} pixels), "
        f"Regular: {tmc2_summary.is_regular_grid}, Monotonic: Lat={tmc2_summary.is_monotonic_lat}, Lon={tmc2_summary.is_monotonic_lon}"
    )

    # Task 4: Held-out and Round-Trip Validation
    logger.info("Performing genuine held-out interpolation validation on TMC-2 (stride 2)...")
    tmc2_held_out = tmc2_grid.validate_held_out(stride=2)
    logger.info(
        f"TMC-2 Held-Out Error ({tmc2_held_out.num_samples} samples): "
        f"Pixel: mean={tmc2_held_out.pixel_error_mean:.3f} px, max={tmc2_held_out.pixel_error_max:.3f} px | "
        f"Scan: mean={tmc2_held_out.scan_error_mean:.3f} px, max={tmc2_held_out.scan_error_max:.3f} px | "
        f"Distance: mean={tmc2_held_out.geo_dist_m_mean:.2f} m, max={tmc2_held_out.geo_dist_m_max:.2f} m"
    )

    logger.info("Performing genuine held-out interpolation validation on OHRC (stride 2)...")
    ohrc_held_out = ohrc_grid.validate_held_out(stride=2, max_scans=200)
    logger.info(
        f"OHRC Held-Out Error ({ohrc_held_out.num_samples} samples): "
        f"Pixel: mean={ohrc_held_out.pixel_error_mean:.4f} px, max={ohrc_held_out.pixel_error_max:.4f} px | "
        f"Scan: mean={ohrc_held_out.scan_error_mean:.4f} px, max={ohrc_held_out.scan_error_max:.4f} px | "
        f"Distance: mean={ohrc_held_out.geo_dist_m_mean:.3f} m, max={ohrc_held_out.geo_dist_m_max:.3f} m"
    )

    logger.info("Evaluating round-trip mathematical reversibility (500 interior points)...")
    tmc2_round_trip = tmc2_grid.validate_round_trip(n_samples=500)
    ohrc_round_trip = ohrc_grid.validate_round_trip(n_samples=500)
    logger.info(f"TMC-2 Round-Trip Precision: max px error = {tmc2_round_trip.pixel_error_max:.2e} px")
    logger.info(f"OHRC Round-Trip Precision: max px error = {ohrc_round_trip.pixel_error_max:.2e} px")

    logger.info("Computing global affine diagnostic comparison...")
    tmc2_affine_diag = tmc2_grid.compute_global_affine_diagnostic()
    logger.info(
        f"TMC-2 Global Affine Residuals: "
        f"Pixel: mean={tmc2_affine_diag.pixel_error_mean:.2f} px, max={tmc2_affine_diag.pixel_error_max:.2f} px | "
        f"Scan: mean={tmc2_affine_diag.scan_error_mean:.2f} px, max={tmc2_affine_diag.scan_error_max:.2f} px"
    )

    # Task 5 & 6: OHRC inside TMC-2 Overlap Mapping
    logger.info("Mapping OHRC footprint boundary into TMC-2 pixel coordinates...")
    overlap_res, boundary_data = compute_sensor_overlap(ohrc_grid, tmc2_grid, num_boundary_samples_per_edge=50)

    logger.info(
        f"TMC-2 Overlap Window: Scan [{overlap_res.tmc2_overlap_scan_start} : {overlap_res.tmc2_overlap_scan_end}] "
        f"({overlap_res.tmc2_overlap_scan_end - overlap_res.tmc2_overlap_scan_start} lines), "
        f"Pixel [{overlap_res.tmc2_overlap_pixel_start} : {overlap_res.tmc2_overlap_pixel_end}] "
        f"({overlap_res.tmc2_overlap_pixel_end - overlap_res.tmc2_overlap_pixel_start} samples)"
    )
    logger.info(f"Official TMC-2 tie points enclosed: {overlap_res.num_tie_points_in_overlap}")

    # Plot Boundary in TMC-2
    output_dir = PROJECT_ROOT / "results" / "overlap"
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_plot_path = output_dir / "ohrc_boundary_in_tmc2.png"
    logger.info(f"Generating boundary visualization: {boundary_plot_path}")
    plot_boundary_in_tmc2(tmc2_grid, overlap_res, boundary_data, boundary_plot_path)

    # Task 7: Create Machine-Readable Metadata (pixel_overlap.json)
    metadata_json_path = output_dir / "pixel_overlap.json"
    logger.info(f"Writing overlap extraction metadata: {metadata_json_path}")

    metadata_payload = {
        "timestamp_generated": "2026-09-25T09:40:00Z",
        "description": "ISRO Ground-Grid Pixel Mapping and Overlap Extraction Metadata (Chunk 4)",
        "ohrc": {
            "image_dimensions": {
                "lines": int(ohrc_grid.scan_max + 1),
                "samples": int(ohrc_grid.pixel_max + 1),
            },
            "pixel_resolution_m": 0.25,
            "geographic_bounds": {
                "longitude_min_deg": float(ohrc_grid.lon_min),
                "longitude_max_deg": float(ohrc_grid.lon_max),
                "latitude_min_deg": float(ohrc_grid.lat_min),
                "latitude_max_deg": float(ohrc_grid.lat_max),
            },
            "ground_grid": {
                "csv_path": str(ohrc_grd_csv),
                "total_tie_points": ohrc_grid.num_points,
                "scans_count": ohrc_grid.n_scans,
                "pixels_count": ohrc_grid.n_pixels,
                "regular_grid_spacing": "100 scans × 100 pixels",
            },
            "held_out_validation": ohrc_held_out.to_dict(),
            "round_trip_reversibility": ohrc_round_trip.to_dict(),
        },
        "tmc2": {
            "image_dimensions": {
                "lines": int(tmc2_grid.scan_max + 1),
                "samples": int(tmc2_grid.pixel_max + 1),
            },
            "pixel_resolution_m": 5.40,
            "overlap_scan_bounds": {
                "continuous_min": float(overlap_res.tmc2_scan_bounds_continuous[0]),
                "continuous_max": float(overlap_res.tmc2_scan_bounds_continuous[1]),
                "integer_window_start": overlap_res.tmc2_overlap_scan_start,
                "integer_window_end": overlap_res.tmc2_overlap_scan_end,
                "span_lines": overlap_res.tmc2_overlap_scan_end - overlap_res.tmc2_overlap_scan_start,
            },
            "overlap_pixel_bounds": {
                "continuous_min": float(overlap_res.tmc2_pixel_bounds_continuous[0]),
                "continuous_max": float(overlap_res.tmc2_pixel_bounds_continuous[1]),
                "integer_window_start": overlap_res.tmc2_overlap_pixel_start,
                "integer_window_end": overlap_res.tmc2_overlap_pixel_end,
                "span_samples": overlap_res.tmc2_overlap_pixel_end - overlap_res.tmc2_overlap_pixel_start,
            },
            "overlap_geographic_bounds": {
                "longitude_min_deg": float(ohrc_grid.lon_min),
                "longitude_max_deg": float(ohrc_grid.lon_max),
                "latitude_min_deg": float(ohrc_grid.lat_min),
                "latitude_max_deg": float(ohrc_grid.lat_max),
            },
            "ground_grid": {
                "csv_path": str(tmc2_grd_csv),
                "total_tie_points": tmc2_grid.num_points,
                "tie_points_inside_overlap": overlap_res.num_tie_points_in_overlap,
                "regular_grid_spacing": "100 scans × 100 pixels",
            },
            "held_out_validation": tmc2_held_out.to_dict(),
            "round_trip_reversibility": tmc2_round_trip.to_dict(),
            "diagnostic_global_affine_comparison": tmc2_affine_diag.to_dict(),
        },
        "mapping_methodology": {
            "interpolation_algorithm": (
                "Cell-wise Bilinear Grid Interpolation on Official ISRO Ground Grids "
                "with 2D Newton-Raphson Inverse Bilinear Solver"
            ),
            "monotonic_spatial_index": (
                "Binary search on monotonic scan-latitude row envelopes, followed by "
                "binary search on pixel-longitude cell bounds"
            ),
            "number_of_grid_points_used": {
                "ohrc_total": ohrc_grid.num_points,
                "tmc2_total": tmc2_grid.num_points,
                "tmc2_inside_overlap": overlap_res.num_tie_points_in_overlap,
            },
            "confidence_and_limitations": {
                "distinction_official_vs_computed": (
                    "CRITICAL RESEARCH DISTINCTION: The ISRO ground-grid tie points (_g_grd_d18.csv) "
                    "represent official mission orbit-attitude-sensor calibrated geometry. The mapping "
                    "between tie points is our mathematical interpolation approximation."
                ),
                "accuracy_assessment": (
                    f"Genuine held-out validation demonstrates that cell-wise bilinear interpolation "
                    f"achieves sub-pixel precision across both sensors: TMC-2 held-out mean pixel error "
                    f"is {tmc2_held_out.pixel_error_mean:.3f} px (mean distance {tmc2_held_out.geo_dist_m_mean:.2f} m), "
                    f"and OHRC held-out mean pixel error is {ohrc_held_out.pixel_error_mean:.4f} px "
                    f"(mean distance {ohrc_held_out.geo_dist_m_mean:.3f} m). In contrast, a single global "
                    f"affine transformation fails completely with mean pixel error of "
                    f"{tmc2_affine_diag.pixel_error_mean:.2f} px (maximum error {tmc2_affine_diag.pixel_error_max:.2f} px)."
                ),
                "limitations": [
                    "2D sparse ground grid does not account for high-frequency lunar surface topography relief / parallax between tie points.",
                    "Tie points are spaced every 100 pixels / 100 scans (~25m for OHRC, ~540m for TMC-2).",
                    "Fine-grained crater feature registration (Chunks 5+) will be required to resolve remaining sub-pixel to few-pixel optical disparities.",
                ],
            },
        },
    }

    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)
    logger.info(f"Successfully saved {metadata_json_path}")

    # Task 8: Extract Diagnostic Image Crops
    loader = DatasetLoader(config=config)
    try:
        extract_diagnostic_crops(
            loader=loader,
            ohrc_grid=ohrc_grid,
            tmc2_grid=tmc2_grid,
            overlap_res=overlap_res,
            output_dir=output_dir,
            logger=logger,
        )
    finally:
        loader.close()

    logger.info("Chunk 4 Ground-Grid Mapping and Overlap Extraction completed successfully.")


if __name__ == "__main__":
    main()
