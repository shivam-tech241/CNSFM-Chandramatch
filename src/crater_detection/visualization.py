"""Visualization routines for crater detection overlays, scale distributions, and QC contact sheets."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.crater_detection.base import CraterDetection


def plot_crater_overlay(
    image: np.ndarray,
    detections: List[CraterDetection],
    output_path: Union[str, Path],
    sensor_name: str,
    max_craters_to_draw: int = 1500,
    stride: int = 1,
) -> None:
    """Render and save an image with overlaid crater circles and center markers."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    h, w = image.shape[:2]
    # For large images, display a downsampled overview if image is huge
    scale = 1.0
    if max(h, w) > 2400:
        scale = 2400.0 / max(h, w)
        import cv2
        disp_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        disp_img = image

    # Aspect ratio for figure
    aspect = w / h
    fig_w = 10.0
    fig_h = max(4.0, fig_w / max(aspect, 0.2))
    fig_h = min(fig_h, 14.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.imshow(disp_img, cmap="gray", aspect="auto")

    # Draw up to max_craters_to_draw highest confidence detections
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)[:max_craters_to_draw]

    for det in sorted_dets:
        cx_scaled = det.center_x * scale
        cy_scaled = det.center_y * scale
        r_scaled = det.radius_px * scale

        # Circle ring
        circ = patches.Circle(
            (cx_scaled, cy_scaled),
            r_scaled,
            fill=False,
            edgecolor="#00FFAA",
            linewidth=0.8,
            alpha=0.75,
        )
        ax.add_patch(circ)

        # Center dot
        ax.plot(cx_scaled, cy_scaled, marker="+", color="#FF3366", markersize=3, markeredgewidth=0.7, alpha=0.8)

    ax.set_title(
        f"{sensor_name} — Crater Detections Overlay ({len(detections)} total, showing top {len(sorted_dets)})",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel("Pixel Column", fontsize=9)
    ax.set_ylabel("Pixel Row", fontsize=9)
    plt.tight_layout()
    plt.savefig(p, bbox_inches="tight")
    plt.close(fig)


def plot_scale_comparison(
    ohrc_detections: List[CraterDetection],
    tmc2_detections: List[CraterDetection],
    output_path: Union[str, Path],
) -> None:
    """Plot 4-panel comparison of pixel diameters vs physical diameters in meters."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    ohrc_diams_px = [d.diameter_px for d in ohrc_detections]
    tmc2_diams_px = [d.diameter_px for d in tmc2_detections]
    ohrc_diams_m = [d.diameter_m for d in ohrc_detections]
    tmc2_diams_m = [d.diameter_m for d in tmc2_detections]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # 1. OHRC Pixel Diameters
    axes[0, 0].hist(ohrc_diams_px, bins=40, color="#1f77b4", edgecolor="black", alpha=0.75)
    axes[0, 0].set_title(
        f"OHRC Crater Diameter (Pixels)\nMean: {np.mean(ohrc_diams_px):.1f} px | Med: {np.median(ohrc_diams_px):.1f} px",
        fontsize=10,
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("Diameter (pixels)", fontsize=9)
    axes[0, 0].set_ylabel("Frequency", fontsize=9)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # 2. TMC-2 Pixel Diameters
    axes[0, 1].hist(tmc2_diams_px, bins=40, color="#ff7f0e", edgecolor="black", alpha=0.75)
    axes[0, 1].set_title(
        f"TMC-2 Crater Diameter (Pixels)\nMean: {np.mean(tmc2_diams_px):.1f} px | Med: {np.median(tmc2_diams_px):.1f} px",
        fontsize=10,
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("Diameter (pixels)", fontsize=9)
    axes[0, 1].set_ylabel("Frequency", fontsize=9)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # 3. OHRC Physical Diameters (Meters)
    axes[1, 0].hist(ohrc_diams_m, bins=40, color="#2ca02c", edgecolor="black", alpha=0.75)
    axes[1, 0].set_title(
        f"OHRC Physical Diameter (Meters, GSD=0.25 m)\nMean: {np.mean(ohrc_diams_m):.1f} m | Med: {np.median(ohrc_diams_m):.1f} m",
        fontsize=10,
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("Physical Diameter (meters)", fontsize=9)
    axes[1, 0].set_ylabel("Frequency", fontsize=9)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 4. Cross-Sensor Physical Diameter Overlay
    axes[1, 1].hist(ohrc_diams_m, bins=50, color="#2ca02c", alpha=0.5, density=True, label="OHRC (0.25 m/px)")
    axes[1, 1].hist(tmc2_diams_m, bins=50, color="#d62728", alpha=0.5, density=True, label="TMC-2 (5.40 m/px)")
    axes[1, 1].axvline(50.0, color="black", linestyle="--", linewidth=1.2, label="Cross-Sensor Scale Threshold (~50m)")
    axes[1, 1].set_title(
        "Cross-Sensor Physical Size Distribution (Normalized Density)\nLog-scale demonstrates scale separation",
        fontsize=10,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("Physical Diameter (meters)", fontsize=9)
    axes[1, 1].set_ylabel("Probability Density", fontsize=9)
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(fontsize=8, loc="upper right")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        "Resolution Scale Analysis: OHRC (0.25 m/px) vs TMC-2 (5.40 m/px)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(p, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_density(
    ohrc_detections: List[CraterDetection],
    tmc2_detections: List[CraterDetection],
    output_path: Union[str, Path],
) -> None:
    """Plot spatial scatter of crater detections color-coded by confidence."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=150)

    # OHRC Scatter
    o_x = [d.center_x for d in ohrc_detections]
    o_y = [d.center_y for d in ohrc_detections]
    o_c = [d.confidence for d in ohrc_detections]

    sc0 = axes[0].scatter(o_x, o_y, c=o_c, cmap="viridis", s=3, alpha=0.6)
    axes[0].set_title(f"OHRC Crater Spatial Density (N={len(ohrc_detections)})", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("X (pixels)", fontsize=9)
    axes[0].set_ylabel("Y (pixels)", fontsize=9)
    axes[0].invert_yaxis()
    plt.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04, label="Confidence")

    # TMC-2 Scatter
    t_x = [d.center_x for d in tmc2_detections]
    t_y = [d.center_y for d in tmc2_detections]
    t_c = [d.confidence for d in tmc2_detections]

    sc1 = axes[1].scatter(t_x, t_y, c=t_c, cmap="plasma", s=4, alpha=0.6)
    axes[1].set_title(f"TMC-2 Crater Spatial Density (N={len(tmc2_detections)})", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("X (pixels)", fontsize=9)
    axes[1].set_ylabel("Y (pixels)", fontsize=9)
    axes[1].invert_yaxis()
    plt.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04, label="Confidence")

    plt.suptitle("Crater Spatial Distribution and Detection Confidence Across Overlap", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(p, bbox_inches="tight")
    plt.close(fig)


def create_qc_contact_sheet(
    image: np.ndarray,
    detections: List[CraterDetection],
    dark_mask: np.ndarray,
    output_path: Union[str, Path],
    sensor_name: str = "OHRC",
) -> Dict[str, Any]:
    """Generate a 7-panel qualitative inspection contact sheet covering the required QC categories:

    1. Clear crater
    2. Shallow crater
    3. Crater in bright terrain
    4. Crater in dark region
    5. Possible false positive
    6. Overlapping / nearby craters
    7. Small crater near resolution limit
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    h_img, w_img = image.shape[:2]

    used_keys = set()
    selected = {}

    # 1. Clear crater: high confidence, prominent radius
    cand_clear = [d for d in detections if d.confidence > 0.75 and d.radius_px > 25]
    sel1 = cand_clear[0] if cand_clear else detections[0]
    selected["1. Clear Crater"] = sel1
    used_keys.add((round(sel1.center_x, 1), round(sel1.center_y, 1)))

    # 2. Shallow crater: lower confidence and moderate radius
    cand_shallow = [
        d for d in detections
        if 0.32 <= d.confidence <= 0.42 and d.radius_px > 25
        and (round(d.center_x, 1), round(d.center_y, 1)) not in used_keys
    ]
    sel2 = cand_shallow[0] if cand_shallow else detections[1]
    selected["2. Shallow Crater"] = sel2
    used_keys.add((round(sel2.center_x, 1), round(sel2.center_y, 1)))

    # 3. Crater in bright terrain: high local intensity in CLAHE image
    cand_bright = []
    for d in detections:
        k = (round(d.center_x, 1), round(d.center_y, 1))
        if k in used_keys:
            continue
        iy, ix = int(np.clip(d.center_y, 0, h_img - 1)), int(np.clip(d.center_x, 0, w_img - 1))
        local_patch = image[max(0, iy - 10):min(h_img, iy + 10), max(0, ix - 10):min(w_img, ix + 10)]
        if np.mean(local_patch) > 125.0:
            cand_bright.append(d)
    sel3 = cand_bright[0] if cand_bright else detections[2]
    selected["3. Crater in Bright Terrain"] = sel3
    used_keys.add((round(sel3.center_x, 1), round(sel3.center_y, 1)))

    # 4. Crater in dark region: center pixel inside dark_mask
    cand_dark = []
    for d in detections:
        k = (round(d.center_x, 1), round(d.center_y, 1))
        if k in used_keys:
            continue
        iy, ix = int(np.clip(d.center_y, 0, h_img - 1)), int(np.clip(d.center_x, 0, w_img - 1))
        if dark_mask[iy, ix] > 0:
            cand_dark.append(d)
    sel4 = cand_dark[0] if cand_dark else detections[3]
    selected["4. Crater in Dark Region"] = sel4
    used_keys.add((round(sel4.center_x, 1), round(sel4.center_y, 1)))

    # 5. Possible false positive: lowest confidence candidate at detection margin
    cand_fp = [
        d for d in sorted(detections, key=lambda x: x.confidence)
        if (round(d.center_x, 1), round(d.center_y, 1)) not in used_keys
    ]
    sel5 = cand_fp[0] if cand_fp else detections[4]
    selected["5. Possible False Positive"] = sel5
    used_keys.add((round(sel5.center_x, 1), round(sel5.center_y, 1)))

    # 6. Overlapping / nearby craters: crater situated close to a neighbor
    sel6 = None
    for d in detections[:500]:
        k = (round(d.center_x, 1), round(d.center_y, 1))
        if k in used_keys:
            continue
        for d_other in detections[:500]:
            if d_other is not d:
                dist = np.hypot(d.center_x - d_other.center_x, d.center_y - d_other.center_y)
                if 0.8 * (d.radius_px + d_other.radius_px) < dist < 1.8 * (d.radius_px + d_other.radius_px):
                    sel6 = d
                    break
        if sel6 is not None:
            break
    if sel6 is None:
        sel6 = detections[5]
    selected["6. Overlapping / Nearby"] = sel6
    used_keys.add((round(sel6.center_x, 1), round(sel6.center_y, 1)))

    # 7. Small crater near resolution limit: smallest radius
    cand_small = [
        d for d in sorted(detections, key=lambda x: x.radius_px)
        if (round(d.center_x, 1), round(d.center_y, 1)) not in used_keys
    ]
    sel7 = cand_small[0] if cand_small else detections[6]
    selected["7. Small Crater (Resolution Limit)"] = sel7

    # Plot 1 row x 7 columns or 2 rows (4 + 3)
    fig, axes = plt.subplots(1, 7, figsize=(21, 3.8), dpi=160)
    qc_metadata = {}

    for idx, (title, det) in enumerate(selected.items()):
        ax = axes[idx]
        cx, cy, r = det.center_x, det.center_y, det.radius_px

        # Extract 2.5x window around crater
        win_size = int(max(64, r * 3.0))
        y0 = int(max(0, cy - win_size))
        y1 = int(min(h_img, cy + win_size))
        x0 = int(max(0, cx - win_size))
        x1 = int(min(w_img, cx + win_size))

        patch = image[y0:y1, x0:x1]
        ax.imshow(patch, cmap="gray")

        # Local coordinates within patch
        local_cx = cx - x0
        local_cy = cy - y0

        # Draw circle and center
        circ = patches.Circle(
            (local_cx, local_cy),
            r,
            fill=False,
            edgecolor="#00FFAA" if "False" not in title else "#FF3333",
            linewidth=1.2,
        )
        ax.add_patch(circ)
        ax.plot(local_cx, local_cy, "+", color="#FF3366", markersize=6)

        ax.set_title(
            f"{title}\nr={r:.1f}px ({det.diameter_m:.1f}m)\nconf={det.confidence:.3f}",
            fontsize=8,
            fontweight="bold",
        )
        ax.axis("off")

        qc_metadata[title] = {
            "center_x": float(cx),
            "center_y": float(cy),
            "radius_px": float(r),
            "diameter_m": float(det.diameter_m),
            "confidence": float(det.confidence),
            "source_sensor": det.source_sensor,
        }

    plt.suptitle(
        f"Qualitative Quality Control Contact Sheet — {sensor_name} Crater Detections",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(p, bbox_inches="tight")
    plt.close(fig)

    return qc_metadata
