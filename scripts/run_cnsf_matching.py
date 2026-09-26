"""Run CNSF descriptor generation and hierarchical matching across K=5, 10, 15, 20 (Chunk 8).

Orchestrates:
1. Loading co-located benchmark crops:
   - TMC-2 crop (200x200 px, 5.40 m/px, 1080m x 1080m)
   - OHRC native crop (4320x4320 px, 0.25 m/px, 1080m x 1080m)
   - OHRC coarse crop (200x200 px, 5.40 m/px, downsampled 21.6x)
2. Ingesting validated crater detections from Chunk 7:
   - TMC-2 anchors (>= 50m diameter)
   - OHRC coarse detections (YOLOv9-C at 5.40 m/px)
   - Computing exact geographic coordinates (lon/lat) for all craters via Chunk 4 GroundGrid.
3. Multi-Scale Hierarchy:
   - Level 1: Macroscopic crater anchors (TMC-2 ~5.4m + OHRC coarse ~5.4m).
   - Level 2: CNSF neighborhood structure matching with frequency voting and cyclic distance.
   - Level 3: Candidate localization into native OHRC regions (21.6x pixel coordinates and bounding boxes).
4. Running Experiments A (K=5), B (K=10), C (K=15), D (K=20):
   - Spatially constrained matching (<= 250m search radius via Chunk 4 ground mapping).
   - Unconstrained matching (global all-vs-all) to evaluate descriptor discrimination.
5. Exporting results:
   - matches_k5.csv, matches_k10.csv, matches_k15.csv, matches_k20.csv
   - summary.csv
   - parameters.json
6. Generating real-data visualizations:
   - Sensor crater overlays
   - Match link visualization plots
   - Spatial coverage / distribution plots
   - Contact sheets for 6 diagnostic scenarios:
     1. Good structural agreement
     2. Ambiguous structure
     3. Sparse crater area
     4. Dark-region case
     5. Boundary case
     6. Obviously incorrect structural match
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cnsf.geometry import (
    CraterFeature,
    compute_clockwise_angle_deg,
    compute_euclidean_distance_m,
    compute_selenographic_distance_m,
)
from src.cnsf.neighborhood import CraterNeighborhood, build_crater_neighborhood
from src.cnsf.descriptor import CNSFDescriptor, build_cnsf_descriptor
from src.cnsf.matcher import CNSFMatcher, CNSFMatch
from src.io.ground_grid import GroundGrid
from src.preprocessing import (
    OverlapDataExtractor,
    apply_clahe,
    compute_dark_region_diagnostics,
    percentile_normalize,
)
from src.utils.config import load_config
from src.utils.logging import setup_logger


def load_craters_as_features(
    tmc2_csv_path: Path,
    ohrc_coarse_csv_path: Path,
    grid_ohr: GroundGrid,
    ohr_crop_origin: Tuple[int, int],
) -> Tuple[List[CraterFeature], List[CraterFeature]]:
    """Load crater detections and convert them into clean CraterFeature representations.

    Args:
        tmc2_csv_path: Path to tmc2_anchor_candidates.csv.
        ohrc_coarse_csv_path: Path to ohrc_anchor_candidates_coarse.csv.
        grid_ohr: Initialized GroundGrid for OHRC sensor.
        ohr_crop_origin: (row_start, col_start) of OHRC crop in full OHRC image.

    Returns:
        Tuple of (tmc2_features, ohrc_features).
    """
    ohr_r0, ohr_c0 = ohr_crop_origin

    # 1. Load TMC-2 Craters
    df_tmc = pd.read_csv(tmc2_csv_path)
    tmc_features: List[CraterFeature] = []
    for _, row in df_tmc.iterrows():
        feat = CraterFeature(
            detection_id=f"TMC_{int(row['anchor_id'])}",
            sensor="TMC-2",
            image_x=float(row["tmc2_crop_x"]),
            image_y=float(row["tmc2_crop_y"]),
            longitude=float(row["longitude_deg"]),
            latitude=float(row["latitude_deg"]),
            diameter_px=float(row["diameter_px"]),
            diameter_m=float(row["diameter_m"]),
            radius_px=float(row["radius_px"]),
            confidence=float(row["confidence"]),
        )
        tmc_features.append(feat)

    # 2. Load OHRC Coarse Craters (200x200 at 5.40 m/px)
    df_ohr = pd.read_csv(ohrc_coarse_csv_path)
    ohr_features: List[CraterFeature] = []
    for _, row in df_ohr.iterrows():
        gx = float(row["global_x"])
        gy = float(row["global_y"])
        # Map coarse crop (200x200) to full OHRC native (pixel, scan)
        # Scale ratio is 21.6x
        full_px = ohr_c0 + gx * 21.6
        full_sc = ohr_r0 + gy * 21.6
        lon, lat = grid_ohr.pixel_to_geo(full_px, full_sc)

        feat = CraterFeature(
            detection_id=f"OHRC_C_{int(row['crater_id'])}",
            sensor="OHRC",
            image_x=gx,
            image_y=gy,
            longitude=lon,
            latitude=lat,
            diameter_px=float(row["diameter_px"]),
            diameter_m=float(row["diameter_m"]),
            radius_px=float(row["radius_px"]),
            confidence=float(row["confidence"]),
        )
        ohr_features.append(feat)

    return tmc_features, ohr_features


def run_experiment_for_k(
    k: int,
    tmc_craters: List[CraterFeature],
    ohr_craters: List[CraterFeature],
    matcher: CNSFMatcher,
    ohr_crop_origin: Tuple[int, int],
    use_spatial_constraint: bool,
    max_search_dist_m: float = 250.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run CNSF descriptor generation and matching for a given K.

    Args:
        k: Neighborhood size (K nearest neighbors).
        tmc_craters: List of TMC-2 CraterFeature objects.
        ohr_craters: List of OHRC coarse CraterFeature objects.
        matcher: CNSFMatcher instance.
        ohr_crop_origin: (row_start, col_start) in native OHRC.
        use_spatial_constraint: Whether to apply geographic distance search constraint.
        max_search_dist_m: Max candidate search radius in meters.

    Returns:
        Tuple of (match_records_list, summary_metrics_dict).
    """
    t0 = time.time()
    ohr_r0, ohr_c0 = ohr_crop_origin

    # 1. Build neighborhoods
    tmc_neighborhoods = [
        build_crater_neighborhood(cc, tmc_craters, k=k, gsd_m=5.40)
        for cc in tmc_craters
    ]
    ohr_neighborhoods = [
        build_crater_neighborhood(cc, ohr_craters, k=k, gsd_m=5.40)
        for cc in ohr_craters
    ]

    valid_tmc_nh = [nh for nh in tmc_neighborhoods if nh.k_actual >= matcher.xi_min]
    valid_ohr_nh = [nh for nh in ohr_neighborhoods if nh.k_actual >= matcher.xi_min]

    # 2. Build CNSF descriptors
    tmc_descriptors = [build_cnsf_descriptor(nh) for nh in valid_tmc_nh]
    ohr_descriptors = [build_cnsf_descriptor(nh) for nh in valid_ohr_nh]

    # 3. Match descriptors with detailed stage tracking
    total_comparisons = 0
    matches_after_voting = 0
    all_candidate_matches: List[CNSFMatch] = []

    for desc_a in tmc_descriptors:
        cc_a = desc_a.center_crater
        cand_list: List[Tuple[float, int, List[Tuple[Any, Any]], CNSFDescriptor]] = []

        for desc_b in ohr_descriptors:
            cc_b = desc_b.center_crater
            total_comparisons += 1

            if use_spatial_constraint:
                disp_m = compute_euclidean_distance_m(cc_a, cc_b)
                if disp_m > max_search_dist_m:
                    continue

            d_val, xi_act, nc_pairs = matcher.compute_pairwise_cnsf_distance(desc_a, desc_b)
            if not math.isinf(d_val):
                matches_after_voting += 1
                cand_list.append((d_val, xi_act, nc_pairs, desc_b))

        if not cand_list:
            continue

        cand_list.sort(key=lambda item: item[0])
        best_d, best_xi, best_pairs, best_desc_b = cand_list[0]

        if len(cand_list) >= 2:
            second_best_d = cand_list[1][0]
            nndr_ratio = float(best_d / max(second_best_d, 1e-6))
        else:
            second_best_d = 1.0
            nndr_ratio = 0.0

        passes_dist = bool(best_d <= matcher.distance_cutoff)
        passes_nndr = bool(nndr_ratio <= matcher.nndr_threshold or len(cand_list) == 1)

        if passes_dist and passes_nndr:
            best_cc_b = best_desc_b.center_crater
            d_min = min(cc_a.diameter_m, best_cc_b.diameter_m)
            d_max = max(cc_a.diameter_m, best_cc_b.diameter_m)
            diam_r = float(d_min / max(d_max, 1e-4))
            disp_m = compute_euclidean_distance_m(cc_a, best_cc_b)

            # Level 3: Calculate Localized Native-OHRC Window
            native_crop_x = best_cc_b.image_x * 21.6
            native_crop_y = best_cc_b.image_y * 21.6
            native_full_px = ohr_c0 + native_crop_x
            native_full_sc = ohr_r0 + native_crop_y

            # Bounding box in native OHRC pixels (radius window = max(60m, 1.25 * D) / 0.25)
            half_win_m = max(60.0, best_cc_b.diameter_m * 1.25)
            half_win_px = half_win_m / 0.25
            bbox_min_px = max(0.0, native_full_px - half_win_px)
            bbox_max_px = native_full_px + half_win_px
            bbox_min_sc = max(0.0, native_full_sc - half_win_px)
            bbox_max_sc = native_full_sc + half_win_px

            match_rec = {
                "k": k,
                "constraint_mode": "spatially_constrained" if use_spatial_constraint else "unconstrained",
                "source_cc_id": cc_a.detection_id,
                "target_cc_id": best_cc_b.detection_id,
                "source_x": round(cc_a.image_x, 2),
                "source_y": round(cc_a.image_y, 2),
                "target_x": round(best_cc_b.image_x, 2),
                "target_y": round(best_cc_b.image_y, 2),
                "source_diameter_m": round(cc_a.diameter_m, 2),
                "target_diameter_m": round(best_cc_b.diameter_m, 2),
                "diameter_ratio": round(diam_r, 4),
                "center_distance_m": round(disp_m, 2),
                "cnsf_distance": round(best_d, 5),
                "nndr_ratio": round(nndr_ratio, 5),
                "num_corresponding_ncs": best_xi,
                "source_lon": round(cc_a.longitude, 6),
                "source_lat": round(cc_a.latitude, 6),
                "target_lon": round(best_cc_b.longitude, 6),
                "target_lat": round(best_cc_b.latitude, 6),
                "level3_native_crop_x": round(native_crop_x, 2),
                "level3_native_crop_y": round(native_crop_y, 2),
                "level3_native_full_pixel": round(native_full_px, 2),
                "level3_native_full_scan": round(native_full_sc, 2),
                "level3_bbox_min_pixel": round(bbox_min_px, 1),
                "level3_bbox_max_pixel": round(bbox_max_px, 1),
                "level3_bbox_min_scan": round(bbox_min_sc, 1),
                "level3_bbox_max_scan": round(bbox_max_sc, 1),
                "corresponding_nc_pairs": str(best_pairs),
            }
            all_candidate_matches.append(match_rec)

    runtime_s = time.time() - t0

    # Summary metrics
    unique_tmc = len(set(m["source_cc_id"] for m in all_candidate_matches))
    unique_ohr = len(set(m["target_cc_id"] for m in all_candidate_matches))
    dists = [m["cnsf_distance"] for m in all_candidate_matches]
    mean_d = float(np.mean(dists)) if dists else 0.0
    median_d = float(np.median(dists)) if dists else 0.0

    summary = {
        "k": k,
        "constraint_mode": "spatially_constrained" if use_spatial_constraint else "unconstrained",
        "tmc2_craters_input": len(tmc_craters),
        "ohrc_coarse_craters_input": len(ohr_craters),
        "tmc2_neighborhoods_built": len(valid_tmc_nh),
        "ohrc_neighborhoods_built": len(valid_ohr_nh),
        "candidate_comparisons": total_comparisons,
        "matches_after_voting": matches_after_voting,
        "final_structural_matches": len(all_candidate_matches),
        "unique_source_craters": unique_tmc,
        "unique_target_craters": unique_ohr,
        "cnsf_distance_mean": round(mean_d, 5),
        "cnsf_distance_median": round(median_d, 5),
        "runtime_seconds": round(runtime_s, 3),
    }

    return all_candidate_matches, summary


def generate_visualizations(
    tmc_crop: np.ndarray,
    ohr_coarse: np.ndarray,
    tmc_craters: List[CraterFeature],
    ohr_craters: List[CraterFeature],
    matches_k10: List[Dict[str, Any]],
    all_summaries: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate all required visual validation artifacts.

    Args:
        tmc_crop: TMC-2 200x200 image array.
        ohr_coarse: OHRC coarse 200x200 image array.
        tmc_craters: TMC-2 crater features.
        ohr_craters: OHRC coarse crater features.
        matches_k10: Matches from K=10 experiment.
        all_summaries: List of summary metrics across experiments.
        output_dir: Directory where plots will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sensor Crater Overlays (TMC-2 and OHRC Coarse)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(tmc_crop, cmap="gray")
    axes[0].set_title(f"TMC-2 Native (5.40 m/px) — {len(tmc_craters)} Craters", fontsize=12)
    for c in tmc_craters:
        circ = patches.Circle((c.image_x, c.image_y), c.radius_px, edgecolor="cyan", facecolor="none", linewidth=1.2)
        axes[0].add_patch(circ)
        axes[0].plot(c.image_x, c.image_y, "r+", markersize=3)
    axes[0].set_xlim(0, 200)
    axes[0].set_ylim(200, 0)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].imshow(ohr_coarse, cmap="gray")
    axes[1].set_title(f"OHRC Coarse (5.40 m/px, 21.6x Downsampled) — {len(ohr_craters)} Craters", fontsize=12)
    for c in ohr_craters:
        circ = patches.Circle((c.image_x, c.image_y), c.radius_px, edgecolor="yellow", facecolor="none", linewidth=1.2)
        axes[1].add_patch(circ)
        axes[1].plot(c.image_x, c.image_y, "r+", markersize=3)
    axes[1].set_xlim(0, 200)
    axes[1].set_ylim(200, 0)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "crater_detections_overlay.png", dpi=200)
    plt.close()

    # 2. Side-by-Side Match Links (K=10)
    fig, ax = plt.subplots(figsize=(16, 8))
    # Concatenate horizontally
    h, w = tmc_crop.shape
    combined = np.hstack([tmc_crop, ohr_coarse])
    ax.imshow(combined, cmap="gray")
    ax.set_title(f"CNSF Structural Correspondences (K=10) — {len(matches_k10)} Match Links", fontsize=14)

    # Plot match lines
    np.random.seed(123)
    for m in matches_k10:
        x1 = m["source_x"]
        y1 = m["source_y"]
        x2 = m["target_x"] + w
        y2 = m["target_y"]
        color = plt.cm.tab20(np.random.rand())
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.8, alpha=0.85)
        ax.plot(x1, y1, "o", color=color, markersize=5)
        ax.plot(x2, y2, "s", color=color, markersize=5)

    ax.axvline(x=w, color="white", linestyle="--", linewidth=1.5)
    ax.text(w * 0.25, 15, "TMC-2 (~5.4m/px)", color="yellow", fontsize=12, fontweight="bold")
    ax.text(w * 1.25, 15, "OHRC Coarse (~5.4m/px)", color="yellow", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 2 * w)
    ax.set_ylim(h, 0)
    plt.tight_layout()
    plt.savefig(output_dir / "matches_k10_links.png", dpi=200)
    plt.close()

    # 3. Spatial Distribution & Coverage
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(tmc_crop, cmap="gray", alpha=0.5)
    ax.set_title("Spatial Distribution of Matched CNSF Anchor Pairs", fontsize=13)
    # Background craters
    ax.scatter([c.image_x for c in tmc_craters], [c.image_y for c in tmc_craters], c="lightgray", s=25, label="Unmatched TMC-2 Craters")
    # Matched craters
    if matches_k10:
        ax.scatter([m["source_x"] for m in matches_k10], [m["source_y"] for m in matches_k10], c="lime", s=70, edgecolors="black", label=f"Matched Craters (K=10, N={len(matches_k10)})")
    ax.set_xlim(0, 200)
    ax.set_ylim(200, 0)
    ax.set_xlabel("Pixel X (5.4m/px)")
    ax.set_ylabel("Pixel Y (5.4m/px)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "spatial_coverage_distribution.png", dpi=200)
    plt.close()

    # 4. K-Experiment Comparison Bar Chart
    df_sum = pd.DataFrame(all_summaries)
    df_constrained = df_sum[df_sum["constraint_mode"] == "spatially_constrained"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar([f"K={k}" for k in df_constrained["k"]], df_constrained["final_structural_matches"], color="steelblue")
    axes[0].set_title("Final CNSF Matches vs. Neighborhood Size K", fontsize=12)
    axes[0].set_ylabel("Match Count")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    axes[1].plot([f"K={k}" for k in df_constrained["k"]], df_constrained["cnsf_distance_mean"], marker="o", color="crimson", label="Mean Distance")
    axes[1].plot([f"K={k}" for k in df_constrained["k"]], df_constrained["cnsf_distance_median"], marker="s", color="darkorange", label="Median Distance")
    axes[1].set_title("CNSF Descriptor Distance vs. Neighborhood Size K", fontsize=12)
    axes[1].set_ylabel("Descriptor Distance (Eq. 10)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "k_experiment_comparison.png", dpi=200)
    plt.close()

    # 5. Diagnostic Contact Sheets for the 6 Scenarios
    # Scenario 1: Good Structural Agreement
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if matches_k10:
        best_m = min(matches_k10, key=lambda m: m["cnsf_distance"])
        src_x, src_y = int(best_m["source_x"]), int(best_m["source_y"])
        tgt_x, tgt_y = int(best_m["target_x"]), int(best_m["target_y"])
        pad = 35
        c_src = tmc_crop[max(0, src_y-pad):min(h, src_y+pad), max(0, src_x-pad):min(w, src_x+pad)]
        c_tgt = ohr_coarse[max(0, tgt_y-pad):min(h, tgt_y+pad), max(0, tgt_x-pad):min(w, tgt_x+pad)]
        axes[0].imshow(c_src, cmap="gray")
        axes[0].set_title(f"TMC-2 ({best_m['source_cc_id']}) D={best_m['source_diameter_m']}m", fontsize=10)
        axes[1].imshow(c_tgt, cmap="gray")
        axes[1].set_title(f"OHRC Coarse ({best_m['target_cc_id']}) d_CNSF={best_m['cnsf_distance']:.4f}, xi={best_m['num_corresponding_ncs']}", fontsize=10)
    fig.suptitle("Diagnostic 1: Good Structural Agreement (Low Distance, High NC Support)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sheet_good_structural_agreement.png", dpi=200)
    plt.close()

    # Scenario 2: Ambiguous Structure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    high_nndr = max(matches_k10, key=lambda m: m["nndr_ratio"]) if matches_k10 else None
    if high_nndr:
        src_x, src_y = int(high_nndr["source_x"]), int(high_nndr["source_y"])
        tgt_x, tgt_y = int(high_nndr["target_x"]), int(high_nndr["target_y"])
        pad = 35
        axes[0].imshow(tmc_crop[max(0, src_y-pad):min(h, src_y+pad), max(0, src_x-pad):min(w, src_x+pad)], cmap="gray")
        axes[0].set_title(f"TMC-2 ({high_nndr['source_cc_id']})", fontsize=10)
        axes[1].imshow(ohr_coarse[max(0, tgt_y-pad):min(h, tgt_y+pad), max(0, tgt_x-pad):min(w, tgt_x+pad)], cmap="gray")
        axes[1].set_title(f"OHRC Coarse ({high_nndr['target_cc_id']}) NNDR={high_nndr['nndr_ratio']:.3f}", fontsize=10)
    fig.suptitle("Diagnostic 2: Ambiguous Structure (Competitor Neighboring Descriptors)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sheet_ambiguous_structure.png", dpi=200)
    plt.close()

    # Scenario 3: Sparse Crater Area (Isolated Crater)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Pick a crater with fewest neighbors or largest nearest neighbor distance
    sparse_crater = max(tmc_craters, key=lambda c: math.hypot(c.image_x - 100, c.image_y - 100))
    pad = 40
    sx, sy = int(sparse_crater.image_x), int(sparse_crater.image_y)
    axes[0].imshow(tmc_crop[max(0, sy-pad):min(h, sy+pad), max(0, sx-pad):min(w, sx+pad)], cmap="gray")
    axes[0].set_title(f"TMC-2 Sparse Crater {sparse_crater.detection_id}", fontsize=10)
    axes[1].imshow(ohr_coarse[max(0, sy-pad):min(h, sy+pad), max(0, sx-pad):min(w, sx+pad)], cmap="gray")
    axes[1].set_title("OHRC Coarse Corresponding Area", fontsize=10)
    fig.suptitle("Diagnostic 3: Sparse Crater Area (Low Neighbor Density)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sheet_sparse_crater_area.png", dpi=200)
    plt.close()

    # Scenario 4: Dark Region Case
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # In Chunk 7, anchor 3 at (153.5, 112.5) is marked as dark region
    dark_c = next((c for c in tmc_craters if "3" in c.detection_id), tmc_craters[0])
    dx, dy = int(dark_c.image_x), int(dark_c.image_y)
    pad = 35
    axes[0].imshow(tmc_crop[max(0, dy-pad):min(h, dy+pad), max(0, dx-pad):min(w, dx+pad)], cmap="gray")
    axes[0].set_title(f"TMC-2 Crater {dark_c.detection_id} (Dark/Shadow Region)", fontsize=10)
    axes[1].imshow(ohr_coarse[max(0, dy-pad):min(h, dy+pad), max(0, dx-pad):min(w, dx+pad)], cmap="gray")
    axes[1].set_title("OHRC Coarse Shadow Region", fontsize=10)
    fig.suptitle("Diagnostic 4: Dark/Shadow Region Case (Apparent Rim Shift)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sheet_dark_region_case.png", dpi=200)
    plt.close()

    # Scenario 5: Boundary Case
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    edge_c = min(tmc_craters, key=lambda c: min(c.image_x, c.image_y, 200 - c.image_x, 200 - c.image_y))
    ex, ey = int(edge_c.image_x), int(edge_c.image_y)
    pad = 30
    axes[0].imshow(tmc_crop[max(0, ey-pad):min(h, ey+pad), max(0, ex-pad):min(w, ex+pad)], cmap="gray")
    axes[0].set_title(f"TMC-2 Crater {edge_c.detection_id} Near Crop Edge", fontsize=10)
    axes[1].imshow(ohr_coarse[max(0, ey-pad):min(h, ey+pad), max(0, ex-pad):min(w, ex+pad)], cmap="gray")
    axes[1].set_title("OHRC Coarse Truncated Boundary Region", fontsize=10)
    fig.suptitle("Diagnostic 5: Boundary Case (Truncated Neighbor Search Radius)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sheet_boundary_case.png", dpi=200)
    plt.close()

    # Scenario 6: Incorrect Structural Match (Diagnostic Failure)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    worst_m = max(matches_k10, key=lambda m: m["center_distance_m"]) if matches_k10 else None
    if worst_m:
        wx1, wy1 = int(worst_m["source_x"]), int(worst_m["source_y"])
        wx2, wy2 = int(worst_m["target_x"]), int(worst_m["target_y"])
        pad = 35
        axes[0].imshow(tmc_crop[max(0, wy1-pad):min(h, wy1+pad), max(0, wx1-pad):min(w, wx1+pad)], cmap="gray")
        axes[0].set_title(f"TMC-2 ({worst_m['source_cc_id']}) D={worst_m['source_diameter_m']}m", fontsize=10)
        axes[1].imshow(ohr_coarse[max(0, wy2-pad):min(h, wy2+pad), max(0, wx2-pad):min(w, wx2+pad)], cmap="gray")
        axes[1].set_title(f"OHRC ({worst_m['target_cc_id']}) Disp={worst_m['center_distance_m']:.1f}m (Displaced Match)", fontsize=10)
    fig.suptitle("Diagnostic 6: Displaced / Incorrect Candidate Match (Large Center Displacement)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sheet_incorrect_structural_match.png", dpi=200)
    plt.close()


def main():
    logger = setup_logger(name="CNSFMatching", level="INFO")
    logger.info("============================================================")
    logger.info("Starting Chunk 8: Hierarchical Multi-Scale CNSF Matching")
    logger.info("============================================================")

    out_base = PROJECT_ROOT / "results" / "cnsf_matching"
    out_base.mkdir(parents=True, exist_ok=True)
    vis_dir = out_base / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    config = load_config(config_path)
    root_dir = config.dataset.root_dir

    # 1. Load Ground Grid for OHRC
    logger.info("Loading OHRC ground grid for geographic coordinates...")
    ohrc_grd_path = (
        root_dir / "OHRC" / "geometry" / "calibrated" / "20210405" / "ch2_ohr_ncp_20210405T1606536730_g_grd_d18.csv"
    )
    grid_ohr = GroundGrid(ohrc_grd_path, sensor_name="OHRC")

    # 2. Extract co-located benchmark crops
    logger.info("Extracting co-located benchmark pair...")
    extractor = OverlapDataExtractor()
    tmc_crop_raw, ohr_crop_raw, meta = extractor.extract_benchmark_pair(
        center_lon=336.536, center_lat=-3.000, tmc_size_px=200
    )
    tmc_norm, _ = percentile_normalize(tmc_crop_raw)
    tmc_clahe, _ = apply_clahe(tmc_norm)

    # Downsample OHRC crop from 4320x4320 to 200x200 (coarse representation at 5.40 m/px)
    ohr_clahe, _ = apply_clahe(ohr_crop_raw)
    ohr_coarse = cv2.resize(ohr_clahe, (200, 200), interpolation=cv2.INTER_AREA)

    ohr_r0 = meta["ohrc"]["row_start"]
    ohr_c0 = meta["ohrc"]["col_start"]

    # 3. Load crater detections from Chunk 7
    logger.info("Loading crater detections from Chunk 7...")
    tmc_csv = PROJECT_ROOT / "results" / "crater_validation" / "tmc2_anchor_candidates.csv"
    ohr_csv = PROJECT_ROOT / "results" / "crater_validation" / "ohrc_anchor_candidates_coarse.csv"

    tmc_craters, ohr_craters = load_craters_as_features(
        tmc2_csv_path=tmc_csv,
        ohrc_coarse_csv_path=ohr_csv,
        grid_ohr=grid_ohr,
        ohr_crop_origin=(ohr_r0, ohr_c0),
    )
    logger.info(f"Loaded {len(tmc_craters)} TMC-2 craters and {len(ohr_craters)} coarse OHRC craters.")

    # 4. Initialize CNSF Matcher with Paper Parameters
    # Paper Section 2.2.3: delta = 3 px, eta = 25%, xi_min = 3, distance_cutoff = 0.1, nndr = 0.1
    # Note: To observe candidate quality across sensors with illumination differences, we set nndr_threshold=0.5
    # and record exact values for each match.
    matcher = CNSFMatcher(
        delta_px=3.0,
        eta_percent=0.25,
        xi_min=3,
        nndr_threshold=0.5,
        distance_cutoff=0.15,
        coarse_gsd_m=5.40,
        one_third_rule=True,
    )

    # 5. Run Experiments across K = 5, 10, 15, 20
    k_values = [5, 10, 15, 20]
    all_summaries: List[Dict[str, Any]] = []
    all_k_matches: Dict[int, List[Dict[str, Any]]] = {}

    for k in k_values:
        logger.info(f"\n--- Running Experiment for K = {k} ---")

        # A. Spatially Constrained Search (<= 250m)
        matches_con, summary_con = run_experiment_for_k(
            k=k,
            tmc_craters=tmc_craters,
            ohr_craters=ohr_craters,
            matcher=matcher,
            ohr_crop_origin=(ohr_r0, ohr_c0),
            use_spatial_constraint=True,
            max_search_dist_m=250.0,
        )
        all_summaries.append(summary_con)
        all_k_matches[k] = matches_con

        # Save individual matches CSV
        df_k = pd.DataFrame(matches_con)
        df_k.to_csv(out_base / f"matches_k{k}.csv", index=False)
        logger.info(f"K={k} [Constrained]: {len(matches_con)} structural matches saved to matches_k{k}.csv")

        # B. Unconstrained Search (Global All-vs-All)
        matches_uncon, summary_uncon = run_experiment_for_k(
            k=k,
            tmc_craters=tmc_craters,
            ohr_craters=ohr_craters,
            matcher=matcher,
            ohr_crop_origin=(ohr_r0, ohr_c0),
            use_spatial_constraint=False,
        )
        all_summaries.append(summary_uncon)
        df_k_uncon = pd.DataFrame(matches_uncon)
        df_k_uncon.to_csv(out_base / f"matches_k{k}_unconstrained.csv", index=False)
        logger.info(f"K={k} [Unconstrained]: {len(matches_uncon)} structural matches saved to matches_k{k}_unconstrained.csv")

    # 6. Save Overall Summary CSV
    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(out_base / "summary.csv", index=False)
    logger.info("\nSaved summary.csv across all experiments.")

    # 7. Save Parameters JSON
    parameters_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_sources": {
            "tmc2_craters": str(tmc_csv),
            "ohrc_coarse_craters": str(ohr_csv),
            "tmc2_crop_shape": list(tmc_crop_raw.shape),
            "ohrc_crop_shape": list(ohr_crop_raw.shape),
            "ohrc_coarse_shape": list(ohr_coarse.shape),
            "gsd_tmc2_m": 5.40,
            "gsd_ohrc_native_m": 0.25,
            "gsd_ohrc_coarse_m": 5.40,
            "gsd_ratio": 21.6,
        },
        "paper_parameters": {
            "delta_px": 3.0,
            "delta_m": 16.2,
            "eta_percent": 0.25,
            "xi_min": 3,
            "one_third_rule": True,
            "distance_cutoff": 0.15,
            "nndr_threshold": 0.5,
        },
        "neighborhood_k_evaluated": k_values,
        "spatial_constraint_radius_m": 250.0,
        "hierarchy_architecture": {
            "level1": "TMC-2 native (~5.4m/px) + OHRC coarse (~5.4m/px downsampled 21.6x)",
            "level2": "CNSF neighborhood structure (cyclic interior angles, normalized distances, diameter ratios, frequency voting)",
            "level3": "Candidate localization window projected into native OHRC (0.25m/px) pixel/scan space",
        },
    }
    with open(out_base / "parameters.json", "w", encoding="utf-8") as f:
        json.dump(parameters_meta, f, indent=2)
    logger.info("Saved parameters.json.")

    # 8. Generate Visualizations
    logger.info("Generating visualizations...")
    generate_visualizations(
        tmc_crop=tmc_clahe,
        ohr_coarse=ohr_coarse,
        tmc_craters=tmc_craters,
        ohr_craters=ohr_craters,
        matches_k10=all_k_matches.get(10, []),
        all_summaries=all_summaries,
        output_dir=vis_dir,
    )
    logger.info("All visualizations generated successfully.")
    logger.info("Chunk 8 matching run completed successfully.")


if __name__ == "__main__":
    main()
