"""Run geometric refinement and validation of CNSF candidates (Chunk 9).

Orchestrates:
1. Ingesting Chunk 8 CNSF candidate matches across K=5, 10, 15, 20 (both constrained and unconstrained).
2. Converting selenographic coordinates to local metric planar coordinates (meters) centered at benchmark.
3. Computing initial physical displacements for all candidate pairs (candidate_displacements.csv).
4. Estimating robust geometric models (SimilarityModel and AffineModel) via RANSAC.
5. Evaluating residual statistics (RMSE, mean, median, P95, max) in physical meters.
6. Measuring spatial coverage, convex hull area, grid cell occupancy, and nearest-neighbor spacing.
7. Conducting residual threshold sensitivity analysis across [10, 20, 30, 50, 75, 100, 150] meters.
8. Investigating correlation between geometric residuals and physical crater properties (diameter, CNSF distance, etc.).
9. Exporting all required tables:
   - summary.csv
   - threshold_sensitivity.csv
   - k_comparison.csv
   - model_comparison.csv
   - candidate_displacements.csv
   - geometric_inliers.csv
   - geometric_outliers.csv
   - parameters.json
10. Generating 12 diagnostic visualization figures.
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
from scipy.spatial import ConvexHull

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometric_validation.models import SimilarityModel, AffineModel
from src.geometric_validation.estimators import RANSACEstimator, EstimationResult
from src.geometric_validation.residuals import (
    compute_candidate_displacements,
    compute_residual_statistics,
    lonlat_to_local_planar_m,
    BENCHMARK_CENTER_LON,
    BENCHMARK_CENTER_LAT,
)
from src.geometric_validation.coverage import (
    compute_spatial_coverage,
    SpatialCoverageMetrics,
)
from src.geometric_validation.diagnostics import (
    analyze_candidate_consistency,
    analyze_residual_correlations,
    evaluate_threshold_sensitivity,
)
from src.preprocessing import OverlapDataExtractor, apply_clahe, percentile_normalize
from src.utils.config import load_config
from src.utils.logging import setup_logger


def load_all_cnsf_candidates(
    cnsf_results_dir: Path,
    k_values: List[int] = [5, 10, 15, 20],
) -> Dict[str, pd.DataFrame]:
    """Load all Chunk 8 candidate files for constrained and unconstrained runs.

    Returns:
        Dictionary mapping key (e.g. 'k10_constrained', 'k10_unconstrained') to DataFrame.
    """
    candidate_tables: Dict[str, pd.DataFrame] = {}

    for k in k_values:
        fc = cnsf_results_dir / f"matches_k{k}.csv"
        fu = cnsf_results_dir / f"matches_k{k}_unconstrained.csv"

        if fc.exists():
            df_c = pd.read_csv(fc)
            df_c["k"] = k
            df_c["mode"] = "spatially_constrained"
            candidate_tables[f"k{k}_constrained"] = compute_candidate_displacements(df_c)

        if fu.exists():
            df_u = pd.read_csv(fu)
            df_u["k"] = k
            df_u["mode"] = "unconstrained"
            candidate_tables[f"k{k}_unconstrained"] = compute_candidate_displacements(df_u)

    return candidate_tables


def generate_visualizations(
    tmc_crop: np.ndarray,
    ohr_coarse: np.ndarray,
    df_displacements: pd.DataFrame,
    inliers_df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    sim_result: EstimationResult,
    aff_result: EstimationResult,
    cov_metrics: SpatialCoverageMetrics,
    df_sensitivity: pd.DataFrame,
    df_k_comp: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate all 12 required visualization artifacts for Chunk 9."""
    output_dir.mkdir(parents=True, exist_ok=True)
    h, w = tmc_crop.shape

    # 1. Candidate Displacement Distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    constrained_disp = df_displacements[df_displacements["mode"] == "spatially_constrained"]["displacement_m"]
    unconstrained_disp = df_displacements[df_displacements["mode"] == "unconstrained"]["displacement_m"]

    axes[0].hist(constrained_disp, bins=15, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].set_title(f"Spatially Constrained Initial Displacements (N={len(constrained_disp)})", fontsize=11)
    axes[0].set_xlabel("Displacement (m)")
    axes[0].set_ylabel("Candidate Count")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].hist(unconstrained_disp, bins=25, color="darkorange", edgecolor="black", alpha=0.7)
    axes[1].set_title(f"Unconstrained Initial Displacements (N={len(unconstrained_disp)})", fontsize=11)
    axes[1].set_xlabel("Displacement (m)")
    axes[1].set_ylabel("Candidate Count")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_dir / "candidate_displacement_distribution.png", dpi=200)
    plt.close()

    # 2. Similarity Model Inliers vs Outliers
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(tmc_crop, cmap="gray", alpha=0.6)
    ax.set_title(f"Similarity Model: {sim_result.num_inliers} Inliers vs {len(outliers_df)} Outliers (Thresh={sim_result.threshold_m}m)", fontsize=12)

    if not outliers_df.empty:
        ax.scatter(outliers_df["source_x"], outliers_df["source_y"], c="red", marker="x", s=50, label=f"Geometric Outliers (N={len(outliers_df)})", zorder=3)
    if not inliers_df.empty:
        ax.scatter(inliers_df["source_x"], inliers_df["source_y"], c="lime", edgecolors="black", s=60, label=f"Geometric Inliers (N={len(inliers_df)})", zorder=4)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("TMC-2 Pixel X (5.4m/px)")
    ax.set_ylabel("TMC-2 Pixel Y (5.4m/px)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "similarity_model_inliers_outliers.png", dpi=200)
    plt.close()

    # 3. Affine Model Inliers vs Outliers
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(tmc_crop, cmap="gray", alpha=0.6)
    ax.set_title(f"Affine Model: {aff_result.num_inliers} Inliers vs {len(outliers_df)} Outliers (RMSE={aff_result.rmse_m:.2f}m)", fontsize=12)

    if not outliers_df.empty:
        ax.scatter(outliers_df["source_x"], outliers_df["source_y"], c="crimson", marker="x", s=50, label=f"Outliers (N={len(outliers_df)})", zorder=3)
    if not inliers_df.empty:
        ax.scatter(inliers_df["source_x"], inliers_df["source_y"], c="cyan", edgecolors="blue", s=60, label=f"Affine Inliers (N={len(inliers_df)})", zorder=4)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("TMC-2 Pixel X (5.4m/px)")
    ax.set_ylabel("TMC-2 Pixel Y (5.4m/px)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "affine_model_inliers_outliers.png", dpi=200)
    plt.close()

    # 4. Spatial Coverage Grid
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(tmc_crop, cmap="gray", alpha=0.5)
    ax.set_title(f"Spatial Coverage Grid (4x4 Cells) — Occupancy: {cov_metrics.occupied_cells}/16 ({cov_metrics.cell_occupancy_ratio*100:.1f}%)", fontsize=12)

    # Draw 4x4 grid lines
    grid_n = 4
    for i in range(1, grid_n):
        ax.axvline(x=i * (w / grid_n), color="yellow", linestyle="--", alpha=0.5)
        ax.axhline(y=i * (h / grid_n), color="yellow", linestyle="--", alpha=0.5)

    if not inliers_df.empty:
        inlier_pts_px = inliers_df[["source_x", "source_y"]].values
        ax.scatter(inlier_pts_px[:, 0], inlier_pts_px[:, 1], c="lime", edgecolors="black", s=70, label="Geometric Inliers", zorder=4)

        # Convex hull in pixel space if >= 3
        if len(inlier_pts_px) >= 3:
            try:
                hull = ConvexHull(inlier_pts_px)
                for simplex in hull.simplices:
                    ax.plot(inlier_pts_px[simplex, 0], inlier_pts_px[simplex, 1], "cyan", linewidth=2.0)
            except Exception:
                pass

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "spatial_coverage_grid.png", dpi=200)
    plt.close()

    # 5. Residual Distribution Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    if not inliers_df.empty:
        res = inliers_df["residual_m"]
        ax.hist(res, bins=12, color="mediumseagreen", edgecolor="black", alpha=0.8)
        ax.axvline(x=float(np.mean(res)), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(res):.2f}m")
        ax.axvline(x=float(np.median(res)), color="darkorange", linestyle="-.", linewidth=2, label=f"Median: {np.median(res):.2f}m")
        ax.axvline(x=float(np.sqrt(np.mean(res**2))), color="purple", linestyle=":", linewidth=2, label=f"RMSE: {np.sqrt(np.mean(res**2)):.2f}m")
        ax.set_title("Geometric Inlier Residual Distribution (Meters)", fontsize=12)
        ax.set_xlabel("Residual Error (m)")
        ax.set_ylabel("Inlier Count")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "residual_distribution_histogram.png", dpi=200)
    plt.close()

    # 6. Residual vs CNSF Distance
    fig, ax = plt.subplots(figsize=(8, 5))
    if not inliers_df.empty and "cnsf_distance" in inliers_df.columns:
        x_cnsf = inliers_df["cnsf_distance"]
        y_res = inliers_df["residual_m"]
        ax.scatter(x_cnsf, y_res, c="royalblue", edgecolors="black", s=50, alpha=0.85)

        # Linear regression trend line
        if len(x_cnsf) > 2:
            p = np.polyfit(x_cnsf, y_res, 1)
            x_line = np.linspace(min(x_cnsf), max(x_cnsf), 50)
            ax.plot(x_line, np.polyval(p, x_line), "r--", label=f"Linear Fit (Slope={p[0]:.2f})")

        ax.set_title("Geometric Residual vs. CNSF Structural Distance", fontsize=12)
        ax.set_xlabel("CNSF Distance d_CNSF (Eq. 10)")
        ax.set_ylabel("Geometric Residual (m)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_cnsf_distance.png", dpi=200)
    plt.close()

    # 7. Residual vs Crater Diameter
    fig, ax = plt.subplots(figsize=(8, 5))
    if not inliers_df.empty and "source_diameter_m" in inliers_df.columns:
        ax.scatter(inliers_df["source_diameter_m"], inliers_df["residual_m"], c="darkviolet", edgecolors="black", s=50)
        ax.set_title("Geometric Residual vs. Central Crater Diameter", fontsize=12)
        ax.set_xlabel("Source Crater Diameter (m)")
        ax.set_ylabel("Geometric Residual (m)")
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_crater_diameter.png", dpi=200)
    plt.close()

    # 8. Residual vs Initial Displacement
    fig, ax = plt.subplots(figsize=(8, 5))
    if not inliers_df.empty and "displacement_m" in inliers_df.columns:
        ax.scatter(inliers_df["displacement_m"], inliers_df["residual_m"], c="teal", edgecolors="black", s=50)
        ax.set_title("Geometric Residual vs. Initial Displacement", fontsize=12)
        ax.set_xlabel("Initial Unmodeled Displacement (m)")
        ax.set_ylabel("Post-Model Residual (m)")
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_confidence.png", dpi=200)
    plt.close()

    # 9. K Comparison (Inliers and RMSE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    df_k_sub = df_k_comp[df_k_comp["mode"] == "spatially_constrained"]
    axes[0].bar([f"K={int(k)}" for k in df_k_sub["k"]], df_k_sub["inliers"], color="steelblue", edgecolor="black")
    axes[0].set_title("Geometric Inlier Count vs. Neighborhood Size K (Constrained)", fontsize=11)
    axes[0].set_ylabel("Inlier Count")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    axes[1].plot([f"K={int(k)}" for k in df_k_sub["k"]], df_k_sub["rmse_m"], marker="o", color="crimson", label="RMSE")
    axes[1].plot([f"K={int(k)}" for k in df_k_sub["k"]], df_k_sub["median_residual_m"], marker="s", color="darkorange", label="Median Residual")
    axes[1].set_title("Geometric Residuals (m) vs. Neighborhood Size K", fontsize=11)
    axes[1].set_ylabel("Residual (m)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "k_comparison.png", dpi=200)
    plt.close()

    # 10. Threshold Sensitivity Curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(df_sensitivity["threshold_m"], df_sensitivity["num_inliers"], marker="o", color="forestgreen", linewidth=2)
    axes[0].set_title("Inlier Count vs. Inlier Residual Threshold", fontsize=11)
    axes[0].set_xlabel("Residual Threshold T_res (m)")
    axes[0].set_ylabel("Inlier Count")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(df_sensitivity["threshold_m"], df_sensitivity["rmse_m"], marker="s", color="firebrick", linewidth=2)
    axes[1].set_title("Inlier RMSE vs. Inlier Residual Threshold", fontsize=11)
    axes[1].set_xlabel("Residual Threshold T_res (m)")
    axes[1].set_ylabel("Inlier RMSE (m)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "threshold_sensitivity_curve.png", dpi=200)
    plt.close()

    # 11. Example Good Geometric Consistency
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if not inliers_df.empty:
        best_inlier = inliers_df.sort_values("residual_m").iloc[0]
        bx, by = int(best_inlier["source_x"]), int(best_inlier["source_y"])
        tx, ty = int(best_inlier["target_x"]), int(best_inlier["target_y"])
        pad = 35
        axes[0].imshow(tmc_crop[max(0, by-pad):min(h, by+pad), max(0, bx-pad):min(w, bx+pad)], cmap="gray")
        axes[0].set_title(f"TMC-2 ({best_inlier['source_cc_id']}) D={best_inlier['source_diameter_m']}m", fontsize=10)
        axes[1].imshow(ohr_coarse[max(0, ty-pad):min(h, ty+pad), max(0, tx-pad):min(w, tx+pad)], cmap="gray")
        axes[1].set_title(f"OHRC ({best_inlier['target_cc_id']}) Residual={best_inlier['residual_m']:.2f}m", fontsize=10)
    fig.suptitle("Example Good Geometric Consistency (Verified Inlier)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "example_good_geometric_consistency.png", dpi=200)
    plt.close()

    # 12. Example Bad / Ambiguous Consistency
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if not outliers_df.empty:
        worst_outlier = outliers_df.sort_values("residual_m", ascending=False).iloc[0]
        ox, oy = int(worst_outlier["source_x"]), int(worst_outlier["source_y"])
        otx, oty = int(worst_outlier["target_x"]), int(worst_outlier["target_y"])
        pad = 35
        axes[0].imshow(tmc_crop[max(0, oy-pad):min(h, oy+pad), max(0, ox-pad):min(w, ox+pad)], cmap="gray")
        axes[0].set_title(f"TMC-2 ({worst_outlier['source_cc_id']}) d_CNSF={worst_outlier['cnsf_distance']:.4f}", fontsize=10)
        axes[1].imshow(ohr_coarse[max(0, oty-pad):min(h, oty+pad), max(0, otx-pad):min(w, otx+pad)], cmap="gray")
        axes[1].set_title(f"OHRC ({worst_outlier['target_cc_id']}) Residual={worst_outlier['residual_m']:.2f}m (Rejected)", fontsize=10)
    fig.suptitle("Example Bad/Ambiguous Consistency (Geometric Outlier)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "example_bad_ambiguous_consistency.png", dpi=200)
    plt.close()


def main():
    logger = setup_logger(name="GeometricValidation", level="INFO")
    logger.info("============================================================")
    logger.info("Starting Chunk 9: Geometric Refinement & Validation")
    logger.info("============================================================")

    out_base = PROJECT_ROOT / "results" / "geometric_validation"
    out_base.mkdir(parents=True, exist_ok=True)
    vis_dir = out_base / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    cnsf_dir = PROJECT_ROOT / "results" / "cnsf_matching"

    # 1. Load benchmark imagery
    logger.info("Loading co-located benchmark crops...")
    extractor = OverlapDataExtractor()
    tmc_crop_raw, ohr_crop_raw, meta = extractor.extract_benchmark_pair(
        center_lon=BENCHMARK_CENTER_LON, center_lat=BENCHMARK_CENTER_LAT, tmc_size_px=200
    )
    tmc_norm, _ = percentile_normalize(tmc_crop_raw)
    tmc_clahe, _ = apply_clahe(tmc_norm)
    ohr_clahe, _ = apply_clahe(ohr_crop_raw)
    ohr_coarse = cv2.resize(ohr_clahe, (200, 200), interpolation=cv2.INTER_AREA)

    # 2. Ingest all Chunk 8 candidates
    logger.info("Loading Chunk 8 candidate correspondences...")
    candidate_tables = load_all_cnsf_candidates(cnsf_dir)
    logger.info(f"Loaded {len(candidate_tables)} candidate datasets.")

    # 3. Export Candidate Displacements
    all_disp_list = []
    for key, df in candidate_tables.items():
        all_disp_list.append(df)
    df_all_disp = pd.concat(all_disp_list, ignore_index=True)
    df_all_disp.to_csv(out_base / "candidate_displacements.csv", index=False)
    logger.info(f"Saved candidate_displacements.csv ({len(df_all_disp)} total candidate instances).")

    # 4. Model Estimations across K and Search Modes
    k_records: List[Dict[str, Any]] = []
    model_records: List[Dict[str, Any]] = []
    summary_records: List[Dict[str, Any]] = []

    estimator = RANSACEstimator(max_iterations=2000, random_seed=42)
    default_threshold_m = 30.0  # Physical inlier threshold in meters (~5.5 pixels at 5.4m/px)

    primary_inliers_df = pd.DataFrame()
    primary_outliers_df = pd.DataFrame()
    primary_sim_result: Optional[EstimationResult] = None
    primary_aff_result: Optional[EstimationResult] = None
    primary_cov_metrics: Optional[SpatialCoverageMetrics] = None

    for key, df_cand in candidate_tables.items():
        k_val = int(df_cand["k"].iloc[0])
        mode_val = str(df_cand["mode"].iloc[0])
        n_cand = len(df_cand)

        src_pts = df_cand[["src_X_m", "src_Y_m"]].values
        tgt_pts = df_cand[["tgt_X_m", "tgt_Y_m"]].values

        t0 = time.time()
        sim_model, sim_res = estimator.estimate(
            SimilarityModel, src_pts, tgt_pts, residual_threshold_m=default_threshold_m
        )
        t_sim = time.time() - t0

        t0 = time.time()
        aff_model, aff_res = estimator.estimate(
            AffineModel, src_pts, tgt_pts, residual_threshold_m=default_threshold_m
        )
        t_aff = time.time() - t0

        # Spatial coverage on inliers
        inlier_mask = aff_res.inlier_mask if aff_res.success else sim_res.inlier_mask
        inlier_pts = src_pts[inlier_mask] if len(inlier_mask) > 0 else np.zeros((0, 2))
        cov = compute_spatial_coverage(inlier_pts)

        # Record K comparison (using Affine as reference model)
        k_records.append({
            "k": k_val,
            "mode": mode_val,
            "candidates": n_cand,
            "inliers": aff_res.num_inliers,
            "inlier_ratio": aff_res.inlier_ratio,
            "mean_residual_m": aff_res.mean_residual_m,
            "median_residual_m": aff_res.median_residual_m,
            "rmse_m": aff_res.rmse_m,
            "p95_m": aff_res.p95_residual_m,
            "max_residual_m": aff_res.max_residual_m,
            "occupied_cells": cov.occupied_cells,
            "occupancy_ratio": cov.cell_occupancy_ratio,
            "runtime_s": round(t_aff, 3),
        })

        # Record Model comparison for this combination
        model_records.append({
            "k": k_val,
            "mode": mode_val,
            "model": "Similarity",
            "success": sim_res.success,
            "inliers": sim_res.num_inliers,
            "inlier_ratio": sim_res.inlier_ratio,
            "mean_residual_m": sim_res.mean_residual_m,
            "median_residual_m": sim_res.median_residual_m,
            "rmse_m": sim_res.rmse_m,
            "scale": sim_res.model_params.get("scale", None),
            "rotation_deg": sim_res.model_params.get("rotation_deg", None),
            "tx_m": sim_res.model_params.get("translation_x_m", None),
            "ty_m": sim_res.model_params.get("translation_y_m", None),
        })
        model_records.append({
            "k": k_val,
            "mode": mode_val,
            "model": "Affine",
            "success": aff_res.success,
            "inliers": aff_res.num_inliers,
            "inlier_ratio": aff_res.inlier_ratio,
            "mean_residual_m": aff_res.mean_residual_m,
            "median_residual_m": aff_res.median_residual_m,
            "rmse_m": aff_res.rmse_m,
            "det": aff_res.model_params.get("determinant", None),
            "cond": aff_res.model_params.get("condition_number", None),
            "anisotropy": aff_res.model_params.get("scale_anisotropy", None),
            "tx_m": aff_res.model_params.get("translation_x_m", None),
            "ty_m": aff_res.model_params.get("translation_y_m", None),
        })

        # Save primary reference configuration (K=10, constrained)
        if k_val == 10 and mode_val == "spatially_constrained":
            primary_sim_result = sim_res
            primary_aff_result = aff_res
            primary_cov_metrics = cov

            df_cand_evaluated = df_cand.copy()
            df_cand_evaluated["residual_m"] = np.round(aff_res.residuals_m, 2)
            df_cand_evaluated["is_inlier"] = aff_res.inlier_mask
            df_cand_evaluated["status"] = np.where(aff_res.inlier_mask, "GEOMETRIC_INLIER", "GEOMETRIC_OUTLIER")

            primary_inliers_df = df_cand_evaluated[df_cand_evaluated["is_inlier"]].copy()
            primary_outliers_df = df_cand_evaluated[~df_cand_evaluated["is_inlier"]].copy()

    # Save k_comparison.csv and model_comparison.csv
    df_k_comp = pd.DataFrame(k_records)
    df_k_comp.to_csv(out_base / "k_comparison.csv", index=False)
    logger.info("Saved k_comparison.csv")

    df_model_comp = pd.DataFrame(model_records)
    df_model_comp.to_csv(out_base / "model_comparison.csv", index=False)
    logger.info("Saved model_comparison.csv")

    # 5. Save Inliers and Outliers
    primary_inliers_df.to_csv(out_base / "geometric_inliers.csv", index=False)
    primary_outliers_df.to_csv(out_base / "geometric_outliers.csv", index=False)
    logger.info(f"Saved geometric_inliers.csv ({len(primary_inliers_df)} rows) and geometric_outliers.csv ({len(primary_outliers_df)} rows).")

    # 6. Threshold Sensitivity Sweep (on K=10 constrained)
    logger.info("Evaluating threshold sensitivity on K=10 constrained candidates...")
    k10_df = candidate_tables["k10_constrained"]
    src_k10 = k10_df[["src_X_m", "src_Y_m"]].values
    tgt_k10 = k10_df[["tgt_X_m", "tgt_Y_m"]].values
    df_sensitivity = evaluate_threshold_sensitivity(
        AffineModel,
        src_k10,
        tgt_k10,
        thresholds_m=[10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0],
        estimator=estimator,
    )
    df_sensitivity.to_csv(out_base / "threshold_sensitivity.csv", index=False)
    logger.info("Saved threshold_sensitivity.csv")

    # 7. Summary CSV
    df_summary = pd.DataFrame(k_records)
    df_summary.to_csv(out_base / "summary.csv", index=False)
    logger.info("Saved summary.csv")

    # 8. Diagnostics: Correlations & Consistency
    df_eval_full = pd.concat([primary_inliers_df, primary_outliers_df], ignore_index=True)
    corr_results = analyze_residual_correlations(df_eval_full)
    consistency_results = analyze_candidate_consistency(df_eval_full)

    # 9. Parameters JSON
    parameters_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(cnsf_dir),
        "benchmark_center": {
            "longitude_deg": BENCHMARK_CENTER_LON,
            "latitude_deg": BENCHMARK_CENTER_LAT,
        },
        "default_residual_threshold_m": default_threshold_m,
        "ransac_parameters": {
            "max_iterations": estimator.max_iterations,
            "confidence": estimator.confidence,
            "random_seed": estimator.random_seed,
        },
        "sensitivity_thresholds_m": [10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0],
        "primary_benchmark_results": {
            "k": 10,
            "mode": "spatially_constrained",
            "candidate_count": len(df_eval_full),
            "inlier_count": len(primary_inliers_df),
            "inlier_ratio": round(len(primary_inliers_df) / max(len(df_eval_full), 1), 4),
            "similarity_model": primary_sim_result.to_dict() if primary_sim_result else {},
            "affine_model": primary_aff_result.to_dict() if primary_aff_result else {},
            "spatial_coverage": primary_cov_metrics.to_dict() if primary_cov_metrics else {},
        },
        "diagnostics": {
            "correlations": corr_results,
            "consistency_groupings": consistency_results,
        },
    }
    with open(out_base / "parameters.json", "w", encoding="utf-8") as f:
        json.dump(parameters_meta, f, indent=2)
    logger.info("Saved parameters.json")

    # 10. Visualizations
    logger.info("Generating 12 visualization figures...")
    generate_visualizations(
        tmc_crop=tmc_clahe,
        ohr_coarse=ohr_coarse,
        df_displacements=df_all_disp,
        inliers_df=primary_inliers_df,
        outliers_df=primary_outliers_df,
        sim_result=primary_sim_result,
        aff_result=primary_aff_result,
        cov_metrics=primary_cov_metrics,
        df_sensitivity=df_sensitivity,
        df_k_comp=df_k_comp,
        output_dir=vis_dir,
    )
    logger.info("All 12 visualizations successfully generated.")
    logger.info("Chunk 9 geometric validation run completed successfully.")


if __name__ == "__main__":
    main()
