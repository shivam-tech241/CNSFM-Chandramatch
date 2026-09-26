"""Diagnostic evaluation, sensitivity analysis, and property correlation routines (Chunk 9).

Provides:
- evaluate_threshold_sensitivity: Evaluates model inlier survival across multiple residual thresholds.
- analyze_residual_correlations: Statistical correlation between geometric residuals and physical crater properties.
- analyze_candidate_consistency: Compares CNSF structural distance against geometric validity.
"""

from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.geometric_validation.models import GeometricModel, SimilarityModel, AffineModel
from src.geometric_validation.estimators import RANSACEstimator, EstimationResult


def evaluate_threshold_sensitivity(
    model_class: Type[GeometricModel],
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    thresholds_m: List[float] = [10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0],
    estimator: Optional[RANSACEstimator] = None,
) -> pd.DataFrame:
    """Evaluate inlier count, ratio, and residual metrics across a sweep of residual thresholds.

    Args:
        model_class: Model class (SimilarityModel or AffineModel).
        src_pts: (N, 2) source points in metric coordinates.
        dst_pts: (N, 2) destination points in metric coordinates.
        thresholds_m: List of residual thresholds to test in meters.
        estimator: Configured RANSACEstimator instance.

    Returns:
        DataFrame recording metrics for each threshold.
    """
    if estimator is None:
        estimator = RANSACEstimator(max_iterations=1000, random_seed=42)

    records: List[Dict[str, Any]] = []
    m_name = model_class().__class__.__name__

    for thresh in thresholds_m:
        model, est_res = estimator.estimate(
            model_class=model_class,
            src_pts=src_pts,
            dst_pts=dst_pts,
            residual_threshold_m=thresh,
        )
        records.append({
            "model_type": m_name,
            "threshold_m": thresh,
            "success": est_res.success,
            "num_candidates": est_res.num_candidates,
            "num_inliers": est_res.num_inliers,
            "inlier_ratio": round(est_res.inlier_ratio, 4),
            "mean_residual_m": round(est_res.mean_residual_m, 3),
            "median_residual_m": round(est_res.median_residual_m, 3),
            "rmse_m": round(est_res.rmse_m, 3),
            "p95_residual_m": round(est_res.p95_residual_m, 3),
            "max_residual_m": round(est_res.max_residual_m, 3),
            "iterations_run": est_res.iterations_run,
            "rejection_reason": est_res.rejection_reason,
        })

    return pd.DataFrame(records)


def analyze_residual_correlations(
    df_eval: pd.DataFrame,
    residual_col: str = "residual_m",
) -> Dict[str, Any]:
    """Analyze statistical correlation between geometric residuals and crater properties.

    Checks:
    - CNSF descriptor distance vs residual
    - Crater diameter vs residual
    - Diameter ratio vs residual
    - Detection confidence vs residual

    Args:
        df_eval: DataFrame containing candidates with residual_m and property columns.
        residual_col: Name of column containing geometric residuals.

    Returns:
        Dictionary of correlation coefficients, p-values, and category breakdowns.
    """
    results: Dict[str, Any] = {}
    valid_df = df_eval.dropna(subset=[residual_col]).copy()
    n = len(valid_df)

    if n < 3:
        return {"error": "Insufficient points for correlation analysis (N < 3)."}

    res_arr = valid_df[residual_col].values

    # 1. Residual vs CNSF Distance
    if "cnsf_distance" in valid_df.columns:
        cnsf_dist = valid_df["cnsf_distance"].values
        pr_val, pr_p = pearsonr(cnsf_dist, res_arr)
        sp_val, sp_p = spearmanr(cnsf_dist, res_arr)
        results["cnsf_distance"] = {
            "pearson_r": round(float(pr_val), 4),
            "pearson_p": round(float(pr_p), 5),
            "spearman_rho": round(float(sp_val), 4),
            "spearman_p": round(float(sp_p), 5),
        }

    # 2. Residual vs Source Diameter
    if "source_diameter_m" in valid_df.columns:
        src_diam = valid_df["source_diameter_m"].values
        pr_val, pr_p = pearsonr(src_diam, res_arr)
        sp_val, sp_p = spearmanr(src_diam, res_arr)
        results["source_diameter_m"] = {
            "pearson_r": round(float(pr_val), 4),
            "pearson_p": round(float(pr_p), 5),
            "spearman_rho": round(float(sp_val), 4),
            "spearman_p": round(float(sp_p), 5),
        }

    # 3. Residual vs Diameter Ratio
    if "diameter_ratio" in valid_df.columns:
        diam_r = valid_df["diameter_ratio"].values
        pr_val, pr_p = pearsonr(diam_r, res_arr)
        sp_val, sp_p = spearmanr(diam_r, res_arr)
        results["diameter_ratio"] = {
            "pearson_r": round(float(pr_val), 4),
            "pearson_p": round(float(pr_p), 5),
            "spearman_rho": round(float(sp_val), 4),
            "spearman_p": round(float(sp_p), 5),
        }

    # 4. Residual vs Initial Displacement
    if "displacement_m" in valid_df.columns:
        disp_arr = valid_df["displacement_m"].values
        pr_val, pr_p = pearsonr(disp_arr, res_arr)
        results["initial_displacement_m"] = {
            "pearson_r": round(float(pr_val), 4),
            "pearson_p": round(float(pr_p), 5),
        }

    return results


def analyze_candidate_consistency(
    df_eval: pd.DataFrame,
    residual_col: str = "residual_m",
    cnsf_dist_col: str = "cnsf_distance",
    low_cnsf_cutoff: float = 0.03,
    high_residual_cutoff_m: float = 50.0,
    moderate_cnsf_cutoff: float = 0.05,
    low_residual_cutoff_m: float = 25.0,
) -> Dict[str, Any]:
    """Identify anomalous candidate groupings (e.g. low CNSF dist but high residual).

    Args:
        df_eval: DataFrame containing candidates with residual_m and cnsf_distance.

    Returns:
        Summary of consistency groupings.
    """
    df = df_eval.dropna(subset=[residual_col, cnsf_dist_col]).copy()

    # Case A: Low CNSF distance but high geometric residual (structural false positive)
    case_a = df[(df[cnsf_dist_col] <= low_cnsf_cutoff) & (df[residual_col] > high_residual_cutoff_m)]

    # Case B: Moderate CNSF distance but strong geometric inlier (robust structural match under noise)
    case_b = df[(df[cnsf_dist_col] >= moderate_cnsf_cutoff) & (df[residual_col] <= low_residual_cutoff_m)]

    # Case C: Low CNSF distance and low residual (ideal correspondence)
    case_c = df[(df[cnsf_dist_col] <= low_cnsf_cutoff) & (df[residual_col] <= low_residual_cutoff_m)]

    return {
        "total_evaluated": len(df),
        "ideal_matches_low_cnsf_low_res": len(case_c),
        "structural_false_positives_low_cnsf_high_res": len(case_a),
        "robust_matches_mod_cnsf_low_res": len(case_b),
        "ideal_match_examples": case_c[["source_cc_id", "target_cc_id", cnsf_dist_col, residual_col]].head(3).to_dict("records"),
        "false_positive_examples": case_a[["source_cc_id", "target_cc_id", cnsf_dist_col, residual_col]].head(3).to_dict("records"),
    }
