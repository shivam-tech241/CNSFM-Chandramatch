"""Robust geometric transformation estimators (RANSAC / Refined Least-Squares).

Provides:
- RANSACEstimator: Robust model estimation resilient to high outlier ratios.
- EstimationResult: Detailed diagnostic results including inlier mask, residuals, and metrics.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np

from src.geometric_validation.models import GeometricModel, SimilarityModel, AffineModel


@dataclass
class EstimationResult:
    """Detailed summary of a geometric model estimation run."""

    model_type: str
    success: bool
    num_candidates: int
    num_inliers: int
    inlier_ratio: float
    threshold_m: float
    model_params: Dict[str, Any] = field(default_factory=dict)
    inlier_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    residuals_m: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    mean_residual_m: float = 0.0
    median_residual_m: float = 0.0
    rmse_m: float = 0.0
    p95_residual_m: float = 0.0
    max_residual_m: float = 0.0
    iterations_run: int = 0
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "success": self.success,
            "num_candidates": self.num_candidates,
            "num_inliers": self.num_inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "threshold_m": round(self.threshold_m, 2),
            "mean_residual_m": round(self.mean_residual_m, 3),
            "median_residual_m": round(self.median_residual_m, 3),
            "rmse_m": round(self.rmse_m, 3),
            "p95_residual_m": round(self.p95_residual_m, 3),
            "max_residual_m": round(self.max_residual_m, 3),
            "iterations_run": self.iterations_run,
            "rejection_reason": self.rejection_reason,
            "model_params": self.model_params,
        }


class RANSACEstimator:
    """RANSAC estimator for 2D spatial transformations."""

    def __init__(
        self,
        max_iterations: int = 2000,
        confidence: float = 0.99,
        random_seed: Optional[int] = 42,
    ):
        self.max_iterations = max_iterations
        self.confidence = confidence
        self.random_seed = random_seed

    def estimate(
        self,
        model_class: Type[GeometricModel],
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        residual_threshold_m: float,
        min_inliers: Optional[int] = None,
    ) -> Tuple[Optional[GeometricModel], EstimationResult]:
        """Estimate the best transformation model using RANSAC with least-squares refinement.

        Args:
            model_class: Type of model to fit (e.g. SimilarityModel, AffineModel).
            src_pts: (N, 2) array of source points in metric space.
            dst_pts: (N, 2) array of destination points in metric space.
            residual_threshold_m: Inlier residual threshold in meters.
            min_inliers: Minimum number of inliers required for acceptance (default: model.min_samples).

        Returns:
            Tuple of (fitted_model, EstimationResult). If failed, model is None.
        """
        src = np.asarray(src_pts, dtype=np.float64)
        dst = np.asarray(dst_pts, dtype=np.float64)
        n = len(src)

        dummy_model = model_class()
        m_samples = dummy_model.min_samples()
        m_type = dummy_model.__class__.__name__

        if min_inliers is None:
            min_inliers = m_samples

        if n < m_samples or len(dst) != n:
            return None, EstimationResult(
                model_type=m_type,
                success=False,
                num_candidates=n,
                num_inliers=0,
                inlier_ratio=0.0,
                threshold_m=residual_threshold_m,
                rejection_reason=f"Insufficient points ({n} < {m_samples})",
            )

        rng = np.random.RandomState(self.random_seed)
        best_inliers: np.ndarray = np.zeros(n, dtype=bool)
        best_inlier_count: int = 0
        best_model: Optional[GeometricModel] = None

        # Number of iterations to achieve target confidence
        adaptive_max_iter = self.max_iterations

        for it in range(self.max_iterations):
            if it >= adaptive_max_iter:
                break

            # Draw random minimal sample
            sample_idx = rng.choice(n, size=m_samples, replace=False)
            src_sample = src[sample_idx]
            dst_sample = dst[sample_idx]

            candidate_model = model_class()
            if not candidate_model.fit(src_sample, dst_sample):
                continue

            # Verify hypothesis is physically plausible before evaluating consensus
            is_plaus, _ = candidate_model.is_plausible()
            if not is_plaus:
                continue

            residuals = candidate_model.compute_residuals(src, dst)
            inliers = residuals <= residual_threshold_m
            inlier_count = int(np.sum(inliers))

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_model = candidate_model

                # Adaptive iteration updating: N = log(1-p) / log(1 - w^m)
                w = float(inlier_count) / float(n)
                if w > 0.0 and w < 1.0:
                    denom = math.log(max(1e-9, 1.0 - (w ** m_samples)))
                    if denom < 0.0:
                        needed = int(math.ceil(math.log(1.0 - self.confidence) / denom))
                        adaptive_max_iter = min(self.max_iterations, max(needed, 50))

        # Check if consensus met minimum inlier threshold
        if best_inlier_count < min_inliers or best_model is None:
            return None, EstimationResult(
                model_type=m_type,
                success=False,
                num_candidates=n,
                num_inliers=best_inlier_count,
                inlier_ratio=float(best_inlier_count / max(n, 1)),
                threshold_m=residual_threshold_m,
                rejection_reason=f"Consensus below minimum inliers ({best_inlier_count} < {min_inliers})",
                iterations_run=it + 1,
            )

        # Refinement: Fit final model on all inliers
        refined_model = model_class()
        refit_ok = refined_model.fit(src[best_inliers], dst[best_inliers])

        if not refit_ok:
            refined_model = best_model

        # Final residual calculation with refined model
        final_residuals = refined_model.compute_residuals(src, dst)
        final_inliers = final_residuals <= residual_threshold_m
        inlier_res = final_residuals[final_inliers]
        final_count = len(inlier_res)

        mean_res = float(np.mean(inlier_res)) if final_count > 0 else 0.0
        median_res = float(np.median(inlier_res)) if final_count > 0 else 0.0
        rmse_res = float(np.sqrt(np.mean(inlier_res ** 2))) if final_count > 0 else 0.0
        p95_res = float(np.percentile(inlier_res, 95)) if final_count > 0 else 0.0
        max_res = float(np.max(inlier_res)) if final_count > 0 else 0.0

        is_plausible, reason = refined_model.is_plausible()

        result = EstimationResult(
            model_type=m_type,
            success=is_plausible,
            num_candidates=n,
            num_inliers=final_count,
            inlier_ratio=float(final_count / max(n, 1)),
            threshold_m=residual_threshold_m,
            model_params=refined_model.get_params(),
            inlier_mask=final_inliers,
            residuals_m=final_residuals,
            mean_residual_m=mean_res,
            median_residual_m=median_res,
            rmse_m=rmse_res,
            p95_residual_m=p95_res,
            max_residual_m=max_res,
            iterations_run=it + 1,
            rejection_reason="" if is_plausible else reason,
        )

        return (refined_model if is_plausible else None), result
