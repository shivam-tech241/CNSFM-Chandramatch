"""Geometric refinement and validation of CNSF structural candidate correspondences (Chunk 9).

Exports:
- GeometricModel, SimilarityModel, AffineModel
- RANSACEstimator, EstimationResult
- lonlat_to_local_planar_m, compute_candidate_displacements, compute_residual_statistics
- SpatialCoverageMetrics, compute_spatial_coverage
- evaluate_threshold_sensitivity, analyze_residual_correlations, analyze_candidate_consistency
"""

from src.geometric_validation.models import (
    AffineModel,
    GeometricModel,
    SimilarityModel,
)
from src.geometric_validation.estimators import (
    EstimationResult,
    RANSACEstimator,
)
from src.geometric_validation.residuals import (
    compute_candidate_displacements,
    compute_residual_statistics,
    lonlat_to_local_planar_m,
)
from src.geometric_validation.coverage import (
    SpatialCoverageMetrics,
    compute_spatial_coverage,
)
from src.geometric_validation.diagnostics import (
    analyze_candidate_consistency,
    analyze_residual_correlations,
    evaluate_threshold_sensitivity,
)

__all__ = [
    "GeometricModel",
    "SimilarityModel",
    "AffineModel",
    "RANSACEstimator",
    "EstimationResult",
    "lonlat_to_local_planar_m",
    "compute_candidate_displacements",
    "compute_residual_statistics",
    "SpatialCoverageMetrics",
    "compute_spatial_coverage",
    "evaluate_threshold_sensitivity",
    "analyze_residual_correlations",
    "analyze_candidate_consistency",
]
