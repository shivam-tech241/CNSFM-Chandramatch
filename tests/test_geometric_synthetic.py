"""Synthetic geometric validation tests (Chunk 9 Part 15).

Tests:
1. Known similarity transformation recovery.
2. Known affine transformation recovery.
3. Robustness to Gaussian coordinate noise (sigma = 2.0 m).
4. Outlier correspondence rejection (40% gross outliers eliminated by RANSAC).
5. Clustered correspondences (spatial coverage validation).
6. Poorly conditioned / collinear geometry handling.
7. Under-determined correspondence set (N < min_samples).
"""

import math
from pathlib import Path
import sys
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometric_validation.models import SimilarityModel, AffineModel
from src.geometric_validation.estimators import RANSACEstimator
from src.geometric_validation.coverage import compute_spatial_coverage


class TestGeometricSynthetic(unittest.TestCase):
    """Synthetic algorithmic validation of geometric estimators and coverage metrics."""

    def setUp(self):
        np.random.seed(42)
        # 20 synthetic well-distributed ground coordinates in a 1000m x 1000m region
        self.src_pts = np.random.uniform(50.0, 950.0, size=(20, 2))
        self.estimator = RANSACEstimator(max_iterations=1000, random_seed=42)

    def test_known_similarity_recovery(self):
        """Recover exact known similarity (s=1.03, theta=8.5 deg, t=[35.0, -42.0])."""
        s_true = 1.03
        rot_true = 8.5
        rad = math.radians(rot_true)
        r_mat = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
        t_true = np.array([35.0, -42.0])

        dst_pts = s_true * np.dot(self.src_pts, r_mat.T) + t_true

        model, result = self.estimator.estimate(
            SimilarityModel,
            self.src_pts,
            dst_pts,
            residual_threshold_m=1.0,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.num_inliers, 20)
        self.assertAlmostEqual(model.scale, s_true, places=4)
        self.assertAlmostEqual(model.rotation_deg, rot_true, places=3)
        np.testing.assert_allclose(model.translation, t_true, atol=1e-3)
        self.assertLess(result.rmse_m, 1e-4)

    def test_known_affine_recovery(self):
        """Recover exact known affine transformation with non-uniform scale and shear."""
        a_true = np.array([[1.04, 0.05], [-0.03, 0.97]])
        t_true = np.array([12.0, -18.0])

        dst_pts = np.dot(self.src_pts, a_true.T) + t_true

        model, result = self.estimator.estimate(
            AffineModel,
            self.src_pts,
            dst_pts,
            residual_threshold_m=1.0,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.num_inliers, 20)
        np.testing.assert_allclose(model.matrix_2x2, a_true, atol=1e-4)
        np.testing.assert_allclose(model.translation, t_true, atol=1e-3)
        self.assertLess(result.rmse_m, 1e-4)

    def test_gaussian_noise_robustness(self):
        """Gaussian jitter (sigma = 2.0m) added to coordinates; RMSE should be ~ 2.0m."""
        s_true = 1.0
        t_true = np.array([10.0, 20.0])
        noise = np.random.normal(0.0, 2.0, size=self.src_pts.shape)
        dst_noisy = self.src_pts + t_true + noise

        model, result = self.estimator.estimate(
            SimilarityModel,
            self.src_pts,
            dst_noisy,
            residual_threshold_m=6.0,  # 3 * sigma threshold
        )

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.num_inliers, 18)
        self.assertLess(result.rmse_m, 3.5)

    def test_outlier_rejection_capability(self):
        """Inject 40% gross outliers (8 out of 20 corrupted by 300m shifts)."""
        dst_pts = self.src_pts + np.array([20.0, -15.0])
        outlier_indices = [2, 5, 7, 11, 13, 16, 18, 19]  # 8 outliers
        dst_pts[outlier_indices] += np.random.uniform(200.0, 500.0, size=(len(outlier_indices), 2))

        model, result = self.estimator.estimate(
            SimilarityModel,
            self.src_pts,
            dst_pts,
            residual_threshold_m=20.0,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.num_inliers, 12)
        # Verify that none of the outlier indices are marked as inliers
        inlier_mask = result.inlier_mask
        for idx in outlier_indices:
            self.assertFalse(inlier_mask[idx], f"Outlier index {idx} was falsely accepted as inlier.")

    def test_spatial_coverage_clustered_vs_distributed(self):
        """Verify spatial coverage metrics distinguish clustered points from distributed points."""
        # 10 points tightly clustered in a 50m x 50m corner
        clustered = np.random.uniform(10.0, 60.0, size=(10, 2))
        cov_clustered = compute_spatial_coverage(clustered, scene_width_m=1000.0, scene_height_m=1000.0, grid_cells_per_axis=4)

        # 10 points spread across 1000m x 1000m
        distributed = np.random.uniform(50.0, 950.0, size=(10, 2))
        cov_distributed = compute_spatial_coverage(distributed, scene_width_m=1000.0, scene_height_m=1000.0, grid_cells_per_axis=4)

        self.assertLess(cov_clustered.occupied_cells, cov_distributed.occupied_cells)
        self.assertLess(cov_clustered.convex_hull_area_m2, cov_distributed.convex_hull_area_m2)
        self.assertLess(cov_clustered.bbox_area_m2, cov_distributed.bbox_area_m2)

    def test_poorly_conditioned_collinear_points(self):
        """Collinear points should fail affine estimation gracefully without crash."""
        collinear_src = np.array([[x, x] for x in range(10)], dtype=np.float64)
        collinear_dst = np.array([[x + 5, x - 2] for x in range(10)], dtype=np.float64)

        model, result = self.estimator.estimate(
            AffineModel,
            collinear_src,
            collinear_dst,
            residual_threshold_m=10.0,
        )
        self.assertFalse(result.success)

    def test_underdetermined_too_few_points(self):
        """Too few points (N=1 for similarity, N=2 for affine) returns failed estimation."""
        pts_1 = np.array([[100.0, 100.0]])
        model_s, res_s = self.estimator.estimate(SimilarityModel, pts_1, pts_1, residual_threshold_m=10.0)
        self.assertFalse(res_s.success)

        pts_2 = np.array([[100.0, 100.0], [200.0, 200.0]])
        model_a, res_a = self.estimator.estimate(AffineModel, pts_2, pts_2, residual_threshold_m=10.0)
        self.assertFalse(res_a.success)


if __name__ == "__main__":
    unittest.main()
