"""Unit tests for geometric transformation models (Chunk 9)."""

import math
from pathlib import Path
import sys
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometric_validation.models import SimilarityModel, AffineModel


class TestGeometricModels(unittest.TestCase):
    """Test suite for SimilarityModel and AffineModel."""

    def setUp(self):
        # 4 non-collinear reference points in meters
        self.src_pts = np.array([
            [100.0, 150.0],
            [350.0, 180.0],
            [220.0, 420.0],
            [480.0, 390.0],
        ], dtype=np.float64)

    def test_similarity_fit_exact_transform(self):
        """Known similarity transform (s=1.08, theta=15 deg, t=[45.0, -30.0]) must be recovered exactly."""
        s_true = 1.08
        rot_deg_true = 15.0
        rad = math.radians(rot_deg_true)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        r_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        t_true = np.array([45.0, -30.0])

        dst_pts = s_true * np.dot(self.src_pts, r_mat.T) + t_true

        model = SimilarityModel()
        ok = model.fit(self.src_pts, dst_pts)
        self.assertTrue(ok)
        self.assertAlmostEqual(model.scale, s_true, places=5)
        self.assertAlmostEqual(model.rotation_deg, rot_deg_true, places=4)
        np.testing.assert_allclose(model.translation, t_true, atol=1e-4)

        # Residuals must be effectively zero
        res = model.compute_residuals(self.src_pts, dst_pts)
        np.testing.assert_allclose(res, 0.0, atol=1e-5)

        is_plaus, reason = model.is_plausible()
        self.assertTrue(is_plaus, reason)

    def test_affine_fit_exact_transform(self):
        """Known affine transform with shear must be recovered exactly."""
        a_true = np.array([[1.05, 0.08], [-0.04, 0.98]])
        t_true = np.array([-25.0, 60.0])
        dst_pts = np.dot(self.src_pts, a_true.T) + t_true

        model = AffineModel()
        ok = model.fit(self.src_pts, dst_pts)
        self.assertTrue(ok)
        np.testing.assert_allclose(model.matrix_2x2, a_true, atol=1e-5)
        np.testing.assert_allclose(model.translation, t_true, atol=1e-5)

        res = model.compute_residuals(self.src_pts, dst_pts)
        np.testing.assert_allclose(res, 0.0, atol=1e-5)

        is_plaus, reason = model.is_plausible()
        self.assertTrue(is_plaus, reason)

    def test_minimum_sample_requirements(self):
        """Similarity requires N>=2, Affine requires N>=3."""
        sim = SimilarityModel()
        self.assertEqual(sim.min_samples(), 2)
        self.assertFalse(sim.fit(self.src_pts[:1], self.src_pts[:1]))
        self.assertTrue(sim.fit(self.src_pts[:2], self.src_pts[:2]))

        aff = AffineModel()
        self.assertEqual(aff.min_samples(), 3)
        self.assertFalse(aff.fit(self.src_pts[:2], self.src_pts[:2]))
        self.assertTrue(aff.fit(self.src_pts[:3], self.src_pts[:3]))

    def test_collinear_points_rejected_by_affine(self):
        """Collinear points (rank < 3) must be rejected by AffineModel."""
        collinear_src = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        collinear_dst = np.array([[15.0, 12.0], [25.0, 22.0], [35.0, 32.0]])
        aff = AffineModel()
        ok = aff.fit(collinear_src, collinear_dst)
        self.assertFalse(ok)

    def test_implausible_model_rejection(self):
        """Models with unphysical scale or non-positive determinant are flagged as implausible."""
        # Scale = 10x
        dst_huge = 10.0 * self.src_pts
        sim = SimilarityModel()
        sim.fit(self.src_pts, dst_huge)
        is_plaus, reason = sim.is_plausible(min_scale=0.7, max_scale=1.3)
        self.assertFalse(is_plaus)
        self.assertIn("Scale", reason)

        # Reflection matrix (negative determinant)
        a_reflect = np.array([[-1.0, 0.0], [0.0, 1.0]])
        dst_reflect = np.dot(self.src_pts, a_reflect.T)
        aff = AffineModel()
        aff.fit(self.src_pts, dst_reflect)
        is_plaus, reason = aff.is_plausible()
        self.assertFalse(is_plaus)
        self.assertIn("determinant", reason)


if __name__ == "__main__":
    unittest.main()
