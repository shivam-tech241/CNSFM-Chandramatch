"""Integration tests for geometric validation pipeline (Chunk 9)."""

from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometric_validation.models import SimilarityModel, AffineModel
from src.geometric_validation.residuals import compute_candidate_displacements, lonlat_to_local_planar_m
from src.geometric_validation.diagnostics import (
    evaluate_threshold_sensitivity,
    analyze_residual_correlations,
    analyze_candidate_consistency,
)
from src.geometric_validation.coverage import compute_spatial_coverage


class TestGeometricValidationPipeline(unittest.TestCase):
    """Integration test suite for Chunk 9 pipeline components."""

    def setUp(self):
        np.random.seed(123)
        # Create a mock dataframe mimicking Chunk 8 candidate output
        center_lon, center_lat = 336.536, -3.000
        n_pairs = 15

        src_lons = center_lon + np.random.uniform(-0.015, 0.015, size=n_pairs)
        src_lats = center_lat + np.random.uniform(-0.015, 0.015, size=n_pairs)

        # Target coordinates with a ~15m translation + small noise
        tgt_lons = src_lons + 0.0003 + np.random.normal(0, 0.00005, size=n_pairs)
        tgt_lats = src_lats - 0.0002 + np.random.normal(0, 0.00005, size=n_pairs)

        self.df_candidates = pd.DataFrame({
            "source_cc_id": [f"TMC_{i}" for i in range(n_pairs)],
            "target_cc_id": [f"OHRC_{i}" for i in range(n_pairs)],
            "source_x": np.random.uniform(20.0, 180.0, size=n_pairs),
            "source_y": np.random.uniform(20.0, 180.0, size=n_pairs),
            "target_x": np.random.uniform(20.0, 180.0, size=n_pairs),
            "target_y": np.random.uniform(20.0, 180.0, size=n_pairs),
            "source_diameter_m": np.random.uniform(60.0, 200.0, size=n_pairs),
            "target_diameter_m": np.random.uniform(60.0, 200.0, size=n_pairs),
            "diameter_ratio": np.random.uniform(0.6, 0.95, size=n_pairs),
            "cnsf_distance": np.random.uniform(0.01, 0.08, size=n_pairs),
            "source_lon": src_lons,
            "source_lat": src_lats,
            "target_lon": tgt_lons,
            "target_lat": tgt_lats,
        })

    def test_end_to_end_displacement_and_sensitivity(self):
        """Displacements, local metric coordinates, and threshold sensitivity sweep."""
        df_disp = compute_candidate_displacements(self.df_candidates)
        self.assertIn("displacement_m", df_disp.columns)
        self.assertIn("src_X_m", df_disp.columns)

        src_pts = df_disp[["src_X_m", "src_Y_m"]].values
        tgt_pts = df_disp[["tgt_X_m", "tgt_Y_m"]].values

        df_sens = evaluate_threshold_sensitivity(
            SimilarityModel,
            src_pts,
            tgt_pts,
            thresholds_m=[10.0, 25.0, 50.0],
        )

        self.assertEqual(len(df_sens), 3)
        self.assertTrue((df_sens["num_inliers"] >= 0).all())

    def test_correlations_and_consistency(self):
        """Correlation analysis and candidate consistency categorization."""
        df = self.df_candidates.copy()
        df["residual_m"] = np.random.uniform(5.0, 45.0, size=len(df))

        corr_res = analyze_residual_correlations(df)
        self.assertIn("cnsf_distance", corr_res)
        self.assertIn("source_diameter_m", corr_res)

        cons_res = analyze_candidate_consistency(df)
        self.assertEqual(cons_res["total_evaluated"], len(df))
        self.assertIn("ideal_matches_low_cnsf_low_res", cons_res)


if __name__ == "__main__":
    unittest.main()
