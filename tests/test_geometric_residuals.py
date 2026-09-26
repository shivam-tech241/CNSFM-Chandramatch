"""Unit tests for residual computation and metric coordinate conversions (Chunk 9)."""

import math
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometric_validation.residuals import (
    compute_candidate_displacements,
    compute_residual_statistics,
    lonlat_to_local_planar_m,
    MOON_RADIUS_M,
)


class TestGeometricResiduals(unittest.TestCase):
    """Test suite for coordinate conversions, displacement, and residual metrics."""

    def test_lonlat_to_local_planar_scaling(self):
        """1 degree of latitude on Moon sphere must equal pi/180 * R_moon ~ 30323.35 meters."""
        center_lon, center_lat = 336.536, -3.000
        lons = np.array([center_lon, center_lon])
        lats = np.array([center_lat, center_lat + 1.0])

        x_m, y_m = lonlat_to_local_planar_m(lons, lats, center_lon, center_lat)

        expected_dy = (math.pi / 180.0) * MOON_RADIUS_M
        self.assertAlmostEqual(x_m[1], 0.0, places=3)
        self.assertAlmostEqual(y_m[1], expected_dy, places=2)

    def test_candidate_displacements_calculation(self):
        """Displacement between two coordinates separated by 100m East and 50m North."""
        center_lon, center_lat = 336.536, -3.000
        # Delta in degrees for ~100m dx and ~50m dy
        d_lat_deg = (50.0 / MOON_RADIUS_M) * (180.0 / math.pi)
        d_lon_deg = (100.0 / (MOON_RADIUS_M * math.cos(math.radians(center_lat)))) * (180.0 / math.pi)

        df = pd.DataFrame([{
            "source_cc_id": "TMC_1",
            "target_cc_id": "OHRC_1",
            "source_lon": center_lon,
            "source_lat": center_lat,
            "target_lon": center_lon + d_lon_deg,
            "target_lat": center_lat + d_lat_deg,
        }])

        df_out = compute_candidate_displacements(df, center_lon, center_lat)
        expected_disp = math.hypot(100.0, 50.0)
        self.assertAlmostEqual(df_out.iloc[0]["displacement_m"], expected_disp, delta=0.5)

    def test_residual_statistics_metrics(self):
        """Verify mean, median, RMSE, P95, and max calculations."""
        res = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        stats = compute_residual_statistics(res)

        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["min_m"], 10.0)
        self.assertEqual(stats["max_m"], 50.0)
        self.assertEqual(stats["mean_m"], 30.0)
        self.assertEqual(stats["median_m"], 30.0)
        # RMSE = sqrt(mean([100, 400, 900, 1600, 2500])) = sqrt(5500/5) = sqrt(1100) ~ 33.166
        self.assertAlmostEqual(stats["rmse_m"], math.sqrt(1100.0), places=2)

    def test_empty_residuals_returns_zeros(self):
        """Empty array returns safe zeros."""
        stats = compute_residual_statistics(np.zeros(0))
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["rmse_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
