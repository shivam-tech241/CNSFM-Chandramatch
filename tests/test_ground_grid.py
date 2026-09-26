"""Unit and integration tests for ISRO ground-grid parsing, interpolation, and overlap extraction.

Chunk 4 Test Suite.
"""

from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.ground_grid import (
    GridSummary,
    GroundGrid,
    OverlapResult,
    ValidationMetrics,
    compute_sensor_overlap,
    geo_to_pixel,
    pixel_to_geo,
)
from src.utils.config import load_config


class TestGroundGridSynthetic(unittest.TestCase):
    """Unit tests on synthetic ground grids for fast isolated validation."""

    def setUp(self):
        # Create a small synthetic regular grid: 5 scans x 4 pixels
        # Scan: 0, 100, 200, 300, 400
        # Pixel: 0, 100, 200, 300
        scans = [0, 100, 200, 300, 400]
        pixels = [0, 100, 200, 300]

        rows = []
        for s in scans:
            for p in pixels:
                # Selenographic lat decreases with scan, lon increases with pixel
                lon = 336.0 + (p / 1000.0)
                lat = -2.0 - (s / 500.0)
                rows.append({"Longitude": lon, "Latitude": lat, "Pixel": p, "Scan": s})

        self.df_synthetic = pd.DataFrame(rows)
        self.tmp_csv = PROJECT_ROOT / "tests" / "synthetic_grid.csv"
        self.df_synthetic.to_csv(self.tmp_csv, index=False)
        self.grid = GroundGrid(self.tmp_csv, sensor_name="SYNTHETIC")

    def tearDown(self):
        if self.tmp_csv.exists():
            self.tmp_csv.unlink()

    def test_grid_parsing_and_structure(self):
        summary = self.grid.get_summary()
        self.assertEqual(summary.num_points, 20)
        self.assertEqual(summary.n_scans, 5)
        self.assertEqual(summary.n_pixels, 4)
        self.assertTrue(summary.is_regular_grid)
        self.assertTrue(summary.is_monotonic_lat)
        self.assertTrue(summary.is_monotonic_lon)
        self.assertEqual(summary.scan_min, 0)
        self.assertEqual(summary.scan_max, 400)
        self.assertEqual(summary.pixel_min, 0)
        self.assertEqual(summary.pixel_max, 300)

    def test_exact_tie_point_interpolation(self):
        # At exact grid vertices, pixel_to_geo must return the exact vertex coordinates
        for idx, row in self.df_synthetic.iterrows():
            lon, lat = self.grid.pixel_to_geo(row["Pixel"], row["Scan"])
            self.assertAlmostEqual(lon, row["Longitude"], places=9)
            self.assertAlmostEqual(lat, row["Latitude"], places=9)

            px, sc = self.grid.geo_to_pixel(row["Longitude"], row["Latitude"])
            self.assertAlmostEqual(px, row["Pixel"], delta=1e-5)
            self.assertAlmostEqual(sc, row["Scan"], delta=1e-5)

    def test_interior_continuous_interpolation(self):
        # Midpoint of cell [0..100] x [0..100]
        # pixel=50, scan=50
        lon, lat = self.grid.pixel_to_geo(50.0, 50.0)
        self.assertAlmostEqual(lon, 336.05, places=7)
        self.assertAlmostEqual(lat, -2.10, places=7)

        # Invert back
        rec_px, rec_sc = self.grid.geo_to_pixel(lon, lat)
        self.assertAlmostEqual(rec_px, 50.0, delta=1e-6)
        self.assertAlmostEqual(rec_sc, 50.0, delta=1e-6)

    def test_vectorized_and_scalar_equivalence(self):
        pixels = np.array([25.0, 75.0, 150.0, 220.0])
        scans = np.array([50.0, 120.0, 250.0, 310.0])

        lons_v, lats_v = self.grid.pixel_to_geo(pixels, scans)
        rec_px_v, rec_sc_v = self.grid.geo_to_pixel(lons_v, lats_v)

        for k in range(len(pixels)):
            lon_s, lat_s = self.grid.pixel_to_geo(pixels[k], scans[k])
            self.assertAlmostEqual(lons_v[k], lon_s, places=10)
            self.assertAlmostEqual(lats_v[k], lat_s, places=10)

            px_s, sc_s = self.grid.geo_to_pixel(lon_s, lat_s)
            self.assertAlmostEqual(rec_px_v[k], px_s, places=6)
            self.assertAlmostEqual(rec_sc_v[k], sc_s, places=6)

    def test_out_of_bounds_handling(self):
        # Out-of-bounds pixel
        with self.assertRaises(ValueError):
            self.grid.pixel_to_geo(-10.0, 50.0, check_bounds=True)

        with self.assertRaises(ValueError):
            self.grid.pixel_to_geo(350.0, 50.0, check_bounds=True)

        # Out-of-bounds scan
        with self.assertRaises(ValueError):
            self.grid.pixel_to_geo(50.0, -10.0, check_bounds=True)

        with self.assertRaises(ValueError):
            self.grid.pixel_to_geo(50.0, 450.0, check_bounds=True)

        # Out-of-bounds geo
        with self.assertRaises(ValueError):
            self.grid.geo_to_pixel(335.0, -2.5, check_bounds=True)

        with self.assertRaises(ValueError):
            self.grid.geo_to_pixel(336.1, -1.0, check_bounds=True)

        # With check_bounds=False, should return NaN without raising
        nan_lon, nan_lat = self.grid.pixel_to_geo(-10.0, 50.0, check_bounds=False)
        self.assertTrue(np.isnan(nan_lon))
        self.assertTrue(np.isnan(nan_lat))

        nan_px, nan_sc = self.grid.geo_to_pixel(330.0, 0.0, check_bounds=False)
        self.assertTrue(np.isnan(nan_px))
        self.assertTrue(np.isnan(nan_sc))

    def test_deterministic_results(self):
        # Successive calls must produce exact bit-for-bit identical results
        lon1, lat1 = self.grid.pixel_to_geo(123.456, 234.567)
        lon2, lat2 = self.grid.pixel_to_geo(123.456, 234.567)
        self.assertEqual(lon1, lon2)
        self.assertEqual(lat1, lat2)

        px1, sc1 = self.grid.geo_to_pixel(lon1, lat1)
        px2, sc2 = self.grid.geo_to_pixel(lon1, lat1)
        self.assertEqual(px1, px2)
        self.assertEqual(sc1, sc2)


class TestGroundGridRealData(unittest.TestCase):
    """Integration tests on actual ISRO OHRC and TMC-2 ground-coordinate grids."""

    @classmethod
    def setUpClass(cls):
        config_path = PROJECT_ROOT / "configs" / "default.yaml"
        cls.config = load_config(config_path)
        cls.root_dir = cls.config.dataset.root_dir

        cls.ohrc_grd_csv = (
            cls.root_dir / "OHRC" / "geometry" / "calibrated" / "20210405" / "ch2_ohr_ncp_20210405T1606536730_g_grd_d18.csv"
        )
        cls.tmc2_grd_csv = (
            cls.root_dir / "TMC" / "geometry" / "calibrated" / "20250807" / "ch2_tmc_ncf_20250807T1904346039_g_grd_d18.csv"
        )

        cls.skip_real = not (cls.ohrc_grd_csv.is_file() and cls.tmc2_grd_csv.is_file())
        if not cls.skip_real:
            cls.ohrc_grid = GroundGrid(cls.ohrc_grd_csv, sensor_name="OHRC")
            cls.tmc2_grid = GroundGrid(cls.tmc2_grd_csv, sensor_name="TMC-2")

    def setUp(self):
        if self.skip_real:
            self.skipTest("Real dataset ground-grid CSV files not found on disk.")

    def test_ohrc_grid_metadata(self):
        s = self.ohrc_grid.get_summary()
        self.assertEqual(s.sensor_name, "OHRC")
        self.assertEqual(s.num_points, 113498)
        self.assertEqual(s.n_scans, 938)
        self.assertEqual(s.n_pixels, 121)
        self.assertEqual(s.scan_step_regular, 100)
        self.assertEqual(s.pixel_step_regular, 100)
        self.assertTrue(s.is_regular_grid)
        self.assertTrue(s.is_monotonic_lat)
        self.assertTrue(s.is_monotonic_lon)
        self.assertAlmostEqual(s.lon_min, 336.484646, places=5)
        self.assertAlmostEqual(s.lon_max, 336.589455, places=5)
        self.assertAlmostEqual(s.lat_min, -3.41690433, places=5)
        self.assertAlmostEqual(s.lat_max, -2.57604793, places=5)

    def test_tmc2_grid_metadata(self):
        s = self.tmc2_grid.get_summary()
        self.assertEqual(s.sensor_name, "TMC-2")
        self.assertEqual(s.num_points, 121114)
        self.assertEqual(s.n_scans, 2954)
        self.assertEqual(s.n_pixels, 41)
        self.assertEqual(s.scan_step_regular, 100)
        self.assertEqual(s.pixel_step_regular, 100)
        self.assertTrue(s.is_regular_grid)
        self.assertTrue(s.is_monotonic_lat)
        self.assertTrue(s.is_monotonic_lon)
        self.assertAlmostEqual(s.lon_min, 336.238746, places=5)
        self.assertAlmostEqual(s.lat_min, -4.96218448, places=5)

    def test_held_out_validation_accuracy(self):
        # Genuine held-out validation with coarse training grid (stride 2)
        # Center points are held out and predicted. Error must be sub-pixel!
        tmc_metrics = self.tmc2_grid.validate_held_out(stride=2, max_scans=100)
        self.assertLess(tmc_metrics.pixel_error_mean, 1.0)
        self.assertLess(tmc_metrics.scan_error_mean, 1.0)
        self.assertLess(tmc_metrics.geo_dist_m_mean, 5.40)  # Mean ground error < 1 TMC-2 pixel (5.4 m)

        ohrc_metrics = self.ohrc_grid.validate_held_out(stride=2, max_scans=50)
        self.assertLess(ohrc_metrics.pixel_error_mean, 0.5)
        self.assertLess(ohrc_metrics.scan_error_mean, 0.5)
        self.assertLess(ohrc_metrics.geo_dist_m_mean, 0.25)  # Mean ground error < 1 OHRC pixel (0.25 m)

    def test_round_trip_numerical_precision(self):
        # Mathematical reversibility on continuous interior points
        ohrc_rt = self.ohrc_grid.validate_round_trip(n_samples=100)
        self.assertLess(ohrc_rt.pixel_error_max, 1e-4)
        self.assertLess(ohrc_rt.scan_error_max, 1e-3)

        tmc_rt = self.tmc2_grid.validate_round_trip(n_samples=100)
        self.assertLess(tmc_rt.pixel_error_max, 1e-4)
        self.assertLess(tmc_rt.scan_error_max, 1e-4)

    def test_overlap_calculation(self):
        overlap_res, b_data = compute_sensor_overlap(self.ohrc_grid, self.tmc2_grid, num_boundary_samples_per_edge=20)
        self.assertIsInstance(overlap_res, OverlapResult)

        # Verify scan bounds cover around 280700 to 286000
        self.assertGreaterEqual(overlap_res.tmc2_overlap_scan_start, 280700)
        self.assertLessEqual(overlap_res.tmc2_overlap_scan_start, 280800)
        self.assertGreaterEqual(overlap_res.tmc2_overlap_scan_end, 285850)
        self.assertLessEqual(overlap_res.tmc2_overlap_scan_end, 285950)

        # Verify pixel bounds cover around 2500 to 3200
        self.assertGreaterEqual(overlap_res.tmc2_overlap_pixel_start, 2500)
        self.assertLessEqual(overlap_res.tmc2_overlap_pixel_start, 2530)
        self.assertGreaterEqual(overlap_res.tmc2_overlap_pixel_end, 3150)
        self.assertLessEqual(overlap_res.tmc2_overlap_pixel_end, 3180)

        # Enclosed tie points should be ~306-385
        self.assertGreater(overlap_res.num_tie_points_in_overlap, 250)

    def test_functional_convenience_api(self):
        # Test geo_to_pixel and pixel_to_geo top-level functions
        lon_test = 336.52
        lat_test = -2.85
        px, sc = geo_to_pixel(lon_test, lat_test, sensor=self.tmc2_grid)
        self.assertAlmostEqual(px, 2949.77, delta=1.0)
        self.assertAlmostEqual(sc, 282447.74, delta=2.0)

        rec_lon, rec_lat = pixel_to_geo(px, sc, sensor=self.tmc2_grid)
        self.assertAlmostEqual(rec_lon, lon_test, places=6)
        self.assertAlmostEqual(rec_lat, lat_test, places=6)


if __name__ == "__main__":
    unittest.main()
