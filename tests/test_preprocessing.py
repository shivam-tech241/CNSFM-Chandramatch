"""Unit and integration tests for Preprocessing and Image Characterization (Chunk 5)."""

import json
from pathlib import Path
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import DatasetLoader
from src.preprocessing import (
    ImageStatistics,
    OverlapDataExtractor,
    apply_clahe,
    compute_dark_region_diagnostics,
    compute_gradients,
    compute_image_statistics,
    compute_scale_diagnostic,
    compute_streaming_image_statistics,
    percentile_normalize,
)


class TestPreprocessingUnit(unittest.TestCase):
    """Unit tests using synthetic arrays for fast isolated component validation."""

    def setUp(self):
        # Create a synthetic 100x100 uint8 test image with known gradient and contrast
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        # Synthetic surface with gradient and dark crater-like dip
        arr = 30 + 0.5 * xx + 0.2 * yy
        # Dip in center
        dist = np.sqrt((xx - 50) ** 2 + (yy - 50) ** 2)
        arr[dist < 20] -= 25.0
        self.synthetic_raw = np.clip(np.round(arr), 0, 255).astype(np.uint8)

    def test_raw_data_remains_unchanged(self):
        original_copy = self.synthetic_raw.copy()

        # Run normalization, CLAHE, gradients, and dark regions
        norm_img, _ = percentile_normalize(self.synthetic_raw)
        clahe_img, _ = apply_clahe(norm_img)
        grads = compute_gradients(norm_img)
        mask, _ = compute_dark_region_diagnostics(self.synthetic_raw)

        # Raw input array must remain bit-for-bit identical
        np.testing.assert_array_equal(self.synthetic_raw, original_copy)

    def test_normalization_deterministic(self):
        norm1, params1 = percentile_normalize(self.synthetic_raw, p_min=1.0, p_max=99.0)
        norm2, params2 = percentile_normalize(self.synthetic_raw, p_min=1.0, p_max=99.0)

        np.testing.assert_array_equal(norm1, norm2)
        self.assertEqual(params1, params2)

    def test_normalization_output_range(self):
        norm_img, params = percentile_normalize(self.synthetic_raw, p_min=5.0, p_max=95.0)

        self.assertEqual(norm_img.dtype, np.uint8)
        self.assertGreaterEqual(int(norm_img.min()), 0)
        self.assertLessEqual(int(norm_img.max()), 255)
        self.assertIn("vmin_intensity", params)
        self.assertIn("vmax_intensity", params)

        # Edge case: constant array does not crash
        constant_arr = np.full((50, 50), 120, dtype=np.uint8)
        norm_const, params_const = percentile_normalize(constant_arr)
        self.assertEqual(norm_const.shape, (50, 50))
        self.assertEqual(norm_const.dtype, np.uint8)

    def test_clahe_output_validity(self):
        norm_img, _ = percentile_normalize(self.synthetic_raw)
        enhanced, params = apply_clahe(norm_img, clip_limit=2.0, tile_grid_size=(8, 8))

        self.assertEqual(enhanced.dtype, np.uint8)
        self.assertEqual(enhanced.shape, norm_img.shape)
        self.assertEqual(params["clip_limit"], 2.0)
        self.assertEqual(params["tile_grid_size"], [8, 8])

        # Non-uint8 input should raise ValueError
        with self.assertRaises(ValueError):
            apply_clahe(self.synthetic_raw.astype(np.float32))

    def test_gradient_output_validity(self):
        norm_img, _ = percentile_normalize(self.synthetic_raw)
        grads = compute_gradients(norm_img, ksize=3)

        self.assertEqual(grads["sobel_x"].shape, norm_img.shape)
        self.assertEqual(grads["sobel_y"].shape, norm_img.shape)
        self.assertEqual(grads["magnitude"].shape, norm_img.shape)

        # Magnitude must be non-negative
        self.assertTrue(np.all(grads["magnitude"] >= 0.0))

        # Visual outputs must be uint8
        self.assertEqual(grads["sobel_x_vis"].dtype, np.uint8)
        self.assertEqual(grads["sobel_y_vis"].dtype, np.uint8)
        self.assertEqual(grads["magnitude_vis"].dtype, np.uint8)

        stats = grads["statistics"]
        self.assertIn("mean_gradient_magnitude", stats)
        self.assertIn("median_gradient_magnitude", stats)
        self.assertIn("fraction_near_zero_gradient", stats)
        self.assertGreaterEqual(stats["fraction_near_zero_gradient"], 0.0)
        self.assertLessEqual(stats["fraction_near_zero_gradient"], 1.0)

    def test_dark_region_mask_validity(self):
        mask, stats = compute_dark_region_diagnostics(self.synthetic_raw, percentile_threshold=5.0)

        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, self.synthetic_raw.shape)

        # Mask should contain only 0 and 255
        unique_vals = set(np.unique(mask))
        self.assertTrue(unique_vals.issubset({0, 255}))

        # Dark fraction should be around 5% (allow +/- 1.5% due to integer discretization)
        self.assertAlmostEqual(stats["dark_fraction"], 0.05, delta=0.02)
        self.assertGreater(stats["connected_component_count"], 0)

    def test_statistics_reproducibility(self):
        stats1 = compute_image_statistics(self.synthetic_raw, sensor_name="Test")
        stats2 = compute_image_statistics(self.synthetic_raw, sensor_name="Test")

        self.assertEqual(stats1.min, stats2.min)
        self.assertEqual(stats1.max, stats2.max)
        self.assertEqual(stats1.mean, stats2.mean)
        self.assertEqual(stats1.median, stats2.median)
        self.assertEqual(stats1.std, stats2.std)
        self.assertEqual(stats1.p1, stats2.p1)
        self.assertEqual(stats1.p5, stats2.p5)
        self.assertEqual(stats1.p50, stats2.p50)
        self.assertEqual(stats1.p95, stats2.p95)
        self.assertEqual(stats1.p99, stats2.p99)

    def test_scale_diagnostic_dimensions(self):
        # High-res native patch: 432x432
        # Target patch: 20x20
        high_res = np.random.randint(0, 255, (432, 432), dtype=np.uint8)
        low_res = np.random.randint(0, 255, (20, 20), dtype=np.uint16)

        downsampled, metrics = compute_scale_diagnostic(high_res, low_res, ohrc_gsd_m=0.25, tmc2_gsd_m=5.40)

        self.assertEqual(downsampled.shape, (20, 20))
        self.assertEqual(downsampled.dtype, np.uint8)
        self.assertEqual(metrics["resolution_scale_ratio"], 21.6)
        self.assertEqual(metrics["ohrc_downsampled_shape"], [20, 20])


class TestPreprocessingIntegration(unittest.TestCase):
    """Integration tests on actual Triplet-1 data and Chunk 4 overlap artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.loader = DatasetLoader()
        cls.extractor = OverlapDataExtractor(loader=cls.loader)

    def test_chunk4_overlap_bounds_consistency(self):
        tmc_crop, meta = self.extractor.extract_tmc2_overlap()

        # Check bounds match Chunk 4 validated overlap bounds
        self.assertEqual(meta.row_start, 280781)
        self.assertEqual(meta.row_end, 285893)
        self.assertEqual(meta.col_start, 2519)
        self.assertEqual(meta.col_end, 3163)

        self.assertEqual(tmc_crop.shape, (5112, 644))
        self.assertEqual(tmc_crop.dtype, np.uint16)

        # Must be strictly read-only
        self.assertFalse(tmc_crop.flags.writeable)
        with self.assertRaises(ValueError):
            tmc_crop[0, 0] = 999

    def test_benchmark_pair_extraction(self):
        tmc_bench, ohrc_bench, meta = self.extractor.extract_benchmark_pair(
            center_lon=336.536, center_lat=-3.000, tmc_size_px=200
        )

        self.assertEqual(tmc_bench.shape, (200, 200))
        self.assertEqual(ohrc_bench.shape, (4320, 4320))
        self.assertEqual(tmc_bench.dtype, np.uint16)
        self.assertEqual(ohrc_bench.dtype, np.uint8)

        # Both must be read-only
        self.assertFalse(tmc_bench.flags.writeable)
        self.assertFalse(ohrc_bench.flags.writeable)

    def test_preprocessing_report_generated(self):
        report_path = PROJECT_ROOT / "results" / "preprocessing" / "preprocessing_report.json"
        self.assertTrue(report_path.is_file(), f"Missing report: {report_path}")

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("raw_statistics", data)
        self.assertIn("normalization_parameters", data)
        self.assertIn("clahe_parameters", data)
        self.assertIn("gradient_statistics", data)
        self.assertIn("dark_region_diagnostics", data)
        self.assertIn("resolution_scale_diagnostic", data)
        self.assertIn("artifact_paths", data)

        # Check file paths exist on disk
        for name, path_str in data["artifact_paths"].items():
            path = Path(path_str)
            self.assertTrue(path.is_file(), f"Artifact file does not exist: {path}")


if __name__ == "__main__":
    unittest.main()
