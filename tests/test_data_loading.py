"""Unit and integration tests for memory-efficient data loading and metadata validation."""

from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import (
    BaseRasterReader,
    OHRCRasterReader,
    TMC2RasterReader,
    DatasetLoader,
    read_region,
    validate_file_size,
    parse_pds4_xml,
    check_metadata_discrepancies,
)
from src.utils.config import load_config


class DummyRasterReader(BaseRasterReader):
    """Subclass for testing BaseRasterReader functionality."""
    pass


class TestDataLoadingUnit(unittest.TestCase):
    """Unit tests using synthetic small arrays to test memory mapping and error logic."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create small 20x30 uint8 test image (600 bytes)
        self.uint8_path = self.temp_path / "test_u8.img"
        self.u8_data = np.arange(600, dtype=np.uint8).reshape((20, 30))
        self.u8_data.tofile(self.uint8_path)

        # Create small 15x25 uint16 little-endian test image (750 elements = 1500 bytes)
        self.uint16_path = self.temp_path / "test_u16.img"
        self.u16_data = np.arange(375, dtype="<u2").reshape((15, 25))
        self.u16_data.tofile(self.uint16_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_file_size_validation_success(self):
        size = validate_file_size(self.uint8_path, expected_lines=20, expected_samples=30, bytes_per_pixel=1)
        self.assertEqual(size, 600)

    def test_file_size_validation_failure(self):
        # Expect lines=21 -> size 630 != 600 -> must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            validate_file_size(self.uint8_path, expected_lines=21, expected_samples=30, bytes_per_pixel=1)
        self.assertIn("File size mismatch", str(ctx.exception))

    def test_read_region_synthetic(self):
        reader = DummyRasterReader(
            file_path=self.uint8_path,
            lines=20,
            samples=30,
            dtype=np.dtype(np.uint8),
            byte_order="none",
        )
        crop = reader.read_region(2, 5, 4, 10)
        expected = self.u8_data[2:5, 4:10]
        np.testing.assert_array_equal(crop, expected)
        self.assertEqual(crop.shape, (3, 6))
        self.assertEqual(crop.dtype, np.uint8)
        reader.close()

    def test_boundary_handling_errors(self):
        reader = DummyRasterReader(
            file_path=self.uint8_path,
            lines=20,
            samples=30,
            dtype=np.dtype(np.uint8),
            byte_order="none",
        )
        # Negative start
        with self.assertRaises(ValueError):
            reader.read_region(-1, 5, 0, 5)
        with self.assertRaises(ValueError):
            reader.read_region(0, 5, -1, 5)

        # Inverted range
        with self.assertRaises(ValueError):
            reader.read_region(5, 5, 0, 5)
        with self.assertRaises(ValueError):
            reader.read_region(6, 5, 0, 5)
        with self.assertRaises(ValueError):
            reader.read_region(0, 5, 10, 5)

        # Out of bounds
        with self.assertRaises(ValueError):
            reader.read_region(0, 21, 0, 5)
        with self.assertRaises(ValueError):
            reader.read_region(0, 5, 0, 31)

        reader.close()

    def test_deterministic_and_read_only(self):
        reader = DummyRasterReader(
            file_path=self.uint8_path,
            lines=20,
            samples=30,
            dtype=np.dtype(np.uint8),
            byte_order="none",
        )
        crop1 = reader.read_region(5, 10, 5, 10)
        crop2 = reader.read_region(5, 10, 5, 10)
        np.testing.assert_array_equal(crop1, crop2)

        # Mutating crop1 must not change crop2 or the file on disk
        crop1[0, 0] = 255
        self.assertNotEqual(crop1[0, 0], crop2[0, 0])

        crop_fresh = reader.read_region(5, 10, 5, 10)
        self.assertEqual(crop_fresh[0, 0], crop2[0, 0])
        reader.close()


class TestDataLoadingRealData(unittest.TestCase):
    """Integration tests on actual Chandrayaan-2 triplet data if available on disk."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = PROJECT_ROOT / "configs" / "default.yaml"
        cls.config = load_config(cls.config_path)
        cls.root_dir = cls.config.dataset.root_dir
        cls.has_real_data = cls.root_dir.is_dir()

    def setUp(self):
        if not self.has_real_data:
            self.skipTest(f"Real dataset directory not found at {self.root_dir}")

    def test_ohrc_file_size_and_xml(self):
        ohrc_cfg = self.config.dataset.ohrc
        img_path = ohrc_cfg.get_img_path(self.root_dir)
        xml_path = ohrc_cfg.get_xml_path(self.root_dir)

        self.assertTrue(img_path.is_file(), f"OHRC img not found: {img_path}")
        self.assertTrue(xml_path.is_file(), f"OHRC xml not found: {xml_path}")

        # Check exact byte size: 93693 * 12000 * 1 = 1,124,316,000 bytes
        expected_bytes = 93693 * 12000 * 1
        actual_bytes = img_path.stat().st_size
        self.assertEqual(actual_bytes, expected_bytes)

        # Parse XML and check metadata
        xml_meta = parse_pds4_xml(xml_path)
        self.assertEqual(xml_meta["lines"], 93693)
        self.assertEqual(xml_meta["samples"], 12000)
        self.assertEqual(xml_meta["dtype"], np.dtype(np.uint8))
        self.assertEqual(xml_meta["file_size"], expected_bytes)
        self.assertAlmostEqual(xml_meta["pixel_resolution_m"], 0.25)

        discrepancies = check_metadata_discrepancies(
            {"lines": ohrc_cfg.lines, "samples": ohrc_cfg.samples, "pixel_resolution_m": ohrc_cfg.pixel_resolution_m},
            xml_meta,
        )
        self.assertEqual(len(discrepancies), 0)

    def test_tmc2_file_size_and_xml(self):
        tmc2_cfg = self.config.dataset.tmc2
        img_path = tmc2_cfg.get_img_path(self.root_dir)
        xml_path = tmc2_cfg.get_xml_path(self.root_dir)

        self.assertTrue(img_path.is_file(), f"TMC-2 img not found: {img_path}")
        self.assertTrue(xml_path.is_file(), f"TMC-2 xml not found: {xml_path}")

        # Check exact byte size: 295234 * 4000 * 2 = 2,361,872,000 bytes
        expected_bytes = 295234 * 4000 * 2
        actual_bytes = img_path.stat().st_size
        self.assertEqual(actual_bytes, expected_bytes)

        # Parse XML and check metadata
        xml_meta = parse_pds4_xml(xml_path)
        self.assertEqual(xml_meta["lines"], 295234)
        self.assertEqual(xml_meta["samples"], 4000)
        self.assertEqual(xml_meta["dtype"], np.dtype("<u2"))
        self.assertEqual(xml_meta["file_size"], expected_bytes)
        self.assertAlmostEqual(xml_meta["pixel_resolution_m"], 5.40)

        discrepancies = check_metadata_discrepancies(
            {"lines": tmc2_cfg.lines, "samples": tmc2_cfg.samples, "pixel_resolution_m": tmc2_cfg.pixel_resolution_m},
            xml_meta,
        )
        self.assertEqual(len(discrepancies), 0)

    def test_ohrc_crop_100x100(self):
        loader = DatasetLoader(config=self.config)
        crop1 = loader.read_region("OHRC", 10000, 10100, 5000, 5100)

        self.assertEqual(crop1.shape, (100, 100))
        self.assertEqual(crop1.dtype, np.uint8)
        self.assertTrue(np.all(np.isfinite(crop1)))

        # Deterministic repeated read
        crop2 = loader.read_region("OHRC", 10000, 10100, 5000, 5100)
        np.testing.assert_array_equal(crop1, crop2)

        # Non-empty surface data check
        self.assertGreater(float(np.max(crop1)), float(np.min(crop1)))

        loader.close()

    def test_tmc2_crop_100x100(self):
        loader = DatasetLoader(config=self.config)
        crop1 = loader.read_region("TMC2", 50000, 50100, 1500, 1600)

        self.assertEqual(crop1.shape, (100, 100))
        self.assertEqual(crop1.dtype, np.dtype("<u2"))
        self.assertTrue(np.all(np.isfinite(crop1)))

        # Deterministic repeated read
        crop2 = loader.read_region("TMC2", 50000, 50100, 1500, 1600)
        np.testing.assert_array_equal(crop1, crop2)

        # Non-empty surface data check
        self.assertGreater(float(np.max(crop1)), float(np.min(crop1)))

        loader.close()

    def test_real_data_boundary_errors(self):
        loader = DatasetLoader(config=self.config)

        # OHRC line out of bounds
        with self.assertRaises(ValueError):
            loader.read_region("OHRC", 93600, 93700, 0, 100)

        # TMC2 sample out of bounds
        with self.assertRaises(ValueError):
            loader.read_region("TMC2", 0, 100, 3950, 4050)

        loader.close()


if __name__ == "__main__":
    unittest.main()
