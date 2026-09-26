"""Unit tests for configuration loading and validation."""

from pathlib import Path
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, AppConfig


class TestConfig(unittest.TestCase):
    def test_load_default_config(self):
        config_path = PROJECT_ROOT / "configs" / "default.yaml"
        self.assertTrue(config_path.is_file(), f"Config file not found at {config_path}")

        config = load_config(config_path)
        self.assertIsInstance(config, AppConfig)

        # Check OHRC metadata matches specification
        self.assertEqual(config.dataset.ohrc.lines, 93693)
        self.assertEqual(config.dataset.ohrc.samples, 12000)
        self.assertAlmostEqual(config.dataset.ohrc.pixel_resolution_m, 0.25)
        self.assertEqual(config.dataset.ohrc.dtype, "uint8")

        # Check TMC-2 metadata matches specification
        self.assertEqual(config.dataset.tmc2.lines, 295234)
        self.assertEqual(config.dataset.tmc2.samples, 4000)
        self.assertAlmostEqual(config.dataset.tmc2.pixel_resolution_m, 5.40)
        self.assertEqual(config.dataset.tmc2.dtype, "uint16")
        self.assertEqual(config.dataset.tmc2.endian, "little")

    def test_missing_config_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_config("non_existent_config.yaml")


if __name__ == "__main__":
    unittest.main()
