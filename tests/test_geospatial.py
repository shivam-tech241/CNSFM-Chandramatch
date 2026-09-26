"""Unit and integration tests for geographic metadata, footprint construction, and overlap analysis."""

from pathlib import Path
import sys
import unittest
import numpy as np
from shapely.geometry import Polygon

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.geospatial import (
    CornerCoordinates,
    GeographicMetadata,
    calculate_spherical_polygon_area_m2,
    compute_footprint_overlap,
    parse_geographic_metadata,
    GroundGridReader,
    MOON_RADIUS_METERS,
)
from src.utils.config import load_config


class TestGeospatialUnit(unittest.TestCase):
    """Unit tests for synthetic geometry and spherical calculations."""

    def test_corner_to_polygon_validity(self):
        corners = CornerCoordinates(
            upper_left_lat=10.0,
            upper_left_lon=20.0,
            upper_right_lat=10.0,
            upper_right_lon=25.0,
            lower_left_lat=5.0,
            lower_left_lon=20.0,
            lower_right_lat=5.0,
            lower_right_lon=25.0,
        )
        poly = corners.to_polygon()
        self.assertTrue(poly.is_valid)
        self.assertEqual(len(poly.exterior.coords), 5)
        self.assertAlmostEqual(poly.area, 25.0)

    def test_spherical_polygon_area(self):
        # 1 deg x 1 deg patch near equator: [0, 1] lon x [0, 1] lat
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        area_m2 = calculate_spherical_polygon_area_m2(poly)
        # Analytical area = R^2 * dlon_rad * (sin(lat2_rad) - sin(lat1_rad))
        expected_m2 = (MOON_RADIUS_METERS**2) * np.radians(1.0) * np.sin(np.radians(1.0))
        self.assertAlmostEqual(area_m2, expected_m2, delta=1.0)

        # Empty polygon area should be 0.0
        self.assertEqual(calculate_spherical_polygon_area_m2(Polygon()), 0.0)

    def test_overlap_calculation_synthetic(self):
        # Large container: [0, 10] x [0, 10]
        c_large = CornerCoordinates(10, 0, 10, 10, 0, 0, 0, 10)
        # Small contained: [2, 4] x [2, 4]
        c_small = CornerCoordinates(4, 2, 4, 4, 2, 2, 2, 4)
        # Disjoint: [20, 30] x [20, 30]
        c_disjoint = CornerCoordinates(30, 20, 30, 30, 20, 20, 20, 30)

        m_large = GeographicMetadata(
            "Large", Path("dummy.xml"), c_large, None, c_large, "Selenographic", 1.0, 0, 0, 0, 100, 100,
            c_large.to_polygon().area, calculate_spherical_polygon_area_m2(c_large.to_polygon()) / 1e6
        )
        m_small = GeographicMetadata(
            "Small", Path("dummy.xml"), c_small, None, c_small, "Selenographic", 1.0, 0, 0, 0, 100, 100,
            c_small.to_polygon().area, calculate_spherical_polygon_area_m2(c_small.to_polygon()) / 1e6
        )
        m_disjoint = GeographicMetadata(
            "Disjoint", Path("dummy.xml"), c_disjoint, None, c_disjoint, "Selenographic", 1.0, 0, 0, 0, 100, 100,
            c_disjoint.to_polygon().area, calculate_spherical_polygon_area_m2(c_disjoint.to_polygon()) / 1e6
        )

        # Contained test
        metrics_cont, inter_cont = compute_footprint_overlap(m_small, m_large)
        self.assertTrue(metrics_cont.is_overlapping)
        self.assertTrue(metrics_cont.sensor1_contained_in_sensor2)
        self.assertAlmostEqual(metrics_cont.overlap_ratio_sensor1, 1.0, places=4)
        self.assertFalse(inter_cont.is_empty)

        # Disjoint test
        metrics_disj, inter_disj = compute_footprint_overlap(m_small, m_disjoint)
        self.assertFalse(metrics_disj.is_overlapping)
        self.assertEqual(metrics_disj.intersection_area_km2, 0.0)
        self.assertEqual(metrics_disj.overlap_ratio_sensor1, 0.0)
        self.assertTrue(inter_disj.is_empty)

    def test_missing_xml_raises(self):
        with self.assertRaises(FileNotFoundError):
            parse_geographic_metadata("non_existent_metadata.xml")


class TestGeospatialRealData(unittest.TestCase):
    """Integration tests on actual Chandrayaan-2 XML and geometry files."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = PROJECT_ROOT / "configs" / "default.yaml"
        cls.config = load_config(cls.config_path)
        cls.root_dir = cls.config.dataset.root_dir
        cls.has_real_data = cls.root_dir.is_dir()

    def setUp(self):
        if not self.has_real_data:
            self.skipTest(f"Dataset root not found: {self.root_dir}")

    def test_ohrc_geographic_metadata(self):
        ohrc_xml = self.config.dataset.ohrc.get_xml_path(self.root_dir)
        meta = parse_geographic_metadata(ohrc_xml, sensor_name="OHRC")

        self.assertEqual(meta.sensor_name, "OHRC")
        self.assertEqual(meta.projection, "Selenographic")
        self.assertAlmostEqual(meta.pixel_resolution_m, 0.25)
        self.assertAlmostEqual(meta.sun_azimuth_deg, 269.646098, places=4)
        self.assertAlmostEqual(meta.solar_incidence_deg, 75.936949, places=4)

        # Verify corner latitudes/longitudes
        self.assertAlmostEqual(meta.active_corners.upper_left_lat, -2.576048, places=5)
        self.assertAlmostEqual(meta.active_corners.lower_left_lat, -3.413866, places=5)
        self.assertAlmostEqual(meta.active_corners.upper_left_lon, 336.486234, places=5)
        self.assertAlmostEqual(meta.active_corners.upper_right_lon, 336.589455, places=5)

        # Footprint area check (~79.38 km^2)
        self.assertGreater(meta.footprint_area_km2, 70.0)
        self.assertLess(meta.footprint_area_km2, 90.0)

    def test_tmc2_geographic_metadata(self):
        tmc2_xml = self.config.dataset.tmc2.get_xml_path(self.root_dir)
        meta = parse_geographic_metadata(tmc2_xml, sensor_name="TMC-2")

        self.assertEqual(meta.sensor_name, "TMC-2")
        self.assertEqual(meta.projection, "Selenographic")
        self.assertAlmostEqual(meta.pixel_resolution_m, 5.40)
        self.assertAlmostEqual(meta.sun_azimuth_deg, 108.866212, places=4)
        self.assertAlmostEqual(meta.solar_incidence_deg, 41.246015, places=4)

        # Verify refined corners
        self.assertIsNotNone(meta.refined_corners)
        self.assertAlmostEqual(meta.active_corners.upper_left_lat, 43.095893, places=5)
        self.assertAlmostEqual(meta.active_corners.lower_left_lat, -4.962184, places=5)

        # Area check (~40,200 km^2)
        self.assertGreater(meta.footprint_area_km2, 38000.0)
        self.assertLess(meta.footprint_area_km2, 43000.0)

    def test_ohrc_inside_tmc2_overlap(self):
        ohrc_xml = self.config.dataset.ohrc.get_xml_path(self.root_dir)
        tmc2_xml = self.config.dataset.tmc2.get_xml_path(self.root_dir)

        ohrc_meta = parse_geographic_metadata(ohrc_xml, sensor_name="OHRC")
        tmc2_meta = parse_geographic_metadata(tmc2_xml, sensor_name="TMC-2")

        metrics, inter_poly = compute_footprint_overlap(ohrc_meta, tmc2_meta)

        self.assertTrue(metrics.is_overlapping)
        self.assertTrue(metrics.sensor1_contained_in_sensor2)
        self.assertAlmostEqual(metrics.overlap_ratio_sensor1, 1.0, places=4)
        self.assertGreater(metrics.intersection_area_km2, 70.0)
        self.assertTrue(inter_poly.is_valid)

    def test_geometry_grid_readers(self):
        ohrc_grd_csv = self.root_dir / "OHRC" / "geometry" / "calibrated" / "20210405" / "ch2_ohr_ncp_20210405T1606536730_g_grd_d18.csv"
        tmc2_grd_csv = self.root_dir / "TMC" / "geometry" / "calibrated" / "20250807" / "ch2_tmc_ncf_20250807T1904346039_g_grd_d18.csv"

        self.assertTrue(ohrc_grd_csv.is_file())
        self.assertTrue(tmc2_grd_csv.is_file())

        ohrc_reader = GroundGridReader(ohrc_grd_csv, "OHRC")
        ohrc_summary = ohrc_reader.get_summary()
        self.assertEqual(ohrc_summary["record_count"], 113498)

        tmc2_reader = GroundGridReader(tmc2_grd_csv, "TMC-2")
        tmc2_summary = tmc2_reader.get_summary()
        self.assertEqual(tmc2_summary["record_count"], 121114)

        # Test finding bounding box in TMC-2 for OHRC bounds
        bbox = tmc2_reader.find_pixel_bounding_box(
            min_lon=336.48, max_lon=336.59, min_lat=-3.42, max_lat=-2.57
        )
        self.assertIsNotNone(bbox)
        self.assertGreaterEqual(bbox["scan_start"], 280000)
        self.assertLessEqual(bbox["scan_end"], 287000)
        self.assertGreaterEqual(bbox["pixel_start"], 2000)
        self.assertLessEqual(bbox["pixel_end"], 3600)


if __name__ == "__main__":
    unittest.main()
