"""Unit tests for CNSF neighborhood construction (src/cnsf/neighborhood.py).

Chunk 8 Test Suite:
- K-nearest neighbor selection by physical Euclidean distance.
- Strict exclusion of the central crater itself.
- Edge case: candidate pool has fewer than K valid neighbors.
- Edge case: empty candidate pool.
- Search radius cutoff constraint.
- Cyclic interior angles sum to 360 degrees.
- Normalized distances average to 1.0 (scale invariance).
- Diameter ratios calculation.
"""

import math
from pathlib import Path
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cnsf.geometry import CraterFeature
from src.cnsf.neighborhood import CraterNeighborhood, build_crater_neighborhood


class TestCNSFNeighborhood(unittest.TestCase):
    """Test suite for crater neighborhood graph construction."""

    def setUp(self):
        # Create a central crater at (100, 100) with diameter 100m
        self.cc = CraterFeature(
            detection_id="CC_0",
            sensor="TEST",
            image_x=100.0,
            image_y=100.0,
            longitude=336.0,
            latitude=-2.0,
            diameter_px=20.0,
            diameter_m=100.0,
            radius_px=10.0,
            confidence=0.95,
        )

        # Create 6 neighbors at known distances and angles
        # NC1: (120, 100) -> dx=+20, dy=0 -> dist=20m, angle=0 deg (East)
        # NC2: (100, 130) -> dx=0, dy=+30 -> dist=30m, angle=90 deg (South in raster)
        # NC3: (60, 100)  -> dx=-40, dy=0 -> dist=40m, angle=180 deg (West)
        # NC4: (100, 50)  -> dx=0, dy=-50 -> dist=50m, angle=270 deg (North in raster)
        # NC5: (160, 100) -> dx=+60, dy=0 -> dist=60m, angle=0 deg
        # NC6: (100, 170) -> dx=0, dy=+70 -> dist=70m, angle=90 deg
        self.pool = [
            CraterFeature("NC_1", "TEST", 120.0, 100.0, 336.01, -2.0, 10.0, 50.0, 5.0, 0.8),
            CraterFeature("NC_2", "TEST", 100.0, 130.0, 336.0, -2.01, 12.0, 60.0, 6.0, 0.85),
            CraterFeature("NC_3", "TEST", 60.0, 100.0, 335.99, -2.0, 14.0, 70.0, 7.0, 0.75),
            CraterFeature("NC_4", "TEST", 100.0, 50.0, 336.0, -1.99, 16.0, 80.0, 8.0, 0.9),
            CraterFeature("NC_5", "TEST", 160.0, 100.0, 336.03, -2.0, 18.0, 90.0, 9.0, 0.7),
            CraterFeature("NC_6", "TEST", 100.0, 170.0, 336.0, -2.03, 20.0, 100.0, 10.0, 0.8),
        ]

    def test_k_nearest_selection(self):
        # Request K=4. Should select NC_1 (20m), NC_2 (30m), NC_3 (40m), NC_4 (50m)
        # NC_5 (60m) and NC_6 (70m) should be excluded
        nh = build_crater_neighborhood(self.cc, self.pool, k=4, gsd_m=1.0)

        self.assertEqual(nh.k_actual, 4)
        self.assertEqual(nh.k_requested, 4)
        selected_ids = {n.detection_id for n in nh.neighbors}
        self.assertEqual(selected_ids, {"NC_1", "NC_2", "NC_3", "NC_4"})

    def test_central_crater_exclusion(self):
        # Even if the candidate pool contains the central crater itself, it must be excluded
        pool_with_cc = self.pool + [self.cc]
        nh = build_crater_neighborhood(self.cc, pool_with_cc, k=4, gsd_m=1.0)

        for n in nh.neighbors:
            self.assertNotEqual(n.detection_id, self.cc.detection_id)
            self.assertFalse(math.isclose(n.image_x, self.cc.image_x) and math.isclose(n.image_y, self.cc.image_y))

    def test_interior_angles_sum_to_360(self):
        # For K=4 neighbors at 0, 90, 180, 270 degrees:
        nh = build_crater_neighborhood(self.cc, self.pool, k=4, gsd_m=1.0)

        self.assertEqual(len(nh.interior_angles_deg), 4)
        for ang in nh.interior_angles_deg:
            self.assertAlmostEqual(ang, 90.0, places=3)
        self.assertAlmostEqual(sum(nh.interior_angles_deg), 360.0, places=3)

    def test_normalized_distances(self):
        # Distances: 20, 30, 40, 50 -> mean = 35.0
        nh = build_crater_neighborhood(self.cc, self.pool, k=4, gsd_m=1.0)

        expected = [20.0 / 35.0, 30.0 / 35.0, 40.0 / 35.0, 50.0 / 35.0]
        np.testing.assert_allclose(sorted(nh.normalized_distances), expected, atol=1e-4)
        self.assertAlmostEqual(float(np.mean(nh.normalized_distances)), 1.0, places=5)

    def test_diameter_ratios(self):
        # Central diameter = 100m. Neighbor diameters = 50, 60, 70, 80m.
        nh = build_crater_neighborhood(self.cc, self.pool, k=4, gsd_m=1.0)

        expected = [0.5, 0.6, 0.7, 0.8]
        np.testing.assert_allclose(sorted(nh.diameter_ratios), expected, atol=1e-4)

    def test_max_search_radius_constraint(self):
        # If max_radius_m = 35.0m, only NC_1 (20m) and NC_2 (30m) should be selected
        nh = build_crater_neighborhood(self.cc, self.pool, k=5, max_radius_m=35.0, gsd_m=1.0)
        self.assertEqual(nh.k_actual, 2)
        selected_ids = {n.detection_id for n in nh.neighbors}
        self.assertEqual(selected_ids, {"NC_1", "NC_2"})

    def test_fewer_than_k_neighbors(self):
        # Request K=10 when pool only has 6
        nh = build_crater_neighborhood(self.cc, self.pool, k=10, gsd_m=1.0)
        self.assertEqual(nh.k_actual, 6)
        self.assertEqual(nh.k_requested, 10)

    def test_empty_candidate_pool(self):
        nh = build_crater_neighborhood(self.cc, [], k=5, gsd_m=1.0)
        self.assertEqual(nh.k_actual, 0)
        self.assertEqual(len(nh.neighbors), 0)


if __name__ == "__main__":
    unittest.main()
