"""Unit tests for CNSF descriptor construction (src/cnsf/descriptor.py).

Chunk 8 Test Suite:
- Number of angular structures matches K(K-1)/2.
- Deterministic generation: identical inputs produce identical descriptors.
- Correct angular structure parameterization (beta, S1, S2, phi0, phi1, phi2).
- Feature vector D dimensionality and values.
- Edge case: K < 2.
"""

from pathlib import Path
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cnsf.descriptor import CNSFDescriptor, build_cnsf_descriptor
from src.cnsf.geometry import CraterFeature
from src.cnsf.neighborhood import build_crater_neighborhood


class TestCNSFDescriptor(unittest.TestCase):
    """Test suite for CNSF descriptor construction."""

    def setUp(self):
        self.cc = CraterFeature("CC", "TEST", 100.0, 100.0, 336.0, -2.0, 20.0, 100.0, 10.0, 0.9)
        self.pool = [
            CraterFeature(f"NC_{i}", "TEST", 100.0 + 30.0 * np.cos(i * np.pi / 5), 100.0 + 30.0 * np.sin(i * np.pi / 5), 336.0, -2.0, 10.0, 50.0 + 5.0 * i, 5.0, 0.8)
            for i in range(10)
        ]

    def test_number_of_angular_structures(self):
        # For K=4: 4*3/2 = 6
        nh4 = build_crater_neighborhood(self.cc, self.pool, k=4, gsd_m=1.0)
        desc4 = build_cnsf_descriptor(nh4)
        self.assertEqual(len(desc4.angular_structures), 6)
        self.assertEqual(len(desc4.feature_vector_d), 6)

        # For K=5: 5*4/2 = 10
        nh5 = build_crater_neighborhood(self.cc, self.pool, k=5, gsd_m=1.0)
        desc5 = build_cnsf_descriptor(nh5)
        self.assertEqual(len(desc5.angular_structures), 10)
        self.assertEqual(len(desc5.feature_vector_d), 10)

        # For K=10: 10*9/2 = 45
        nh10 = build_crater_neighborhood(self.cc, self.pool, k=10, gsd_m=1.0)
        desc10 = build_cnsf_descriptor(nh10)
        self.assertEqual(len(desc10.angular_structures), 45)
        self.assertEqual(len(desc10.feature_vector_d), 45)

    def test_deterministic_generation(self):
        nh1 = build_crater_neighborhood(self.cc, self.pool, k=5, gsd_m=1.0)
        desc1 = build_cnsf_descriptor(nh1)

        nh2 = build_crater_neighborhood(self.cc, self.pool, k=5, gsd_m=1.0)
        desc2 = build_cnsf_descriptor(nh2)

        self.assertEqual(desc1.k, desc2.k)
        self.assertEqual(len(desc1.angular_structures), len(desc2.angular_structures))
        np.testing.assert_array_equal(desc1.feature_vector_d, desc2.feature_vector_d)
        np.testing.assert_array_equal(desc1.normalized_distances, desc2.normalized_distances)
        np.testing.assert_array_equal(desc1.cyclic_interior_angles, desc2.cyclic_interior_angles)

    def test_angular_structure_fields(self):
        nh = build_crater_neighborhood(self.cc, self.pool, k=3, gsd_m=1.0)
        desc = build_cnsf_descriptor(nh)

        for s in desc.angular_structures:
            self.assertEqual(s.cc_id, "CC")
            self.assertGreaterEqual(s.angle_deg, 0.0)
            self.assertLess(s.angle_deg, 360.0)
            self.assertGreater(s.dist1_m, 0.0)
            self.assertGreater(s.dist2_m, 0.0)
            self.assertEqual(s.diam0_m, 100.0)
            self.assertAlmostEqual(s.dist_ratio, s.dist2_m / s.dist1_m, places=5)
            self.assertAlmostEqual(s.diam_ratio1, s.diam1_m / 100.0, places=5)
            self.assertAlmostEqual(s.diam_ratio2, s.diam2_m / 100.0, places=5)

    def test_edge_case_k_less_than_two(self):
        nh = build_crater_neighborhood(self.cc, self.pool[:1], k=1, gsd_m=1.0)
        desc = build_cnsf_descriptor(nh)
        self.assertEqual(desc.k, 1)
        self.assertEqual(len(desc.angular_structures), 0)
        self.assertEqual(len(desc.feature_vector_d), 0)


if __name__ == "__main__":
    unittest.main()
