"""Unit tests for CNSFMatcher (src/cnsf/matcher.py).

Chunk 8 Test Suite:
- Equation 7 & 8 angular structure comparison under error bounds.
- Frequency voting mode calculation and NC filtering.
- xi_min cutoff enforcement.
- NNDR ratio and absolute distance constraint filtering.
- Matcher integration on synthetic feature sets.
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

from src.cnsf.descriptor import AngularStructure, build_cnsf_descriptor
from src.cnsf.geometry import CraterFeature
from src.cnsf.matcher import CNSFMatcher
from src.cnsf.neighborhood import build_crater_neighborhood


class TestCNSFMatcher(unittest.TestCase):
    """Test suite for CNSF matching logic and filtering stages."""

    def setUp(self):
        self.matcher = CNSFMatcher(
            delta_px=3.0,
            eta_percent=0.25,
            xi_min=3,
            nndr_threshold=0.2,
            distance_cutoff=0.1,
            coarse_gsd_m=5.40,
        )

        # Synthetic angular structures
        self.sa = AngularStructure(
            cc_id="CC1", nc1_id="A1", nc2_id="A2",
            angle_deg=60.0, dist1_m=100.0, dist2_m=150.0,
            diam0_m=80.0, diam1_m=40.0, diam2_m=60.0,
            dist_ratio=1.5, diam_ratio1=0.5, diam_ratio2=0.75,
        )

    def test_identical_angular_structures_similar(self):
        is_sim, case = self.matcher.check_angular_structure_similarity(self.sa, self.sa)
        self.assertTrue(is_sim)
        self.assertEqual(case, 1)

    def test_inverted_angular_structure_similar(self):
        # Inverted structure: angle = 360 - 60 = 300 deg, ratio = 1 / 1.5 = 0.6667
        # diam_ratio1 = 0.75, diam_ratio2 = 0.5
        sa_inv = AngularStructure(
            cc_id="CC2", nc1_id="B1", nc2_id="B2",
            angle_deg=300.0, dist1_m=150.0, dist2_m=100.0,
            diam0_m=80.0, diam1_m=60.0, diam2_m=40.0,
            dist_ratio=100.0 / 150.0, diam_ratio1=60.0 / 80.0, diam_ratio2=40.0 / 80.0,
        )
        is_sim, case = self.matcher.check_angular_structure_similarity(self.sa, sa_inv)
        self.assertTrue(is_sim)
        self.assertEqual(case, 2)

    def test_dissimilar_angular_structures_rejected(self):
        # Dissimilar: angle = 120 deg (outside tolerance of 60 deg)
        sa_diff = AngularStructure(
            cc_id="CC3", nc1_id="C1", nc2_id="C2",
            angle_deg=120.0, dist1_m=100.0, dist2_m=150.0,
            diam0_m=80.0, diam1_m=40.0, diam2_m=60.0,
            dist_ratio=1.5, diam_ratio1=0.5, diam_ratio2=0.75,
        )
        is_sim, case = self.matcher.check_angular_structure_similarity(self.sa, sa_diff)
        self.assertFalse(is_sim)
        self.assertEqual(case, 0)

    def test_xi_min_rejection(self):
        # Create two neighborhoods that only share 2 NCs (less than xi_min=3)
        cc_a = CraterFeature("CCA", "TMC", 100.0, 100.0, 336.0, -2.0, 20.0, 100.0, 10.0, 0.9)
        ncs_a = [
            CraterFeature(f"NCA_{i}", "TMC", 100.0 + 30.0 * np.cos(i), 100.0 + 30.0 * np.sin(i), 336.0, -2.0, 10.0, 50.0, 5.0, 0.8)
            for i in range(5)
        ]
        # B only shares 2 corresponding NCs, rest are completely unrelated
        cc_b = CraterFeature("CCB", "OHRC", 100.0, 100.0, 336.0, -2.0, 20.0, 100.0, 10.0, 0.9)
        ncs_b = [
            CraterFeature("NCA_0", "OHRC", 100.0 + 30.0 * np.cos(0), 100.0 + 30.0 * np.sin(0), 336.0, -2.0, 10.0, 50.0, 5.0, 0.8),
            CraterFeature("NCA_1", "OHRC", 100.0 + 30.0 * np.cos(1), 100.0 + 30.0 * np.sin(1), 336.0, -2.0, 10.0, 50.0, 5.0, 0.8),
            CraterFeature("RAND_1", "OHRC", 500.0, 200.0, 336.0, -2.0, 10.0, 20.0, 5.0, 0.8),
            CraterFeature("RAND_2", "OHRC", 200.0, 500.0, 336.0, -2.0, 10.0, 20.0, 5.0, 0.8),
            CraterFeature("RAND_3", "OHRC", 400.0, 400.0, 336.0, -2.0, 10.0, 20.0, 5.0, 0.8),
        ]

        desc_a = build_cnsf_descriptor(build_crater_neighborhood(cc_a, ncs_a, k=5, gsd_m=1.0))
        desc_b = build_cnsf_descriptor(build_crater_neighborhood(cc_b, ncs_b, k=5, gsd_m=1.0))

        dist, xi, pairs = self.matcher.compute_pairwise_cnsf_distance(desc_a, desc_b)
        self.assertTrue(math.isinf(dist))
        self.assertLess(xi, 3)

    def test_nndr_filtering(self):
        # Matcher should retain matches with nndr_ratio <= nndr_threshold
        cc_source = CraterFeature("S1", "TMC", 100.0, 100.0, 336.0, -2.0, 20.0, 100.0, 10.0, 0.9)
        angles = [0.0, 72.0, 144.0, 216.0, 288.0]
        dists = [40.0, 50.0, 60.0, 70.0, 80.0]
        ncs_source = [
            CraterFeature(f"NC_{i}", "TMC", 100.0 + dists[i] * math.cos(math.radians(angles[i])), 100.0 + dists[i] * math.sin(math.radians(angles[i])), 336.0, -2.0, 10.0, 50.0 + 5.0 * i, 5.0, 0.8)
            for i in range(5)
        ]
        desc_s = build_cnsf_descriptor(build_crater_neighborhood(cc_source, ncs_source, k=5, gsd_m=1.0))

        # Target 1: exact match (d ~ 0)
        desc_t1 = build_cnsf_descriptor(build_crater_neighborhood(
            CraterFeature("T1", "OHRC", 100.0, 100.0, 336.0, -2.0, 20.0, 100.0, 10.0, 0.9),
            ncs_source, k=5, gsd_m=1.0
        ))

        # Target 2: different configuration
        ncs_other = [
            CraterFeature(f"NC_OTHER_{i}", "OHRC", 100.0 + 10.0 * i, 100.0 - 40.0 * i, 336.0, -2.0, 10.0, 30.0, 5.0, 0.8)
            for i in range(5)
        ]
        desc_t2 = build_cnsf_descriptor(build_crater_neighborhood(
            CraterFeature("T2", "OHRC", 300.0, 300.0, 336.0, -2.0, 20.0, 100.0, 10.0, 0.9),
            ncs_other, k=5, gsd_m=1.0
        ))

        matches = self.matcher.match_descriptors([desc_s], [desc_t1, desc_t2])
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].source_cc_id, "S1")
        self.assertEqual(matches[0].target_cc_id, "T1")
        self.assertAlmostEqual(matches[0].cnsf_distance, 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
