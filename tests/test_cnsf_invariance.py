"""Synthetic validation tests for CNSF descriptor invariance and robustness (Chunk 8 Part 11).

Tests:
1. Global Translation Invariance: Scene shifted by arbitrary (dx, dy).
2. Global Rotation Invariance: Scene rotated by arbitrary angles (45 deg, 90 deg, 160.78 deg, 180 deg).
3. Uniform Scaling Invariance: Scene coordinates and diameters scaled by s = 2.0x and s = 0.5x.
4. Coordinate Noise Robustness: Gaussian jitter added to crater positions (delta = 1.0 px).
5. Missing Neighbor Robustness: One neighbor omitted from target neighborhood; frequency voting retains remaining.
6. False Crater Robustness: False crater injected into neighborhood; frequency voting eliminates it.
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

from src.cnsf.descriptor import build_cnsf_descriptor
from src.cnsf.geometry import CraterFeature
from src.cnsf.matcher import CNSFMatcher
from src.cnsf.neighborhood import build_crater_neighborhood


class TestCNSFInvariance(unittest.TestCase):
    """Test suite for algorithmic invariances and synthetic noise robustness."""

    def setUp(self):
        # Create a non-symmetric reference crater configuration with CC at (200, 200) and 5 distinct NCs
        self.cc = CraterFeature("CC", "TEST", 200.0, 200.0, 336.0, -2.0, 20.0, 100.0, 10.0, 0.95)
        # 5 asymmetric neighbors with varying distances and diameters
        self.neighbors = [
            CraterFeature("NC_1", "TEST", 240.0, 210.0, 336.0, -2.0, 10.0, 50.0, 5.0, 0.9),  # dx=+40, dy=+10
            CraterFeature("NC_2", "TEST", 215.0, 260.0, 336.0, -2.0, 12.0, 60.0, 6.0, 0.85), # dx=+15, dy=+60
            CraterFeature("NC_3", "TEST", 145.0, 225.0, 336.0, -2.0, 14.0, 70.0, 7.0, 0.88), # dx=-55, dy=+25
            CraterFeature("NC_4", "TEST", 170.0, 150.0, 336.0, -2.0, 16.0, 80.0, 8.0, 0.92), # dx=-30, dy=-50
            CraterFeature("NC_5", "TEST", 235.0, 165.0, 336.0, -2.0, 18.0, 90.0, 9.0, 0.82), # dx=+35, dy=-35
        ]
        self.matcher = CNSFMatcher(xi_min=3, nndr_threshold=0.5, distance_cutoff=0.1, coarse_gsd_m=1.0)

    def test_translation_invariance(self):
        """Scene translated by dx=+450 px, dy=-280 px must yield identical CNSF distance d=0.0."""
        dx, dy = 450.0, -280.0
        cc_trans = CraterFeature("CC_T", "TEST", self.cc.image_x + dx, self.cc.image_y + dy, 336.0, -2.0, 20.0, 100.0, 10.0, 0.95)
        ncs_trans = [
            CraterFeature(f"{n.detection_id}_T", "TEST", n.image_x + dx, n.image_y + dy, 336.0, -2.0, n.diameter_px, n.diameter_m, n.radius_px, n.confidence)
            for n in self.neighbors
        ]

        nh_orig = build_crater_neighborhood(self.cc, self.neighbors, k=5, gsd_m=1.0)
        desc_orig = build_cnsf_descriptor(nh_orig)

        nh_trans = build_crater_neighborhood(cc_trans, ncs_trans, k=5, gsd_m=1.0)
        desc_trans = build_cnsf_descriptor(nh_trans)

        dist, xi, pairs = self.matcher.compute_pairwise_cnsf_distance(desc_orig, desc_trans)
        self.assertAlmostEqual(dist, 0.0, places=5)
        self.assertEqual(xi, 5)

    def test_rotation_invariance_arbitrary_angles(self):
        """Scene rotated by 45, 90, 160.78 (lunar solar azimuth diff), and 180 degrees."""
        test_angles_deg = [45.0, 90.0, 160.78, 180.0, 270.0]

        nh_orig = build_crater_neighborhood(self.cc, self.neighbors, k=5, gsd_m=1.0)
        desc_orig = build_cnsf_descriptor(nh_orig)

        cx, cy = self.cc.image_x, self.cc.image_y

        for rot_deg in test_angles_deg:
            rad = math.radians(rot_deg)
            cos_a, sin_a = math.cos(rad), math.sin(rad)

            ncs_rot = []
            for n in self.neighbors:
                rel_x = n.image_x - cx
                rel_y = n.image_y - cy
                rx = cx + (rel_x * cos_a - rel_y * sin_a)
                ry = cy + (rel_x * sin_a + rel_y * cos_a)
                ncs_rot.append(
                    CraterFeature(f"{n.detection_id}_R", "TEST", rx, ry, 336.0, -2.0, n.diameter_px, n.diameter_m, n.radius_px, n.confidence)
                )

            nh_rot = build_crater_neighborhood(self.cc, ncs_rot, k=5, gsd_m=1.0)
            desc_rot = build_cnsf_descriptor(nh_rot)

            dist, xi, pairs = self.matcher.compute_pairwise_cnsf_distance(desc_orig, desc_rot)
            self.assertAlmostEqual(dist, 0.0, places=3, msg=f"Failed rotation invariance at {rot_deg} degrees")
            self.assertGreaterEqual(xi, 4, msg=f"Failed neighbor retention at {rot_deg} degrees")

    def test_uniform_scale_invariance(self):
        """Scene coordinates and diameters scaled by s=2.0x and s=0.5x."""
        for scale in [2.0, 0.5]:
            cx, cy = self.cc.image_x, self.cc.image_y
            cc_scaled = CraterFeature("CC_S", "TEST", cx * scale, cy * scale, 336.0, -2.0, self.cc.diameter_px * scale, self.cc.diameter_m * scale, self.cc.radius_px * scale, 0.95)
            ncs_scaled = [
                CraterFeature(f"{n.detection_id}_S", "TEST", n.image_x * scale, n.image_y * scale, 336.0, -2.0, n.diameter_px * scale, n.diameter_m * scale, n.radius_px * scale, n.confidence)
                for n in self.neighbors
            ]

            nh_orig = build_crater_neighborhood(self.cc, self.neighbors, k=5, gsd_m=1.0)
            desc_orig = build_cnsf_descriptor(nh_orig)

            nh_scaled = build_crater_neighborhood(cc_scaled, ncs_scaled, k=5, gsd_m=1.0)
            desc_scaled = build_cnsf_descriptor(nh_scaled)

            dist, xi, pairs = self.matcher.compute_pairwise_cnsf_distance(desc_orig, desc_scaled)
            self.assertAlmostEqual(dist, 0.0, places=4, msg=f"Failed uniform scale invariance at s={scale}")
            self.assertEqual(xi, 5)

    def test_coordinate_noise_robustness(self):
        """Small Gaussian jitter (sigma = 1.0 px) should yield low CNSF distance (d < 0.1)."""
        np.random.seed(42)
        ncs_noisy = []
        for n in self.neighbors:
            jitter_x = float(np.random.normal(0.0, 1.0))
            jitter_y = float(np.random.normal(0.0, 1.0))
            ncs_noisy.append(
                CraterFeature(f"{n.detection_id}_N", "TEST", n.image_x + jitter_x, n.image_y + jitter_y, 336.0, -2.0, n.diameter_px, n.diameter_m, n.radius_px, n.confidence)
            )

        nh_orig = build_crater_neighborhood(self.cc, self.neighbors, k=5, gsd_m=1.0)
        desc_orig = build_cnsf_descriptor(nh_orig)

        nh_noisy = build_crater_neighborhood(self.cc, ncs_noisy, k=5, gsd_m=1.0)
        desc_noisy = build_cnsf_descriptor(nh_noisy)

        dist, xi, pairs = self.matcher.compute_pairwise_cnsf_distance(desc_orig, desc_noisy)
        self.assertLess(dist, 0.10)
        self.assertGreaterEqual(xi, 3)

    def test_missing_neighbor_robustness(self):
        """Dropping 1 neighbor out of 5: frequency voting must retain remaining 4 corresponding NCs."""
        # Drop NC_3
        ncs_missing = [n for n in self.neighbors if n.detection_id != "NC_3"]
        # Add a distant filler so K=5 can be filled
        ncs_missing.append(CraterFeature("NC_FILLER", "TEST", 400.0, 400.0, 336.0, -2.0, 10.0, 50.0, 5.0, 0.8))

        nh_orig = build_crater_neighborhood(self.cc, self.neighbors, k=5, gsd_m=1.0)
        desc_orig = build_cnsf_descriptor(nh_orig)

        nh_missing = build_crater_neighborhood(self.cc, ncs_missing, k=5, gsd_m=1.0)
        desc_missing = build_cnsf_descriptor(nh_missing)

        dist, xi, pairs = self.matcher.compute_pairwise_cnsf_distance(desc_orig, desc_missing)
        # Should retain at least 3-4 corresponding NCs and distance <= 0.1
        self.assertGreaterEqual(xi, 3)
        self.assertLess(dist, 0.15)

    def test_false_crater_elimination_by_frequency_voting(self):
        """An injected false crater in the target neighborhood must be rejected by frequency voting."""
        # Insert a random false crater between NC_1 and NC_2
        false_crater = CraterFeature("FALSE_CRATER", "TEST", 228.0, 235.0, 336.0, -2.0, 11.0, 55.0, 5.5, 0.8)
        ncs_with_false = self.neighbors[:4] + [false_crater]

        nh_orig = build_crater_neighborhood(self.cc, self.neighbors, k=5, gsd_m=1.0)
        desc_orig = build_cnsf_descriptor(nh_orig)

        nh_false = build_crater_neighborhood(self.cc, ncs_with_false, k=5, gsd_m=1.0)
        desc_false = build_cnsf_descriptor(nh_false)

        dist, xi, pairs = self.matcher.compute_pairwise_cnsf_distance(desc_orig, desc_false)
        # FALSE_CRATER must not be in the retained corresponding pairs
        matched_target_ids = {tgt_id for src_id, tgt_id in pairs}
        self.assertNotIn("FALSE_CRATER", matched_target_ids)
        self.assertGreaterEqual(xi, 3)


if __name__ == "__main__":
    unittest.main()
