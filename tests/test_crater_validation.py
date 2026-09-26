"""Unit tests for Chunk 7: Cross-Scale Crater Anchor Validation and YOLOv9-C Detector.

Tests numerical correctness, invariants, edge cases, and transformations for:
1. Detector metadata and paper-vs-adapted labelling.
2. GSD conversion and physical crater diameter calculations.
3. Bidirectional projection (TMC-2 -> Geo -> OHRC) and expected OHRC diameters.
4. Scale representation bookkeeping and area downsampling invariants.
5. Candidate crater pairing and spatial distance bounding.
6. Local crater neighborhood construction (K=5, 10, 15, 20).
7. CNSF invariant feature calculations:
   - Normalized distances (Eq. 6)
   - Azimuthal order and interior angles summing to 360 deg (Eq. 7)
   - Diameter ratios (Eq. 8)
8. Edge cases: empty detections, single crater, missing detections, image boundaries.
9. Cyclic structural similarity matching under arbitrary planar rotations.
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

from src.crater_detection import (
    CandidateCraterPair,
    CraterDetection,
    CraterNeighborhood,
    ScaleRepresentationMeta,
    TMC2AnchorCandidate,
    YOLOv9CraterDetector,
    compute_cnsf_structural_similarity,
    construct_crater_neighborhood,
    create_multiscale_representations,
    extract_and_project_tmc2_anchors,
    pair_candidate_craters,
)
from src.io.ground_grid import GroundGrid


class TestDetectorMetadataAndLabelling(unittest.TestCase):
    """Test detector classification, metadata correctness, and non-fabrication rules."""

    def test_paper_detector_naming_and_metadata(self):
        detector = YOLOv9CraterDetector(
            weights_path="data/models/YOLOv9Best.pt",
            name="paper_yolov9c",
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )
        self.assertEqual(detector.name, "paper_yolov9c")
        self.assertEqual(detector.confidence_threshold, 0.25)
        self.assertEqual(detector.iou_threshold, 0.45)
        self.assertEqual(detector.imgsz, 640)
        self.assertEqual(detector.min_radius_px, 3.0)
        self.assertEqual(detector.max_radius_px, 320.0)

    def test_adapted_detector_naming_enforcement(self):
        # When custom or fine-tuned weights are used, name must reflect adapted status
        detector = YOLOv9CraterDetector(
            weights_path="data/models/YOLOv9Best.pt",
            name="adapted_yolov9",
            confidence_threshold=0.30,
        )
        self.assertEqual(detector.name, "adapted_yolov9")
        self.assertNotEqual(detector.name, "paper_yolov9c")


class TestPhysicalScaleAndGSDConversions(unittest.TestCase):
    """Test numerical accuracy of GSD, pixel-to-meter, and meter-to-pixel conversions."""

    def test_gsd_physical_diameter_conversion(self):
        # TMC-2: 5.40 m/px. A 100m crater has diameter 100 / 5.40 = 18.5185 pixels
        diam_m = 100.0
        tmc_gsd = 5.40
        ohr_gsd = 0.25

        tmc_px = diam_m / tmc_gsd
        ohr_px = diam_m / ohr_gsd

        self.assertAlmostEqual(tmc_px, 18.5185185, places=4)
        self.assertAlmostEqual(ohr_px, 400.0, places=4)
        self.assertAlmostEqual(ohr_px / tmc_px, 21.6, places=4)

    def test_multiscale_representations_geometry(self):
        # Create a synthetic 4320 x 4320 image
        synth_img = np.zeros((4320, 4320), dtype=np.uint8)
        # Put a synthetic bright crater in the center
        cv2_img = synth_img.copy()

        representations = create_multiscale_representations(
            cv2_img,
            source_gsd_m=0.25,
            target_scales={"native": 0.25, "intermediate": 1.00, "coarse": 5.40},
        )

        self.assertIn("native", representations)
        self.assertIn("intermediate", representations)
        self.assertIn("coarse", representations)

        nat_arr, nat_meta = representations["native"]
        int_arr, int_meta = representations["intermediate"]
        coa_arr, coa_meta = representations["coarse"]

        self.assertEqual(nat_arr.shape, (4320, 4320))
        self.assertEqual(int_arr.shape, (1080, 1080))
        self.assertEqual(coa_arr.shape, (200, 200))

        self.assertAlmostEqual(nat_meta.resize_factor, 1.0, places=5)
        self.assertAlmostEqual(int_meta.resize_factor, 0.25, places=5)
        self.assertAlmostEqual(coa_meta.resize_factor, 1.0 / 21.6, places=4)


class TestAnchorProjectionAndCandidatePairing(unittest.TestCase):
    """Test geometric projection through ground grids and candidate crater pairing."""

    def setUp(self):
        # Mock ground grids using a small regular synthetic grid
        # 5 scans x 4 pixels
        scans = [0, 100, 200, 300, 400]
        pixels = [0, 100, 200, 300]
        rows_tmc = []
        rows_ohr = []
        for s in scans:
            for p in pixels:
                # TMC: lon increases with p, lat decreases with s
                lon = 336.0 + (p / 1000.0)
                lat = -2.0 - (s / 500.0)
                rows_tmc.append({"Longitude": lon, "Latitude": lat, "Pixel": p, "Scan": s})
                # OHRC: same geographic space, but scaled by 10x
                rows_ohr.append({"Longitude": lon, "Latitude": lat, "Pixel": p * 10, "Scan": s * 10})

        import pandas as pd
        import tempfile
        self.tmpdir = tempfile.TemporaryDirectory()
        tmc_csv = Path(self.tmpdir.name) / "tmc_grid.csv"
        ohr_csv = Path(self.tmpdir.name) / "ohr_grid.csv"
        pd.DataFrame(rows_tmc).to_csv(tmc_csv, index=False)
        pd.DataFrame(rows_ohr).to_csv(ohr_csv, index=False)

        self.grid_tmc = GroundGrid(tmc_csv, sensor_name="TMC-2")
        self.grid_ohr = GroundGrid(ohr_csv, sensor_name="OHRC")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_extract_and_project_tmc2_anchors(self):
        # Create TMC detections
        dets = [
            CraterDetection(
                center_x=50.0,
                center_y=50.0,
                radius_px=10.0,
                diameter_px=20.0,
                diameter_m=108.0,  # 20 * 5.4m
                confidence=0.85,
                source_sensor="TMC-2",
                image_region="test",
                detector_name="paper_yolov9c",
            ),
            CraterDetection(
                center_x=10.0,
                center_y=10.0,
                radius_px=4.0,
                diameter_px=8.0,
                diameter_m=43.2,  # <100m, should be filtered out
                confidence=0.90,
                source_sensor="TMC-2",
                image_region="test",
                detector_name="paper_yolov9c",
            ),
        ]

        anchors = extract_and_project_tmc2_anchors(
            tmc2_detections=dets,
            tmc2_grid=self.grid_tmc,
            ohrc_grid=self.grid_ohr,
            tmc2_crop_origin=(100, 100),
            ohrc_crop_origin=(1000, 1000),
            min_diameter_m=100.0,
        )

        self.assertEqual(len(anchors), 1)
        a = anchors[0]
        self.assertEqual(a.diameter_m, 108.0)
        self.assertAlmostEqual(a.tmc2_full_pixel, 150.0, places=2)
        self.assertAlmostEqual(a.tmc2_full_scan, 150.0, places=2)
        # Because OHRC grid is 10x pixel and scan:
        self.assertAlmostEqual(a.projected_ohrc_full_pixel, 1500.0, places=1)
        self.assertAlmostEqual(a.projected_ohrc_full_scan, 1500.0, places=1)
        self.assertAlmostEqual(a.projected_ohrc_crop_x, 500.0, places=1)
        self.assertAlmostEqual(a.projected_ohrc_crop_y, 500.0, places=1)
        self.assertAlmostEqual(a.expected_ohrc_diameter_px, 108.0 / 0.25, places=2)

    def test_candidate_crater_pairing(self):
        anchor = TMC2AnchorCandidate(
            anchor_id=0,
            source_sensor="TMC-2",
            tmc2_crop_x=50.0,
            tmc2_crop_y=50.0,
            tmc2_full_pixel=150.0,
            tmc2_full_scan=150.0,
            radius_px=10.0,
            diameter_px=20.0,
            diameter_m=100.0,
            confidence=0.85,
            detector="paper_yolov9c",
            scale="5.40m/px",
            tile_id="root",
            is_dark_region=False,
            longitude_deg=336.15,
            latitude_deg=-2.3,
            projected_ohrc_full_pixel=1500.0,
            projected_ohrc_full_scan=1500.0,
            projected_ohrc_crop_x=1080.0,  # in native 0.25m OHRC
            projected_ohrc_crop_y=1080.0,
            expected_ohrc_diameter_px=400.0,
        )

        # In coarse OHRC (5.40m/px, scale_factor = 0.25 / 5.4 = 1/21.6), projected center is:
        # 1080 / 21.6 = 50.0
        ohr_cand_near = CraterDetection(
            center_x=52.0,  # 2 pixels away = 10.8m
            center_y=50.0,
            radius_px=9.0,
            diameter_px=18.0,
            diameter_m=97.2,
            confidence=0.80,
            source_sensor="OHRC",
            image_region="test",
            detector_name="paper_yolov9c",
        )
        ohr_cand_far = CraterDetection(
            center_x=150.0,  # 100 pixels away = 540m
            center_y=150.0,
            radius_px=10.0,
            diameter_px=20.0,
            diameter_m=108.0,
            confidence=0.90,
            source_sensor="OHRC",
            image_region="test",
            detector_name="paper_yolov9c",
        )

        pairs = pair_candidate_craters(
            tmc2_anchors=[anchor],
            ohrc_detections=[ohr_cand_near, ohr_cand_far],
            ohrc_gsd_m=5.40,
            max_search_dist_m=150.0,
            ohrc_scale_label="coarse",
        )

        self.assertEqual(len(pairs), 1)
        p = pairs[0]
        self.assertEqual(p.tmc2_crater_id, 0)
        self.assertEqual(p.ohrc_crater_id, 0)
        self.assertAlmostEqual(p.center_distance_px, 2.0, places=4)
        self.assertAlmostEqual(p.center_distance_m, 10.8, places=4)
        self.assertAlmostEqual(p.diameter_ratio, 97.2 / 100.0, places=4)


class TestCraterNeighborhoodAndCNSFFeatures(unittest.TestCase):
    """Test CNSF geometric and topological feature calculations (Equations 6, 7, 8)."""

    def setUp(self):
        # Create a central crater at (100, 100) with diameter 100m
        # Place 4 neighbors at known angles: 0 deg (East), 90 deg (North), 180 deg (West), 270 deg (South)
        # Distances: 20m, 30m, 40m, 50m
        # Diameters: 50m, 60m, 70m, 80m
        gsd = 1.0  # 1 m/px
        self.gsd = gsd
        self.c0 = CraterDetection(
            center_x=100.0,
            center_y=100.0,
            radius_px=50.0,
            diameter_px=100.0,
            diameter_m=100.0,
            confidence=0.9,
            source_sensor="TEST",
            image_region="test",
            detector_name="synthetic",
        )
        self.neighbors = [
            CraterDetection(center_x=120.0, center_y=100.0, radius_px=25.0, diameter_px=50.0, diameter_m=50.0, confidence=0.8, source_sensor="TEST", image_region="test", detector_name="synthetic"),  # +20 dx -> 0 deg
            CraterDetection(center_x=100.0, center_y=130.0, radius_px=30.0, diameter_px=60.0, diameter_m=60.0, confidence=0.8, source_sensor="TEST", image_region="test", detector_name="synthetic"),  # +30 dy -> 90 deg
            CraterDetection(center_x=60.0, center_y=100.0, radius_px=35.0, diameter_px=70.0, diameter_m=70.0, confidence=0.8, source_sensor="TEST", image_region="test", detector_name="synthetic"),   # -40 dx -> 180 deg
            CraterDetection(center_x=100.0, center_y=50.0, radius_px=40.0, diameter_px=80.0, diameter_m=80.0, confidence=0.8, source_sensor="TEST", image_region="test", detector_name="synthetic"),    # -50 dy -> 270 deg
        ]

    def test_neighborhood_angles_sum_to_360(self):
        nh = construct_crater_neighborhood(
            center_id=0,
            center_x=100.0,
            center_y=100.0,
            center_diameter_m=100.0,
            all_craters=self.neighbors,
            gsd_m=self.gsd,
            k=4,
        )

        self.assertEqual(nh.k_actual, 4)
        # Azimuth angles should be 0, 90, 180, 270
        np.testing.assert_allclose(nh.azimuth_angles_deg, [0.0, 90.0, 180.0, 270.0], atol=1e-3)

        # Interior angles should each be 90 deg and sum exactly to 360 deg
        np.testing.assert_allclose(nh.interior_angles_deg, [90.0, 90.0, 90.0, 90.0], atol=1e-3)
        self.assertAlmostEqual(sum(nh.interior_angles_deg), 360.0, places=4)

    def test_normalized_distances_scale_invariance(self):
        # Eq. 6: d_i' = d_i / (1/K * sum(d_j))
        # Distances are 20, 30, 40, 50 -> mean = 35.0
        nh = construct_crater_neighborhood(
            center_id=0,
            center_x=100.0,
            center_y=100.0,
            center_diameter_m=100.0,
            all_craters=self.neighbors,
            gsd_m=self.gsd,
            k=4,
        )

        expected_norm = [20.0 / 35.0, 30.0 / 35.0, 40.0 / 35.0, 50.0 / 35.0]
        np.testing.assert_allclose(nh.normalized_distances, expected_norm, atol=1e-4)
        # Mean of normalized distances must equal exactly 1.0
        self.assertAlmostEqual(float(np.mean(nh.normalized_distances)), 1.0, places=5)

    def test_diameter_ratios(self):
        # Eq. 8: r_i = D_i / D_c
        # Central diameter = 100m. Neighbor diameters = 50, 60, 70, 80m.
        nh = construct_crater_neighborhood(
            center_id=0,
            center_x=100.0,
            center_y=100.0,
            center_diameter_m=100.0,
            all_craters=self.neighbors,
            gsd_m=self.gsd,
            k=4,
        )

        expected_ratios = [0.5, 0.6, 0.7, 0.8]
        np.testing.assert_allclose(nh.diameter_ratios, expected_ratios, atol=1e-4)

    def test_cyclic_rotation_invariance_similarity(self):
        # Construct neighborhood A
        nh_a = construct_crater_neighborhood(
            center_id=0,
            center_x=100.0,
            center_y=100.0,
            center_diameter_m=100.0,
            all_craters=self.neighbors,
            gsd_m=self.gsd,
            k=4,
        )

        # Rotate entire scene by 90 degrees around center
        # (x', y') = (100 - (y-100), 100 + (x-100))
        rotated_neighbors = []
        for n in self.neighbors:
            dx = n.center_x - 100.0
            dy = n.center_y - 100.0
            rx = 100.0 - dy
            ry = 100.0 + dx
            rotated_neighbors.append(
                CraterDetection(
                    center_x=rx,
                    center_y=ry,
                    radius_px=n.radius_px,
                    diameter_px=n.diameter_px,
                    diameter_m=n.diameter_m,
                    confidence=n.confidence,
                    source_sensor="TEST",
                    image_region="test",
                    detector_name="synthetic",
                )
            )

        nh_b = construct_crater_neighborhood(
            center_id=1,
            center_x=100.0,
            center_y=100.0,
            center_diameter_m=100.0,
            all_craters=rotated_neighbors,
            gsd_m=self.gsd,
            k=4,
        )

        sim = compute_cnsf_structural_similarity(nh_a, nh_b)
        self.assertTrue(sim["valid"])
        self.assertTrue(sim["is_structurally_plausible"])
        self.assertAlmostEqual(sim["mean_angular_error_deg"], 0.0, places=2)
        self.assertAlmostEqual(sim["mean_norm_dist_error"], 0.0, places=2)
        self.assertAlmostEqual(sim["cosine_similarity"], 1.0, places=3)


class TestEdgeCasesAndBoundaries(unittest.TestCase):
    """Test boundary conditions, empty pools, insufficient neighbors, and edge handling."""

    def test_empty_candidate_pool(self):
        nh = construct_crater_neighborhood(
            center_id=0,
            center_x=50.0,
            center_y=50.0,
            center_diameter_m=100.0,
            all_craters=[],
            gsd_m=1.0,
            k=5,
        )
        self.assertEqual(nh.k_actual, 0)
        self.assertEqual(len(nh.neighbor_ids), 0)

    def test_fewer_than_k_candidates(self):
        single_cand = [
            CraterDetection(
                center_x=60.0,
                center_y=50.0,
                radius_px=10.0,
                diameter_px=20.0,
                diameter_m=20.0,
                confidence=0.8,
                source_sensor="TEST",
                image_region="test",
                detector_name="synthetic",
            )
        ]
        nh = construct_crater_neighborhood(
            center_id=0,
            center_x=50.0,
            center_y=50.0,
            center_diameter_m=100.0,
            all_craters=single_cand,
            gsd_m=1.0,
            k=5,
        )
        self.assertEqual(nh.k_actual, 1)
        self.assertEqual(len(nh.neighbor_ids), 1)

    def test_pairing_no_craters_within_search_radius(self):
        anchor = TMC2AnchorCandidate(
            anchor_id=0,
            source_sensor="TMC-2",
            tmc2_crop_x=10.0,
            tmc2_crop_y=10.0,
            tmc2_full_pixel=10.0,
            tmc2_full_scan=10.0,
            radius_px=10.0,
            diameter_px=20.0,
            diameter_m=100.0,
            confidence=0.8,
            detector="paper_yolov9c",
            scale="5.4m/px",
            tile_id="root",
            is_dark_region=False,
            longitude_deg=336.0,
            latitude_deg=-2.0,
            projected_ohrc_full_pixel=100.0,
            projected_ohrc_full_scan=100.0,
            projected_ohrc_crop_x=216.0,  # 10.0 * 21.6
            projected_ohrc_crop_y=216.0,
            expected_ohrc_diameter_px=400.0,
        )
        # Det is far away (e.g. 500m)
        far_cand = CraterDetection(
            center_x=200.0,
            center_y=200.0,
            radius_px=5.0,
            diameter_px=10.0,
            diameter_m=54.0,
            confidence=0.8,
            source_sensor="OHRC",
            image_region="test",
            detector_name="paper_yolov9c",
        )
        pairs = pair_candidate_craters(
            tmc2_anchors=[anchor],
            ohrc_detections=[far_cand],
            ohrc_gsd_m=5.40,
            max_search_dist_m=150.0,
        )
        # Distance = (200 - 10) * 5.4 = 1026m > 150m, should not pair
        self.assertEqual(len(pairs), 0)


if __name__ == "__main__":
    unittest.main()
