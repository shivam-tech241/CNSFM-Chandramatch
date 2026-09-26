"""Unit tests for Chunk 6: Crater Detection module.

Tests cover:
- CraterDetection dataclass creation, validation, and property consistency.
- Coordinate conversion from tile-relative to global image coordinates.
- Physical crater-size conversion ($D_m = D_{px} \times GSD$).
- Circle IoU computation and geometric edge cases.
- Fast spatial-grid NMS duplicate suppression.
- Serialization and deserialization (CSV, JSON, official CNSFM '.craters' format).
- Malformed detection handling and exception raising.
- Detector configuration validation.
- ExternalFileCraterDetector and BaselineRimDetector execution on synthetic imagery.
- Statistical analysis and scale comparison functions.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.crater_detection import (
    BaseCraterDetector,
    BaselineCraterDetector,
    CraterDetection,
    ExternalFileCraterDetector,
    Tile,
    compute_circle_iou,
    compute_dark_region_breakdown,
    compute_detection_statistics,
    compute_distribution_stats,
    compute_scale_comparison,
    compute_sector_statistics,
    generate_tiles,
    load_detections_craters,
    load_detections_csv,
    load_detections_json,
    map_tile_detection_to_global,
    run_tiled_detection,
    save_detections_craters,
    save_detections_csv,
    save_detections_json,
    suppress_duplicate_detections,
)


class TestCraterDetectionDataclass:
    """Tests for CraterDetection dataclass."""

    def test_basic_instantiation_and_properties(self):
        det = CraterDetection(
            center_x=120.5,
            center_y=340.25,
            radius_px=15.0,
            diameter_px=30.0,
            diameter_m=7.5,
            confidence=0.85,
            source_sensor="OHRC",
            image_region="OHRC_Benchmark",
            detector_name="TestDetector",
        )
        assert det.center_x == 120.5
        assert det.center_y == 340.25
        assert det.radius_px == 15.0
        assert det.diameter_px == 30.0
        assert det.diameter_m == 7.5
        assert det.confidence == 0.85
        assert det.source_sensor == "OHRC"

    def test_radius_diameter_auto_consistency(self):
        # Only radius supplied
        det1 = CraterDetection(
            center_x=50.0,
            center_y=50.0,
            radius_px=20.0,
            diameter_px=0.0,  # Should auto-fill to 40.0
            diameter_m=10.0,
            confidence=0.7,
            source_sensor="OHRC",
            image_region="region_a",
            detector_name="Test",
        )
        assert det1.diameter_px == 40.0

        # Only diameter supplied
        det2 = CraterDetection(
            center_x=50.0,
            center_y=50.0,
            radius_px=0.0,  # Should auto-fill to 25.0
            diameter_px=50.0,
            diameter_m=12.5,
            confidence=0.7,
            source_sensor="OHRC",
            image_region="region_a",
            detector_name="Test",
        )
        assert det2.radius_px == 25.0

    def test_confidence_clipping(self):
        det_over = CraterDetection(
            center_x=10.0, center_y=10.0, radius_px=5.0, diameter_px=10.0, diameter_m=2.5,
            confidence=1.45, source_sensor="OHRC", image_region="test", detector_name="Test",
        )
        assert det_over.confidence == 1.0

        det_under = CraterDetection(
            center_x=10.0, center_y=10.0, radius_px=5.0, diameter_px=10.0, diameter_m=2.5,
            confidence=-0.35, source_sensor="OHRC", image_region="test", detector_name="Test",
        )
        assert det_under.confidence == 0.0

    def test_craters_format_serialization(self):
        det = CraterDetection(
            center_x=100.25,
            center_y=200.75,
            radius_px=12.5,
            diameter_px=25.0,
            diameter_m=6.25,
            confidence=0.9123,
            source_sensor="OHRC",
            image_region="test",
            detector_name="Test",
            crater_id=42,
        )
        line = det.to_craters_line()
        assert line == "100.25, 200.75, 25.00, 25.00, 0.9123, 42;"

    def test_craters_format_deserialization(self):
        line = "150.50, 300.20, 40.00, 40.00, 0.8400, 7;"
        det = CraterDetection.from_craters_line(
            line=line,
            source_sensor="OHRC",
            gsd_m=0.25,
            image_region="Benchmark",
        )
        assert det.center_x == 150.50
        assert det.center_y == 300.20
        assert det.diameter_px == 40.00
        assert det.radius_px == 20.00
        assert det.diameter_m == 10.00  # 40 * 0.25
        assert det.confidence == 0.8400
        assert det.crater_id == 7

    def test_malformed_craters_line(self):
        with pytest.raises(ValueError, match="Malformed '.craters' line"):
            CraterDetection.from_craters_line("10.0, 20.0, 30.0;", source_sensor="OHRC", gsd_m=0.25)


class TestFileIO:
    """Tests for CSV, JSON, and .craters I/O."""

    def test_craters_file_roundtrip(self, tmp_path):
        craters_path = tmp_path / "test.craters"
        dets = [
            CraterDetection(10.0, 20.0, 5.0, 10.0, 2.5, 0.8, "OHRC", "reg", "det", crater_id=0),
            CraterDetection(50.0, 60.0, 8.0, 16.0, 4.0, 0.9, "OHRC", "reg", "det", crater_id=1),
        ]
        save_detections_craters(dets, craters_path)
        assert craters_path.is_file()

        loaded = load_detections_craters(craters_path, source_sensor="OHRC", gsd_m=0.25)
        assert len(loaded) == 2
        assert loaded[0].center_x == 10.0
        assert loaded[1].crater_id == 1

    def test_csv_file_roundtrip(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        dets = [
            CraterDetection(12.345, 67.890, 10.0, 20.0, 5.0, 0.7777, "OHRC", "reg", "det", crater_id=0),
            CraterDetection(88.0, 99.0, 15.0, 30.0, 162.0, 0.8888, "TMC-2", "reg", "det", crater_id=1),
        ]
        save_detections_csv(dets, csv_path)
        assert csv_path.is_file()

        loaded = load_detections_csv(csv_path)
        assert len(loaded) == 2
        assert pytest.approx(loaded[0].center_x, 0.001) == 12.345
        assert loaded[1].source_sensor == "TMC-2"
        assert pytest.approx(loaded[1].diameter_m, 0.01) == 162.0

    def test_json_file_roundtrip(self, tmp_path):
        json_path = tmp_path / "test.json"
        dets = [
            CraterDetection(100.0, 200.0, 10.0, 20.0, 5.0, 0.85, "OHRC", "reg", "det", crater_id=0),
        ]
        save_detections_json(dets, json_path)
        assert json_path.is_file()

        loaded = load_detections_json(json_path)
        assert len(loaded) == 1
        assert loaded[0].center_x == 100.0
        assert loaded[0].confidence == 0.85


class TestTilingAndCoordinates:
    """Tests for tiling generation, coordinate translation, and NMS duplicate suppression."""

    def test_generate_tiles_grid(self):
        shape = (1000, 1000)
        tiles = generate_tiles(shape, tile_size=640, overlap=64)
        assert len(tiles) == 4  # 2 rows x 2 cols

        for t in tiles:
            assert t.width <= 640
            assert t.height <= 640
            assert t.row_end <= 1000
            assert t.col_end <= 1000

    def test_generate_tiles_small_image(self):
        shape = (300, 400)
        tiles = generate_tiles(shape, tile_size=640, overlap=64)
        assert len(tiles) == 1
        assert tiles[0].height == 300
        assert tiles[0].width == 400

    def test_generate_tiles_invalid_params(self):
        with pytest.raises(ValueError):
            generate_tiles((500, 500), tile_size=0)
        with pytest.raises(ValueError):
            generate_tiles((500, 500), tile_size=640, overlap=640)

    def test_coordinate_mapping(self):
        tile_det = CraterDetection(
            center_x=45.0,
            center_y=60.0,
            radius_px=10.0,
            diameter_px=20.0,
            diameter_m=5.0,
            confidence=0.8,
            source_sensor="OHRC",
            image_region="tile_r1_c2",
            detector_name="Test",
        )
        global_det = map_tile_detection_to_global(tile_det, row_start=576, col_start=1152, tile_id="tile_r1_c2")
        assert global_det.center_x == 45.0 + 1152
        assert global_det.center_y == 60.0 + 576
        assert global_det.tile_id == "tile_r1_c2"


class TestCircleIoUAndNMS:
    """Tests for geometric IoU and Non-Maximum Suppression."""

    def test_circle_iou_identical(self):
        iou = compute_circle_iou(100.0, 100.0, 20.0, 100.0, 100.0, 20.0)
        assert pytest.approx(iou, 1e-4) == 1.0

    def test_circle_iou_disjoint(self):
        iou = compute_circle_iou(100.0, 100.0, 20.0, 200.0, 200.0, 20.0)
        assert iou == 0.0

    def test_circle_iou_concentric_different_radii(self):
        # Circle 1: r=10 (area=100pi), Circle 2: r=20 (area=400pi)
        # Intersection is circle 1 area (100pi), Union is circle 2 area (400pi)
        # Expected IoU: 100 / 400 = 0.25
        iou = compute_circle_iou(100.0, 100.0, 10.0, 100.0, 100.0, 20.0)
        assert pytest.approx(iou, 1e-4) == 0.25

    def test_suppress_duplicate_detections(self):
        d1 = CraterDetection(100.0, 100.0, 20.0, 40.0, 10.0, 0.90, "OHRC", "reg", "det", crater_id=0)
        # Duplicate with slightly shifted center and lower confidence
        d2 = CraterDetection(102.0, 101.0, 20.0, 40.0, 10.0, 0.70, "OHRC", "reg", "det", crater_id=1)
        # Distinct crater far away
        d3 = CraterDetection(400.0, 400.0, 25.0, 50.0, 12.5, 0.85, "OHRC", "reg", "det", crater_id=2)

        dedup = suppress_duplicate_detections([d1, d2, d3], iou_threshold=0.35)
        assert len(dedup) == 2
        # d1 should be kept (higher confidence), d2 should be suppressed
        assert dedup[0].confidence == 0.90
        assert dedup[1].confidence == 0.85

    def test_suppress_duplicate_empty(self):
        assert suppress_duplicate_detections([]) == []


class TestDetectorInterface:
    """Tests for BaseCraterDetector, configuration validation, and ExternalFileCraterDetector."""

    def test_abstract_class_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseCraterDetector()

    def test_invalid_detector_parameters(self):
        with pytest.raises(ValueError, match="Invalid radius range"):
            BaselineCraterDetector(min_radius_px=100.0, max_radius_px=50.0)

        with pytest.raises(ValueError, match="Confidence threshold must be in"):
            BaselineCraterDetector(confidence_threshold=1.5)

    def test_external_file_detector(self, tmp_path):
        p = tmp_path / "craters.craters"
        p.write_text("100.0, 200.0, 30.0, 30.0, 0.85, 0;\n300.0, 400.0, 20.0, 20.0, 0.25, 1;\n")

        ext_det = ExternalFileCraterDetector(filepath=p, confidence_threshold=0.50)
        dummy_img = np.zeros((500, 500), dtype=np.uint8)
        results = ext_det.detect(dummy_img, gsd_m=0.25, source_sensor="OHRC")

        # Second detection has confidence 0.25 < 0.50, so only 1 returned
        assert len(results) == 1
        assert results[0].center_x == 100.0
        assert results[0].confidence == 0.85


class TestBaselineCraterDetector:
    """Tests for BaselineCraterDetector on synthetic imagery."""

    def test_synthetic_crater_detection(self):
        # Create synthetic image with a prominent circular ring
        img = np.full((300, 300), 50, dtype=np.uint8)
        cv2.circle(img, (150, 150), 30, color=220, thickness=4)
        # Inner shadow
        cv2.circle(img, (145, 150), 20, color=20, thickness=-1)

        detector = BaselineCraterDetector(
            min_radius_px=15.0,
            max_radius_px=50.0,
            min_dist_px=20.0,
            canny_param1=50.0,
            accumulator_param2=25.0,
            confidence_threshold=0.20,
            dp=1.2,
        )

        detections = detector.detect(img, gsd_m=0.25, source_sensor="OHRC")
        assert len(detections) >= 1

        best = max(detections, key=lambda d: d.confidence)
        assert pytest.approx(best.center_x, abs=10) == 150
        assert pytest.approx(best.center_y, abs=10) == 150
        assert pytest.approx(best.radius_px, abs=10) == 30
        assert best.confidence >= 0.20
        assert best.diameter_m == best.diameter_px * 0.25

    def test_run_tiled_detection_pipeline(self):
        img = np.full((800, 800), 40, dtype=np.uint8)
        cv2.circle(img, (200, 200), 30, color=220, thickness=4)
        cv2.circle(img, (600, 600), 30, color=220, thickness=4)

        detector = BaselineCraterDetector(
            min_radius_px=15.0,
            max_radius_px=50.0,
            min_dist_px=20.0,
            canny_param1=50.0,
            accumulator_param2=25.0,
            confidence_threshold=0.20,
            dp=1.2,
        )

        raw, post, meta = run_tiled_detection(
            detector=detector,
            image=img,
            gsd_m=0.25,
            source_sensor="OHRC",
            image_region="TestRegion",
            tile_size=480,
            overlap=64,
        )
        assert meta["num_tiles"] == 4
        assert len(post) >= 2


class TestAnalysisRoutines:
    """Tests for statistical analysis, sector splitting, dark region breakdown, and scale comparison."""

    def test_compute_distribution_stats(self):
        empty_stats = compute_distribution_stats([])
        assert empty_stats["count"] == 0
        assert empty_stats["mean"] == 0.0

        stats = compute_distribution_stats([10.0, 20.0, 30.0, 40.0, 50.0])
        assert stats["count"] == 5
        assert stats["mean"] == 30.0
        assert stats["median"] == 30.0
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0

    def test_compute_sector_statistics(self):
        dets = [
            CraterDetection(50, 100, 10, 20, 5.0, 0.8, "OHRC", "reg", "det"),  # Top
            CraterDetection(50, 500, 10, 20, 5.0, 0.7, "OHRC", "reg", "det"),  # Center
            CraterDetection(50, 800, 10, 20, 5.0, 0.9, "OHRC", "reg", "det"),  # Bottom
        ]
        sectors = compute_sector_statistics(dets, image_height=900, num_sectors=3)
        assert sectors["Top"]["count"] == 1
        assert sectors["Center"]["count"] == 1
        assert sectors["Bottom"]["count"] == 1

    def test_compute_dark_region_breakdown(self):
        dark_mask = np.zeros((100, 100), dtype=np.uint8)
        dark_mask[0:50, :] = 255  # Top half is dark

        dets = [
            CraterDetection(50, 25, 5, 10, 2.5, 0.8, "OHRC", "reg", "det"),  # Inside dark
            CraterDetection(50, 75, 5, 10, 2.5, 0.8, "OHRC", "reg", "det"),  # Outside dark
        ]
        breakdown = compute_dark_region_breakdown(dets, dark_mask)
        assert breakdown["inside_dark_proxy"]["count"] == 1
        assert breakdown["outside_dark_proxy"]["count"] == 1
        assert breakdown["inside_dark_proxy"]["percentage"] == 50.0

    def test_compute_scale_comparison(self):
        ohrc_dets = [
            CraterDetection(10, 10, 10, 20, 5.0, 0.8, "OHRC", "reg", "det"),
            CraterDetection(20, 20, 40, 80, 20.0, 0.8, "OHRC", "reg", "det"),
        ]
        tmc_dets = [
            CraterDetection(10, 10, 10, 20, 108.0, 0.8, "TMC-2", "reg", "det"),
            CraterDetection(20, 20, 20, 40, 216.0, 0.8, "TMC-2", "reg", "det"),
        ]
        res = compute_scale_comparison(ohrc_dets, tmc_dets, ohrc_gsd=0.25, tmc2_gsd=5.40)
        assert res["resolution_ratio"] == 21.6
        assert res["physical_size_range_m"]["has_shared_overlap_window"] is False
        assert res["physical_size_range_m"]["physical_scale_gap_m"] == 108.0 - 20.0
