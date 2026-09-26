"""Crater detection package for extracting crater locations, radii, and features.

Provides:
- CraterDetection dataclass with complete physical and metadata fields.
- BaseCraterDetector abstract interface.
- ExternalFileCraterDetector for CNSFM .craters and CSV detection files.
- BaselineCraterDetector based on circular Hough transform and photometric rim verification.
- Tiling generator, coordinate transformation, and circle NMS deduplication.
- Analysis routines: detection statistics, scale analysis, sector breakdown, dark-region breakdown.
- Visualization routines: overlays, scale histograms, spatial densities, and QC contact sheets.
"""

from src.crater_detection.analysis import (
    compute_dark_region_breakdown,
    compute_detection_statistics,
    compute_distribution_stats,
    compute_scale_comparison,
    compute_sector_statistics,
)
from src.crater_detection.base import (
    BaseCraterDetector,
    CraterDetection,
    ExternalFileCraterDetector,
    load_detections_craters,
    load_detections_csv,
    load_detections_json,
    save_detections_craters,
    save_detections_csv,
    save_detections_json,
)
from src.crater_detection.baseline import BaselineCraterDetector
from src.crater_detection.yolo import YOLOv9CraterDetector
from src.crater_detection.tiling import (
    Tile,
    compute_circle_iou,
    generate_tiles,
    map_tile_detection_to_global,
    run_tiled_detection,
    suppress_duplicate_detections,
)
from src.crater_detection.cross_scale import (
    CandidateCraterPair,
    CraterNeighborhood,
    ScaleRepresentationMeta,
    TMC2AnchorCandidate,
    compute_cnsf_structural_similarity,
    construct_crater_neighborhood,
    create_multiscale_representations,
    extract_and_project_tmc2_anchors,
    pair_candidate_craters,
    plot_cross_scale_pair_contact_sheet,
)
from src.crater_detection.visualization import (
    create_qc_contact_sheet,
    plot_crater_overlay,
    plot_scale_comparison,
    plot_spatial_density,
)

__all__ = [
    "CraterDetection",
    "BaseCraterDetector",
    "ExternalFileCraterDetector",
    "BaselineCraterDetector",
    "YOLOv9CraterDetector",
    "Tile",
    "generate_tiles",
    "map_tile_detection_to_global",
    "compute_circle_iou",
    "suppress_duplicate_detections",
    "run_tiled_detection",
    "save_detections_csv",
    "load_detections_csv",
    "save_detections_json",
    "load_detections_json",
    "save_detections_craters",
    "load_detections_craters",
    "compute_distribution_stats",
    "compute_detection_statistics",
    "compute_sector_statistics",
    "compute_dark_region_breakdown",
    "compute_scale_comparison",
    "plot_crater_overlay",
    "plot_scale_comparison",
    "plot_spatial_density",
    "create_qc_contact_sheet",
    "ScaleRepresentationMeta",
    "TMC2AnchorCandidate",
    "CandidateCraterPair",
    "CraterNeighborhood",
    "create_multiscale_representations",
    "extract_and_project_tmc2_anchors",
    "pair_candidate_craters",
    "construct_crater_neighborhood",
    "compute_cnsf_structural_similarity",
    "plot_cross_scale_pair_contact_sheet",
]
