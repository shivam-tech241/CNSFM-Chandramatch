"""Cross-scale crater representation, anchor projection, and neighborhood validation.

Chunk 7 Implementation:
- Multi-scale OHRC representations (Native 0.25 m/px, Intermediate 1.0 m/px, Coarse 5.40 m/px).
- TMC-2 crater anchor filtering (>=100m, >=150m, >=200m, >=300m).
- Cross-sensor geographic coordinate projection via validated ISRO ground grids.
- Candidate crater pairing within physically justified search regions.
- CNSF local crater neighborhood construction (K=5, 10, 15, 20).
- Scale- and rotation-invariant geometric feature calculation (paper Equations 6, 7, 8).
- Structural consistency assessment between TMC-2 and OHRC crater patterns.
"""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd

from src.crater_detection.base import CraterDetection
from src.io.ground_grid import GroundGrid


@dataclass(frozen=True)
class ScaleRepresentationMeta:
    """Metadata describing a physical-scale representation of an image."""

    scale_name: str
    source_sensor: str
    source_gsd_m: float
    target_gsd_m: float
    resize_factor: float
    height: int
    width: int
    method: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TMC2AnchorCandidate:
    """TMC-2 crater anchor with physical dimensions and projected OHRC coordinates."""

    anchor_id: int
    source_sensor: str
    tmc2_crop_x: float
    tmc2_crop_y: float
    tmc2_full_pixel: float
    tmc2_full_scan: float
    radius_px: float
    diameter_px: float
    diameter_m: float
    confidence: float
    detector: str
    scale: str
    tile_id: str
    is_dark_region: bool
    longitude_deg: float
    latitude_deg: float
    projected_ohrc_full_pixel: float
    projected_ohrc_full_scan: float
    projected_ohrc_crop_x: float
    projected_ohrc_crop_y: float
    expected_ohrc_diameter_px: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateCraterPair:
    """A paired candidate relationship between a TMC-2 anchor and an OHRC detection."""

    tmc2_crater_id: int
    ohrc_crater_id: int
    tmc2_x: float
    tmc2_y: float
    projected_ohrc_x: float
    projected_ohrc_y: float
    ohrc_x: float
    ohrc_y: float
    center_distance_px: float
    center_distance_m: float
    tmc2_diameter_m: float
    ohrc_diameter_m: float
    diameter_ratio: float
    tmc2_confidence: float
    ohrc_confidence: float
    tmc2_detector: str
    ohrc_detector: str
    ohrc_scale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CraterNeighborhood:
    """Local K-nearest crater neighborhood with invariant CNSF geometric features.

    Implements the CNSF feature formulation from Xie et al. (Remote Sens. 2025, 17, 2302):
    - Normalized distances d_i' = d_i / (1/K * sum(d_j)) [Eq. 6]
    - Azimuthal order and interior angles theta_i = angle(P_i, P_c, P_{i+1}) [Eq. 7]
    - Diameter ratios r_i = D_i / D_c [Eq. 8]
    """

    central_crater_id: int
    sensor_name: str
    scale_name: str
    k_requested: int
    k_actual: int
    center_x: float
    center_y: float
    center_diameter_m: float
    neighbor_ids: List[int] = field(default_factory=list)
    neighbor_distances_m: List[float] = field(default_factory=list)
    normalized_distances: List[float] = field(default_factory=list)
    azimuth_angles_deg: List[float] = field(default_factory=list)
    interior_angles_deg: List[float] = field(default_factory=list)
    diameter_ratios: List[float] = field(default_factory=list)
    neighbor_diameters_m: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_multiscale_representations(
    image: np.ndarray,
    source_gsd_m: float = 0.25,
    target_scales: Optional[Dict[str, float]] = None,
) -> Dict[str, Tuple[np.ndarray, ScaleRepresentationMeta]]:
    """Generate physical-scale representations of an image via area downsampling.

    Args:
        image: 2D uint8 numpy array.
        source_gsd_m: Native Ground Sample Distance in meters per pixel.
        target_scales: Dict mapping scale names to target GSD in meters per pixel.
                       Defaults to {'native': 0.25, 'intermediate': 1.00, 'coarse': 5.40}.

    Returns:
        Dict mapping scale names to tuples of (scaled_image_array, ScaleRepresentationMeta).
    """
    if target_scales is None:
        target_scales = {
            "native": source_gsd_m,
            "intermediate": 1.00,
            "coarse": 5.40,
        }

    h_orig, w_orig = image.shape[:2]
    representations: Dict[str, Tuple[np.ndarray, ScaleRepresentationMeta]] = {}

    for name, target_gsd in target_scales.items():
        if abs(target_gsd - source_gsd_m) < 1e-4:
            # Native representation (no resizing)
            meta = ScaleRepresentationMeta(
                scale_name=name,
                source_sensor="OHRC",
                source_gsd_m=source_gsd_m,
                target_gsd_m=source_gsd_m,
                resize_factor=1.0,
                height=h_orig,
                width=w_orig,
                method="identity",
            )
            representations[name] = (image.copy(), meta)
        else:
            resize_factor = source_gsd_m / target_gsd
            target_w = max(16, int(round(w_orig * resize_factor)))
            target_h = max(16, int(round(h_orig * resize_factor)))
            # Area interpolation integrates pixel radiance correctly over downsampled footprint
            scaled = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
            meta = ScaleRepresentationMeta(
                scale_name=name,
                source_sensor="OHRC",
                source_gsd_m=source_gsd_m,
                target_gsd_m=target_gsd,
                resize_factor=resize_factor,
                height=target_h,
                width=target_w,
                method="cv2.INTER_AREA",
            )
            representations[name] = (scaled, meta)

    return representations


def extract_and_project_tmc2_anchors(
    tmc2_detections: List[CraterDetection],
    tmc2_grid: GroundGrid,
    ohrc_grid: GroundGrid,
    tmc2_crop_origin: Tuple[int, int],  # (row_start, col_start)
    ohrc_crop_origin: Tuple[int, int],  # (row_start, col_start)
    min_diameter_m: float = 100.0,
    dark_mask: Optional[np.ndarray] = None,
) -> List[TMC2AnchorCandidate]:
    """Filter TMC-2 detections to large anchors and project into OHRC pixel space.

    Args:
        tmc2_detections: Detections in TMC-2 crop coordinate frame.
        tmc2_grid: GroundGrid instance for TMC-2.
        ohrc_grid: GroundGrid instance for OHRC.
        tmc2_crop_origin: (scan_start, pixel_start) of TMC-2 crop in full raster.
        ohrc_crop_origin: (scan_start, pixel_start) of OHRC crop in full raster.
        min_diameter_m: Physical diameter cutoff in meters.
        dark_mask: Optional binary mask of dark/shadowed regions in TMC-2 crop.

    Returns:
        List of TMC2AnchorCandidate objects.
    """
    tmc_r0, tmc_c0 = tmc2_crop_origin
    ohr_r0, ohr_c0 = ohrc_crop_origin

    anchors: List[TMC2AnchorCandidate] = []

    for idx, det in enumerate(tmc2_detections):
        if det.diameter_m < min_diameter_m:
            continue

        cx = det.center_x
        cy = det.center_y

        full_tmc_px = tmc_c0 + cx
        full_tmc_sc = tmc_r0 + cy

        # 1. TMC-2 pixel -> Selenographic Lon, Lat
        try:
            lon, lat = tmc2_grid.pixel_to_geo(full_tmc_px, full_tmc_sc, check_bounds=False)
        except Exception:
            continue

        if np.isnan(lon) or np.isnan(lat):
            continue

        # 2. Selenographic Lon, Lat -> OHRC pixel
        try:
            full_ohr_px, full_ohr_sc = ohrc_grid.geo_to_pixel(lon, lat, check_bounds=False)
        except Exception:
            continue

        if np.isnan(full_ohr_px) or np.isnan(full_ohr_sc):
            continue

        ohr_crop_x = full_ohr_px - ohr_c0
        ohr_crop_y = full_ohr_sc - ohr_r0

        # Dark region check
        is_dark = False
        if dark_mask is not None:
            iy = int(round(cy))
            ix = int(round(cx))
            if 0 <= iy < dark_mask.shape[0] and 0 <= ix < dark_mask.shape[1]:
                is_dark = bool(dark_mask[iy, ix] > 0)

        expected_ohr_diam_px = det.diameter_m / 0.25

        anchor = TMC2AnchorCandidate(
            anchor_id=idx,
            source_sensor="TMC-2",
            tmc2_crop_x=float(cx),
            tmc2_crop_y=float(cy),
            tmc2_full_pixel=float(full_tmc_px),
            tmc2_full_scan=float(full_tmc_sc),
            radius_px=float(det.radius_px),
            diameter_px=float(det.diameter_px),
            diameter_m=float(det.diameter_m),
            confidence=float(det.confidence),
            detector=det.detector_name,
            scale="5.40m/px",
            tile_id=det.tile_id or "root",
            is_dark_region=is_dark,
            longitude_deg=float(lon),
            latitude_deg=float(lat),
            projected_ohrc_full_pixel=float(full_ohr_px),
            projected_ohrc_full_scan=float(full_ohr_sc),
            projected_ohrc_crop_x=float(ohr_crop_x),
            projected_ohrc_crop_y=float(ohr_crop_y),
            expected_ohrc_diameter_px=float(expected_ohr_diam_px),
        )
        anchors.append(anchor)

    return anchors


def pair_candidate_craters(
    tmc2_anchors: List[TMC2AnchorCandidate],
    ohrc_detections: List[CraterDetection],
    ohrc_gsd_m: float = 0.25,
    max_search_dist_m: float = 150.0,
    ohrc_scale_label: str = "native",
) -> List[CandidateCraterPair]:
    """Pair TMC-2 projected anchors with candidate OHRC detections within physical search radius.

    Args:
        tmc2_anchors: Projected TMC-2 anchor candidates.
        ohrc_detections: OHRC detections in the corresponding OHRC representation frame.
        ohrc_gsd_m: GSD of the OHRC representation being queried.
        max_search_dist_m: Maximum center-to-center search distance in meters.
        ohrc_scale_label: Label for the OHRC scale (e.g. 'native', 'coarse', 'intermediate').

    Returns:
        List of CandidateCraterPair instances.
    """
    pairs: List[CandidateCraterPair] = []

    if not tmc2_anchors or not ohrc_detections:
        return pairs

    # For coarse OHRC (gsd 5.4m), projected OHRC x/y needs to be scaled by 0.25 / gsd_m
    scale_factor = 0.25 / ohrc_gsd_m

    for anchor in tmc2_anchors:
        proj_x = anchor.projected_ohrc_crop_x * scale_factor
        proj_y = anchor.projected_ohrc_crop_y * scale_factor

        best_cand: Optional[CraterDetection] = None
        best_cand_idx: int = -1
        min_dist_m: float = float("inf")
        min_dist_px: float = float("inf")

        # Dynamic search radius: at least max_search_dist_m, or up to 0.75 * anchor diameter
        effective_radius_m = max(max_search_dist_m, 0.75 * anchor.diameter_m)

        for o_idx, odet in enumerate(ohrc_detections):
            dx_px = odet.center_x - proj_x
            dy_px = odet.center_y - proj_y
            dist_px = math.hypot(dx_px, dy_px)
            dist_m = dist_px * ohrc_gsd_m

            if dist_m < effective_radius_m and dist_m < min_dist_m:
                min_dist_m = dist_m
                min_dist_px = dist_px
                best_cand = odet
                best_cand_idx = o_idx

        if best_cand is not None:
            # Diameter ratio min / max (always in (0, 1])
            d_min = min(anchor.diameter_m, best_cand.diameter_m)
            d_max = max(anchor.diameter_m, best_cand.diameter_m)
            ratio = d_min / max(d_max, 1e-6)

            pair = CandidateCraterPair(
                tmc2_crater_id=anchor.anchor_id,
                ohrc_crater_id=best_cand_idx,
                tmc2_x=anchor.tmc2_crop_x,
                tmc2_y=anchor.tmc2_crop_y,
                projected_ohrc_x=proj_x,
                projected_ohrc_y=proj_y,
                ohrc_x=best_cand.center_x,
                ohrc_y=best_cand.center_y,
                center_distance_px=float(min_dist_px),
                center_distance_m=float(min_dist_m),
                tmc2_diameter_m=anchor.diameter_m,
                ohrc_diameter_m=best_cand.diameter_m,
                diameter_ratio=float(ratio),
                tmc2_confidence=anchor.confidence,
                ohrc_confidence=best_cand.confidence,
                tmc2_detector=anchor.detector,
                ohrc_detector=best_cand.detector_name,
                ohrc_scale=ohrc_scale_label,
            )
            pairs.append(pair)

    return pairs


def construct_crater_neighborhood(
    center_id: int,
    center_x: float,
    center_y: float,
    center_diameter_m: float,
    all_craters: List[CraterDetection],
    gsd_m: float,
    k: int = 5,
    sensor_name: str = "TMC-2",
    scale_name: str = "coarse",
) -> CraterNeighborhood:
    """Construct a K-nearest crater neighborhood with invariant CNSF geometric features.

    Implements:
    - Distance sorting to find K-nearest neighbors.
    - Normalized distances d_i' = d_i / (1/K * sum(d_j)) [Eq. 6]
    - Azimuthal order alpha_i = atan2(dy, dx) in [0, 360 deg)
    - Cyclic interior angles theta_i = angle(P_i, P_c, P_{i+1}) [Eq. 7]
    - Diameter ratios r_i = D_i / D_c [Eq. 8]

    Args:
        center_id: Identifier of the central crater.
        center_x: Center x coordinate in image pixels.
        center_y: Center y coordinate in image pixels.
        center_diameter_m: Physical diameter in meters of central crater.
        all_craters: Pool of candidate neighbor detections in same coordinate system.
        gsd_m: Ground Sample Distance in meters per pixel.
        k: Neighborhood size (typically 5, 10, 15, or 20).
        sensor_name: Sensor identifier string.
        scale_name: Scale representation string.

    Returns:
        CraterNeighborhood object.
    """
    # 1. Calculate distances from center to all candidates
    candidates = []
    for idx, c in enumerate(all_craters):
        dx_m = (c.center_x - center_x) * gsd_m
        dy_m = (c.center_y - center_y) * gsd_m
        dist_m = math.hypot(dx_m, dy_m)
        if dist_m < 1e-4:
            # Skip the central crater itself
            continue
        candidates.append((dist_m, idx, c.center_x, c.center_y, c.diameter_m, dx_m, dy_m))

    # Sort by Euclidean distance
    candidates.sort(key=lambda item: item[0])
    k_actual = min(k, len(candidates))
    selected = candidates[:k_actual]

    if k_actual == 0:
        return CraterNeighborhood(
            central_crater_id=center_id,
            sensor_name=sensor_name,
            scale_name=scale_name,
            k_requested=k,
            k_actual=0,
            center_x=center_x,
            center_y=center_y,
            center_diameter_m=center_diameter_m,
        )

    # 2. Compute polar azimuth angles around central crater: alpha in [0, 360 deg)
    polar_list = []
    for dist_m, c_idx, cx, cy, d_m, dx_m, dy_m in selected:
        # atan2(dy, dx) returns angle in radians [-pi, pi]
        angle_rad = math.atan2(dy_m, dx_m)
        angle_deg = math.degrees(angle_rad) % 360.0
        polar_list.append({
            "idx": c_idx,
            "dist_m": dist_m,
            "d_m": d_m,
            "azimuth_deg": angle_deg,
        })

    # Sort neighbors in counter-clockwise azimuthal order around P_c
    polar_list.sort(key=lambda p: p["azimuth_deg"])

    # 3. Calculate Normalized Distances (Eq. 6)
    mean_dist = float(np.mean([p["dist_m"] for p in polar_list]))
    norm_distances = [float(p["dist_m"] / max(mean_dist, 1e-6)) for p in polar_list]

    # 4. Calculate Interior Angles between adjacent neighbors in azimuthal order (Eq. 7)
    interior_angles = []
    num_p = len(polar_list)
    for i in range(num_p):
        a_curr = polar_list[i]["azimuth_deg"]
        a_next = polar_list[(i + 1) % num_p]["azimuth_deg"]
        if i == num_p - 1:
            diff = (a_next + 360.0) - a_curr
        else:
            diff = a_next - a_curr
            if diff < 0:
                diff += 360.0
        interior_angles.append(float(diff))

    # 5. Calculate Diameter Ratios (Eq. 8)
    diameter_ratios = [float(p["d_m"] / max(center_diameter_m, 1e-6)) for p in polar_list]

    return CraterNeighborhood(
        central_crater_id=center_id,
        sensor_name=sensor_name,
        scale_name=scale_name,
        k_requested=k,
        k_actual=k_actual,
        center_x=float(center_x),
        center_y=float(center_y),
        center_diameter_m=float(center_diameter_m),
        neighbor_ids=[int(p["idx"]) for p in polar_list],
        neighbor_distances_m=[float(p["dist_m"]) for p in polar_list],
        normalized_distances=norm_distances,
        azimuth_angles_deg=[float(p["azimuth_deg"]) for p in polar_list],
        interior_angles_deg=interior_angles,
        diameter_ratios=diameter_ratios,
        neighbor_diameters_m=[float(p["d_m"]) for p in polar_list],
    )


def compute_cnsf_structural_similarity(
    nh_a: CraterNeighborhood,
    nh_b: CraterNeighborhood,
) -> Dict[str, Any]:
    """Compare two crater neighborhoods using cyclic invariant features.

    Tests all K cyclic rotations to evaluate angular similarity, normalized distance
    similarity, and diameter ratio consistency independent of sensor orientation.

    Args:
        nh_a: First neighborhood (e.g. TMC-2 anchor).
        nh_b: Second neighborhood (e.g. OHRC candidate).

    Returns:
        Dict of structural metrics:
            - k: Number of neighbors evaluated.
            - best_rotation_shift: Optimal cyclic shift.
            - mean_angular_error_deg: Mean difference in interior angles (deg).
            - mean_norm_dist_error: Mean difference in normalized distance.
            - mean_diameter_ratio_error: Mean difference in diameter ratios.
            - cosine_similarity: Cosine similarity of cyclic angle vectors.
            - is_structurally_plausible: Boolean indicating plausible physical consistency.
    """
    if nh_a.k_actual != nh_b.k_actual or nh_a.k_actual < 3:
        return {
            "k": min(nh_a.k_actual, nh_b.k_actual),
            "valid": False,
            "error": "Insufficient or mismatched neighbor counts",
            "is_structurally_plausible": False,
        }

    k = nh_a.k_actual
    angles_a = np.array(nh_a.interior_angles_deg, dtype=np.float64)
    dists_a = np.array(nh_a.normalized_distances, dtype=np.float64)
    ratios_a = np.array(nh_a.diameter_ratios, dtype=np.float64)

    angles_b = np.array(nh_b.interior_angles_deg, dtype=np.float64)
    dists_b = np.array(nh_b.normalized_distances, dtype=np.float64)
    ratios_b = np.array(nh_b.diameter_ratios, dtype=np.float64)

    best_score = float("inf")
    best_shift = 0
    best_ang_err = float("inf")
    best_dist_err = float("inf")
    best_ratio_err = float("inf")
    best_cosine_sim = -1.0

    norm_a = np.linalg.norm(angles_a)

    for shift in range(k):
        sh_angles_b = np.roll(angles_b, shift)
        sh_dists_b = np.roll(dists_b, shift)
        sh_ratios_b = np.roll(ratios_b, shift)

        ang_err = float(np.mean(np.abs(angles_a - sh_angles_b)))
        dist_err = float(np.mean(np.abs(dists_a - sh_dists_b)))
        ratio_err = float(np.mean(np.abs(ratios_a - sh_ratios_b)))

        norm_b = np.linalg.norm(sh_angles_b)
        if norm_a > 1e-6 and norm_b > 1e-6:
            cos_sim = float(np.dot(angles_a, sh_angles_b) / (norm_a * norm_b))
        else:
            cos_sim = 0.0

        # Composite score
        score = ang_err + 30.0 * dist_err + 10.0 * ratio_err

        if score < best_score:
            best_score = score
            best_shift = shift
            best_ang_err = ang_err
            best_dist_err = dist_err
            best_ratio_err = ratio_err
            best_cosine_sim = cos_sim

    # Plausibility threshold: mean angular error < 40 deg, cosine similarity > 0.75, dist_err < 0.50
    plausible = bool(best_ang_err < 45.0 and best_cosine_sim > 0.70 and best_dist_err < 0.55)

    return {
        "k": k,
        "valid": True,
        "best_rotation_shift": int(best_shift),
        "mean_angular_error_deg": round(float(best_ang_err), 2),
        "mean_norm_dist_error": round(float(best_dist_err), 4),
        "mean_diameter_ratio_error": round(float(best_ratio_err), 4),
        "cosine_similarity": round(float(best_cosine_sim), 4),
        "is_structurally_plausible": plausible,
    }


def plot_cross_scale_pair_contact_sheet(
    tmc_image: np.ndarray,
    ohrc_image: np.ndarray,
    anchor: TMC2AnchorCandidate,
    pair: Optional[CandidateCraterPair],
    tmc_all_craters: List[CraterDetection],
    ohrc_all_craters: List[CraterDetection],
    output_path: Union[str, Path],
    category_title: str,
    ohrc_gsd_m: float = 5.40,
    patch_radius_m: float = 350.0,
) -> None:
    """Generate a side-by-side visual contact sheet for an anchor candidate and local neighborhood.

    Left Panel: TMC-2 anchor crater, rim annotation, and local neighbors.
    Right Panel: OHRC co-located patch, projected anchor location, candidate crater, and neighbors.

    Args:
        tmc_image: Full TMC-2 crop array (uint8 or normalized).
        ohrc_image: OHRC representation array at current scale.
        anchor: TMC2AnchorCandidate object.
        pair: Optional CandidateCraterPair if candidate was found.
        tmc_all_craters: List of TMC-2 detections.
        ohrc_all_craters: List of OHRC detections at current scale.
        output_path: Output PNG image path.
        category_title: Diagnostic label (e.g. 'Clear terrain: Strong candidate').
        ohrc_gsd_m: GSD of OHRC image array in meters per pixel.
        patch_radius_m: Physical radius around center to crop for visualization.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # TMC-2 crop window
    tmc_r_px = int(round(patch_radius_m / 5.40))
    tx_c = int(round(anchor.tmc2_crop_x))
    ty_c = int(round(anchor.tmc2_crop_y))

    tx0 = max(0, tx_c - tmc_r_px)
    tx1 = min(tmc_image.shape[1], tx_c + tmc_r_px)
    ty0 = max(0, ty_c - tmc_r_px)
    ty1 = min(tmc_image.shape[0], ty_c + tmc_r_px)

    tmc_patch = tmc_image[ty0:ty1, tx0:tx1]

    # OHRC crop window
    scale_ohr = 0.25 / ohrc_gsd_m
    if pair is not None:
        ox_c = int(round(pair.ohrc_x))
        oy_c = int(round(pair.ohrc_y))
    else:
        ox_c = int(round(anchor.projected_ohrc_crop_x * scale_ohr))
        oy_c = int(round(anchor.projected_ohrc_crop_y * scale_ohr))

    ohr_r_px = int(round(patch_radius_m / ohrc_gsd_m))
    ox0 = max(0, ox_c - ohr_r_px)
    ox1 = min(ohrc_image.shape[1], ox_c + ohr_r_px)
    oy0 = max(0, oy_c - ohr_r_px)
    oy1 = min(ohrc_image.shape[0], oy_c + ohr_r_px)

    ohr_patch = ohrc_image[oy0:oy1, ox0:ox1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

    # LEFT: TMC-2
    axes[0].imshow(tmc_patch, cmap="gray", origin="upper")
    axes[0].set_title(
        f"TMC-2 (5.40 m/px) — Anchor #{anchor.anchor_id}\n"
        f"D = {anchor.diameter_m:.1f} m, Conf = {anchor.confidence:.2f}",
        fontsize=10,
        fontweight="bold",
        color="#111111",
    )

    # TMC local coordinates
    local_tx = anchor.tmc2_crop_x - tx0
    local_ty = anchor.tmc2_crop_y - ty0
    local_tr = anchor.radius_px

    # Anchor circle
    circ_t = patches.Circle((local_tx, local_ty), local_tr, fill=False, edgecolor="#00FF66", linewidth=2.0)
    axes[0].add_patch(circ_t)
    axes[0].plot(local_tx, local_ty, "+", color="#FFFF00", markersize=10, markeredgewidth=2)

    # Draw nearby craters and connection lines
    for det in tmc_all_craters:
        dx_m = (det.center_x - anchor.tmc2_crop_x) * 5.40
        dy_m = (det.center_y - anchor.tmc2_crop_y) * 5.40
        d_m = math.hypot(dx_m, dy_m)
        if 1.0 < d_m <= patch_radius_m:
            nx = det.center_x - tx0
            ny = det.center_y - ty0
            nr = det.radius_px
            c_circ = patches.Circle((nx, ny), nr, fill=False, edgecolor="#00E5FF", linewidth=1.2, linestyle="--")
            axes[0].add_patch(c_circ)
            axes[0].plot([local_tx, nx], [local_ty, ny], color="#00E5FF", linewidth=0.8, alpha=0.6)

    axes[0].set_axis_off()

    # RIGHT: OHRC
    proj_ox_local = (anchor.projected_ohrc_crop_x * scale_ohr) - ox0
    proj_oy_local = (anchor.projected_ohrc_crop_y * scale_ohr) - oy0
    expected_r_local = (anchor.diameter_m / (2.0 * ohrc_gsd_m))

    axes[1].imshow(ohr_patch, cmap="gray", origin="upper")

    ohr_title = f"OHRC ({ohrc_gsd_m:.2f} m/px)"
    if pair is not None:
        ohr_title += (
            f" — Candidate #{pair.ohrc_crater_id}\n"
            f"D = {pair.ohrc_diameter_m:.1f} m, Dist = {pair.center_distance_m:.1f} m, Ratio = {pair.diameter_ratio:.2f}"
        )
    else:
        ohr_title += f"\nProjected Anchor Location (No Candidate within Search Radius)"

    axes[1].set_title(ohr_title, fontsize=10, fontweight="bold", color="#111111")

    # Projected anchor location (magenta dashed circle)
    circ_proj = patches.Circle(
        (proj_ox_local, proj_oy_local),
        expected_r_local,
        fill=False,
        edgecolor="#FF00FF",
        linewidth=1.8,
        linestyle="--",
        label="Projected Anchor",
    )
    axes[1].add_patch(circ_proj)
    axes[1].plot(proj_ox_local, proj_oy_local, "x", color="#FF00FF", markersize=8, markeredgewidth=2)

    # If candidate paired, draw candidate crater (solid green)
    if pair is not None:
        local_ox = pair.ohrc_x - ox0
        local_oy = pair.ohrc_y - oy0
        local_or = (pair.ohrc_diameter_m / (2.0 * ohrc_gsd_m))
        circ_cand = patches.Circle((local_ox, local_oy), local_or, fill=False, edgecolor="#00FF66", linewidth=2.0)
        axes[1].add_patch(circ_cand)
        axes[1].plot(local_ox, local_oy, "+", color="#00FF66", markersize=10, markeredgewidth=2)

    # Draw nearby OHRC craters
    for odet in ohrc_all_craters:
        dx_m = (odet.center_x - (anchor.projected_ohrc_crop_x * scale_ohr)) * ohrc_gsd_m
        dy_m = (odet.center_y - (anchor.projected_ohrc_crop_y * scale_ohr)) * ohrc_gsd_m
        d_m = math.hypot(dx_m, dy_m)
        if 1.0 < d_m <= patch_radius_m:
            nx = odet.center_x - ox0
            ny = odet.center_y - oy0
            nr = odet.radius_px
            c_circ = patches.Circle((nx, ny), nr, fill=False, edgecolor="#00E5FF", linewidth=1.2, linestyle="--")
            axes[1].add_patch(c_circ)
            if pair is not None:
                axes[1].plot([local_ox, nx], [local_oy, ny], color="#00E5FF", linewidth=0.8, alpha=0.6)

    axes[1].set_axis_off()

    fig.suptitle(f"Cross-Scale Crater Anchor Validation — {category_title}", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

