"""Crater neighborhood construction and topological feature calculation.

Chunk 8 Implementation:
- K-nearest neighbor search using physical Euclidean distance.
- Exclusion of the central crater itself.
- Edge case handling when fewer than K neighbors exist.
- Invariant geometric feature extraction:
  - Normalized distance ratios (Section 2.2.2 / Eq. 6)
  - Azimuthal ordering and interior angles (Section 2.2.3 / Eq. 7)
  - Crater diameter ratios (Section 2.2.2 / Eq. 8)
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.cnsf.geometry import CraterFeature, compute_euclidean_distance_m


@dataclass
class CraterNeighborhood:
    """Represents a localized crater neighborhood graph centered at a Central Crater (CC).

    Attributes:
    - center_crater: The central crater feature (CC).
    - neighbors: List of K nearest neighbor crater features (NCs).
    - k_requested: Requested neighborhood size K (e.g. 5, 10, 15, 20).
    - k_actual: Actual number of valid neighbors found.
    - neighbor_distances_m: List of Euclidean distances from CC to each NC in meters.
    - normalized_distances: Scale-invariant distances d_i' = d_i / (1/K * sum(d_j)).
    - azimuth_angles_deg: Polar angles of NCs around CC in [0, 360) degrees.
    - interior_angles_deg: Angles between adjacent NCs in cyclic sorted order (sum = 360 deg).
    - diameter_ratios: Scale-invariant diameter ratios r_i = phi_i / phi_0.
    - search_radius_m: Physical radius used during neighbor search in meters.
    """

    center_crater: CraterFeature
    neighbors: List[CraterFeature] = field(default_factory=list)
    k_requested: int = 5
    k_actual: int = 0
    neighbor_distances_m: List[float] = field(default_factory=list)
    normalized_distances: List[float] = field(default_factory=list)
    azimuth_angles_deg: List[float] = field(default_factory=list)
    interior_angles_deg: List[float] = field(default_factory=list)
    diameter_ratios: List[float] = field(default_factory=list)
    search_radius_m: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center_crater": self.center_crater.to_dict(),
            "neighbors": [n.to_dict() for n in self.neighbors],
            "k_requested": self.k_requested,
            "k_actual": self.k_actual,
            "neighbor_distances_m": self.neighbor_distances_m,
            "normalized_distances": self.normalized_distances,
            "azimuth_angles_deg": self.azimuth_angles_deg,
            "interior_angles_deg": self.interior_angles_deg,
            "diameter_ratios": self.diameter_ratios,
            "search_radius_m": self.search_radius_m,
        }


def build_crater_neighborhood(
    center_crater: CraterFeature,
    candidate_pool: List[CraterFeature],
    k: int = 15,
    max_radius_m: Optional[float] = None,
    gsd_m: Optional[float] = None,
) -> CraterNeighborhood:
    """Construct the K-nearest crater neighborhood for a central crater.

    Args:
        center_crater: The central crater feature (CC).
        candidate_pool: Pool of all available crater detections.
        k: Number of nearest neighbors to retrieve (default: 15).
        max_radius_m: Optional maximum physical search radius in meters.
        gsd_m: Optional GSD in meters per pixel.

    Returns:
        Populated CraterNeighborhood instance.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    # Compute distances to all candidates, strictly excluding the central crater itself
    candidates_with_dist: List[Tuple[float, CraterFeature]] = []
    for cand in candidate_pool:
        # Exclude exact identity match (by ID or spatial co-location < 0.01m)
        if cand.detection_id == center_crater.detection_id:
            continue

        dist_m = compute_euclidean_distance_m(center_crater, cand, gsd_m=gsd_m)
        if dist_m < 0.1:  # co-located duplicate
            continue

        if max_radius_m is not None and dist_m > max_radius_m:
            continue

        candidates_with_dist.append((dist_m, cand))

    # Sort ascending by physical Euclidean distance
    candidates_with_dist.sort(key=lambda item: item[0])

    k_actual = min(k, len(candidates_with_dist))
    selected = candidates_with_dist[:k_actual]

    if k_actual == 0:
        return CraterNeighborhood(
            center_crater=center_crater,
            neighbors=[],
            k_requested=k,
            k_actual=0,
            search_radius_m=max_radius_m,
        )

    # Extract polar azimuth angles around central crater: alpha in [0, 360 deg)
    # Using image coordinates: dx = x_nc - x_cc, dy = y_nc - y_cc
    polar_candidates = []
    for dist_m, cand in selected:
        dx = cand.image_x - center_crater.image_x
        dy = cand.image_y - center_crater.image_y
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad) % 360.0
        polar_candidates.append({
            "cand": cand,
            "dist_m": dist_m,
            "azimuth_deg": angle_deg,
        })

    # Sort in counter-clockwise (or clockwise cyclic) order by azimuth angle
    polar_candidates.sort(key=lambda p: p["azimuth_deg"])

    sorted_neighbors = [p["cand"] for p in polar_candidates]
    sorted_dists_m = [p["dist_m"] for p in polar_candidates]
    sorted_azimuths = [p["azimuth_deg"] for p in polar_candidates]

    # Normalized distances (Eq. 6): d_i' = d_i / (1/K * sum(d_j))
    mean_dist_m = float(np.mean(sorted_dists_m)) if sorted_dists_m else 1.0
    norm_distances = [float(d / max(mean_dist_m, 1e-6)) for d in sorted_dists_m]

    # Interior angles between adjacent neighbors in cyclic sequence (sum = 360 deg)
    interior_angles: List[float] = []
    num_p = len(sorted_azimuths)
    for i in range(num_p):
        a_curr = sorted_azimuths[i]
        a_next = sorted_azimuths[(i + 1) % num_p]
        if i == num_p - 1:
            diff = (a_next + 360.0) - a_curr
        else:
            diff = a_next - a_curr
            if diff < 0.0:
                diff += 360.0
        interior_angles.append(float(diff))

    # Diameter ratios (Eq. 8): r_i = phi_i / phi_0
    phi_0 = max(center_crater.diameter_m, 1e-4)
    diameter_ratios = [float(n.diameter_m / phi_0) for n in sorted_neighbors]

    return CraterNeighborhood(
        center_crater=center_crater,
        neighbors=sorted_neighbors,
        k_requested=k,
        k_actual=k_actual,
        neighbor_distances_m=sorted_dists_m,
        normalized_distances=norm_distances,
        azimuth_angles_deg=sorted_azimuths,
        interior_angles_deg=interior_angles,
        diameter_ratios=diameter_ratios,
        search_radius_m=max_radius_m,
    )
