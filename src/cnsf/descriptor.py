"""Crater Neighborhood Structure Feature (CNSF) descriptor representation.

Chunk 8 Implementation:
- Decomposes a crater neighborhood into K(K-1)/2 clockwise angular structures (Section 2.2.3).
- Constructs scale-, rotation-, and translation-invariant geometric features (Equations 4, 6, 7, 8).
- Assembles feature vectors D for cosine distance measurement (Equation 10).
- Guarantees deterministic descriptor generation.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.cnsf.geometry import AngularStructure, CraterFeature, compute_clockwise_angle_deg
from src.cnsf.neighborhood import CraterNeighborhood


@dataclass
class CNSFDescriptor:
    """Complete Crater Neighborhood Structure Feature (CNSF) descriptor.

    Attributes:
    - center_crater: Central crater feature (CC).
    - neighbors: List of K nearest neighbor crater features (NCs).
    - k: Number of neighbors in the descriptor.
    - angular_structures: List of K(K-1)/2 AngularStructure objects.
    - cyclic_interior_angles: Sequence of adjacent interior angles summing to 360 degrees.
    - normalized_distances: Sequence of scale-normalized distances d_i' (mean = 1.0).
    - diameter_ratios: Sequence of neighbor-to-center diameter ratios phi_i / phi_0.
    - feature_vector_d: Vector of angular structure scalar projections (Eq. 4).
    """

    center_crater: CraterFeature
    neighbors: List[CraterFeature] = field(default_factory=list)
    k: int = 0
    angular_structures: List[AngularStructure] = field(default_factory=list)
    azimuth_angles_deg: List[float] = field(default_factory=list)
    cyclic_interior_angles: List[float] = field(default_factory=list)
    normalized_distances: List[float] = field(default_factory=list)
    diameter_ratios: List[float] = field(default_factory=list)
    feature_vector_d: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center_crater": self.center_crater.to_dict(),
            "neighbors": [n.to_dict() for n in self.neighbors],
            "k": self.k,
            "num_angular_structures": len(self.angular_structures),
            "azimuth_angles_deg": self.azimuth_angles_deg,
            "cyclic_interior_angles": self.cyclic_interior_angles,
            "normalized_distances": self.normalized_distances,
            "diameter_ratios": self.diameter_ratios,
            "feature_vector_d": self.feature_vector_d.tolist(),
        }


def build_cnsf_descriptor(
    neighborhood: CraterNeighborhood,
) -> CNSFDescriptor:
    """Build a deterministic CNSF descriptor from a constructed crater neighborhood.

    Extracts:
    1. K(K-1)/2 clockwise angular structures (Section 2.2.3, Figure 4).
    2. Cyclic interior angle sequence (Eq. 7).
    3. Normalized distance sequence (Eq. 6).
    4. Diameter ratio sequence (Eq. 8).
    5. Feature vector D for distance measurement (Eq. 4 & 10).

    Args:
        neighborhood: Input CraterNeighborhood instance.

    Returns:
        Populated CNSFDescriptor instance.
    """
    cc = neighborhood.center_crater
    ncs = neighborhood.neighbors
    k = neighborhood.k_actual

    if k < 2:
        return CNSFDescriptor(
            center_crater=cc,
            neighbors=ncs,
            k=k,
            angular_structures=[],
            azimuth_angles_deg=neighborhood.azimuth_angles_deg,
            cyclic_interior_angles=neighborhood.interior_angles_deg,
            normalized_distances=neighborhood.normalized_distances,
            diameter_ratios=neighborhood.diameter_ratios,
            feature_vector_d=np.zeros(0, dtype=np.float64),
        )

    # 1. Generate all K(K-1)/2 pairwise angular structures formed by CC and two NCs
    angular_structures: List[AngularStructure] = []
    vector_d_elements: List[float] = []

    # Iterate over all ordered pairs (i, j) with i < j
    for i in range(k):
        nc1 = ncs[i]
        s1 = neighborhood.neighbor_distances_m[i]
        phi1 = nc1.diameter_m

        for j in range(i + 1, k):
            nc2 = ncs[j]
            s2 = neighborhood.neighbor_distances_m[j]
            phi2 = nc2.diameter_m

            # Compute clockwise angle beta from CC->NC1 to CC->NC2
            beta = compute_clockwise_angle_deg(
                x_c=cc.image_x,
                y_c=cc.image_y,
                x1=nc1.image_x,
                y1=nc1.image_y,
                x2=nc2.image_x,
                y2=nc2.image_y,
            )

            dist_ratio = float(s2 / max(s1, 1e-4))
            diam0 = max(cc.diameter_m, 1e-4)
            diam_ratio1 = float(phi1 / diam0)
            diam_ratio2 = float(phi2 / diam0)

            ang_struct = AngularStructure(
                cc_id=cc.detection_id,
                nc1_id=nc1.detection_id,
                nc2_id=nc2.detection_id,
                angle_deg=beta,
                dist1_m=s1,
                dist2_m=s2,
                diam0_m=cc.diameter_m,
                diam1_m=phi1,
                diam2_m=phi2,
                dist_ratio=dist_ratio,
                diam_ratio1=diam_ratio1,
                diam_ratio2=diam_ratio2,
            )
            angular_structures.append(ang_struct)

            # Equation 4 scalar term: |O A| |O B| cos(beta)
            # Using normalized distances to ensure uniform scale invariance:
            s1_norm = neighborhood.normalized_distances[i]
            s2_norm = neighborhood.normalized_distances[j]
            beta_rad = math.radians(beta)
            d_val = float(s1_norm * s2_norm * math.cos(beta_rad))
            vector_d_elements.append(d_val)

    feature_vec_d = np.array(vector_d_elements, dtype=np.float64)

    return CNSFDescriptor(
        center_crater=cc,
        neighbors=ncs,
        k=k,
        angular_structures=angular_structures,
        azimuth_angles_deg=neighborhood.azimuth_angles_deg,
        cyclic_interior_angles=neighborhood.interior_angles_deg,
        normalized_distances=neighborhood.normalized_distances,
        diameter_ratios=neighborhood.diameter_ratios,
        feature_vector_d=feature_vec_d,
    )
