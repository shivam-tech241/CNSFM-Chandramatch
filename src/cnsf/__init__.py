"""Crater Neighborhood Structure Feature (CNSF) package.

Chunk 8 Implementation of Xie et al. (Remote Sensing 2025, 17, 2302):
- Explicit crater feature model and coordinate handling (geometry.py).
- K-nearest neighbor graph construction and cyclic topological features (neighborhood.py).
- Invariant CNSF descriptor construction (descriptor.py).
- Frequency voting, distance calculation, and NNDR matching (matcher.py).
"""

from src.cnsf.geometry import (
    LUNAR_RADIUS_METERS,
    AngularStructure,
    CraterFeature,
    compute_clockwise_angle_deg,
    compute_euclidean_distance_m,
    compute_selenographic_distance_m,
)
from src.cnsf.neighborhood import (
    CraterNeighborhood,
    build_crater_neighborhood,
)
from src.cnsf.descriptor import (
    CNSFDescriptor,
    build_cnsf_descriptor,
)
from src.cnsf.matcher import (
    CNSFMatch,
    CNSFMatcher,
)

__all__ = [
    "CraterFeature",
    "AngularStructure",
    "CraterNeighborhood",
    "CNSFDescriptor",
    "CNSFMatch",
    "CNSFMatcher",
    "LUNAR_RADIUS_METERS",
    "compute_clockwise_angle_deg",
    "compute_euclidean_distance_m",
    "compute_selenographic_distance_m",
    "build_crater_neighborhood",
    "build_cnsf_descriptor",
]
