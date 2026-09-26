"""CNSF structural matching engine with frequency voting and NNDR filtering.

Chunk 8 Implementation:
- Pairwise angular structure similarity verification under error bounds (Equations 7 & 8).
- Frequency voting to eliminate non-corresponding Nearby Craters (Section 2.2.3).
- Verification of minimum corresponding neighbors (xi >= xi_min = 3).
- CNSF distance measurement via cosine metric (Equation 10).
- Nearest Neighbor Distance Ratio (NNDR) matching and absolute distance cutoff (Section 2.2.4).
- Spatially constrained and unconstrained candidate search modes.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from src.cnsf.descriptor import AngularStructure, CNSFDescriptor
from src.cnsf.geometry import CraterFeature, compute_euclidean_distance_m


@dataclass
class CNSFMatch:
    """Represents a matched crater correspondence produced by CNSF structural matching.

    Attributes:
    - source_cc_id: Central crater ID from source image (e.g. TMC-2).
    - target_cc_id: Central crater ID from target image (e.g. OHRC coarse).
    - source_x: Center pixel x coordinate in source image.
    - source_y: Center pixel y coordinate in source image.
    - target_x: Center pixel x coordinate in target image.
    - target_y: Center pixel y coordinate in target image.
    - source_diameter_m: Physical diameter of source CC in meters.
    - target_diameter_m: Physical diameter of target CC in meters.
    - diameter_ratio: min(D_src, D_tgt) / max(D_src, D_tgt) in (0, 1].
    - center_distance_m: Physical center displacement in meters.
    - cnsf_distance: Computed CNSF cosine distance d_CNSF in [0.0, 2.0] (Eq. 10).
    - nndr_ratio: Ratio of closest to second-closest CNSF distance (d1 / d2).
    - num_corresponding_ncs: Number of mutually supporting corresponding NCs (xi).
    - k: Neighborhood size evaluated.
    - source_sensor: Source sensor name.
    - target_sensor: Target sensor name.
    - corresponding_nc_pairs: List of paired NC IDs [(src_nc_id, tgt_nc_id), ...].
    """

    source_cc_id: Union[int, str]
    target_cc_id: Union[int, str]
    source_x: float
    source_y: float
    target_x: float
    target_y: float
    source_diameter_m: float
    target_diameter_m: float
    diameter_ratio: float
    center_distance_m: float
    cnsf_distance: float
    nndr_ratio: float
    num_corresponding_ncs: int
    k: int
    source_sensor: str = "TMC-2"
    target_sensor: str = "OHRC"
    corresponding_nc_pairs: List[Tuple[Any, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CNSFMatcher:
    """Matches Crater Neighborhood Structure Feature (CNSF) descriptors.

    Implements the exact methodology of Sections 2.2.3, 2.2.4, and 3.3.3 of Xie et al. (2025).
    """

    def __init__(
        self,
        delta_px: float = 3.0,
        eta_percent: float = 0.25,
        xi_min: int = 3,
        nndr_threshold: float = 0.1,
        distance_cutoff: float = 0.1,
        coarse_gsd_m: float = 5.40,
        one_third_rule: bool = True,
    ):
        """Initialize the CNSF matcher with paper-defined parameters.

        Args:
            delta_px: Crater center localization error in pixels (paper default: 3.0).
            eta_percent: Crater diameter detection error fraction (paper default: 0.25 = 25%).
            xi_min: Minimum number of corresponding NCs required (paper default: 3).
            nndr_threshold: Nearest Neighbor Distance Ratio threshold (paper default: 0.1).
            distance_cutoff: Maximum allowable CNSF distance for a match (paper default: 0.1).
            coarse_gsd_m: GSD in meters/pixel for physical error scaling (default: 5.40).
            one_third_rule: Apply 1/3 scaling to analytical error bounds per Section 2.2.3.
        """
        self.delta_px = delta_px
        self.eta_percent = eta_percent
        self.xi_min = xi_min
        self.nndr_threshold = nndr_threshold
        self.distance_cutoff = distance_cutoff
        self.coarse_gsd_m = coarse_gsd_m
        self.one_third_rule = one_third_rule
        # Delta in physical meters
        self.delta_m = delta_px * coarse_gsd_m

    def check_angular_structure_similarity(
        self,
        s_a: AngularStructure,
        s_b: AngularStructure,
    ) -> Tuple[bool, int]:
        """Check if two angular structures are similar under Equation 7 and Equation 8.

        Args:
            s_a: AngularStructure from descriptor A.
            s_b: AngularStructure from descriptor B.

        Returns:
            Tuple of:
                - is_similar: bool indicating whether similarity conditions are met.
                - orientation_case: 1 for Case 1 (same direction), 2 for Case 2 (inverted), 0 for none.
        """
        delta = self.delta_m
        eta = self.eta_percent
        factor = (1.0 / 3.0) if self.one_third_rule else 1.0

        # Maximum allowable tolerances from Equation 8
        # Tolerance for angle beta (in degrees)
        def _safe_asin_deg(arg: float) -> float:
            clamped = min(1.0, max(-1.0, arg))
            return math.degrees(math.asin(clamped))

        t_beta = factor * (
            _safe_asin_deg(delta / max(s_a.dist1_m, 1e-3))
            + _safe_asin_deg(delta / max(s_a.dist2_m, 1e-3))
            + _safe_asin_deg(delta / max(s_b.dist1_m, 1e-3))
            + _safe_asin_deg(delta / max(s_b.dist2_m, 1e-3))
        )
        t_beta = max(5.0, t_beta)  # numerical safety floor 5 degrees

        # Tolerance for side length ratio R_S
        denom_a = max(s_a.dist1_m - delta, 1.0)
        denom_b = max(s_b.dist1_m - delta, 1.0)
        t_rs = factor * (
            (delta * (s_a.dist2_m + s_a.dist1_m)) / (s_a.dist1_m * denom_a)
            + (delta * (s_b.dist2_m + s_b.dist1_m)) / (s_b.dist1_m * denom_b)
        )
        t_rs = max(0.1, t_rs)

        # Tolerance for crater diameter ratios R_phi1 and R_phi2
        denom_eta = max(1.0 - eta, 0.1)
        t_rphi1 = factor * (
            (2.0 * s_a.diam1_m * eta) / (s_a.diam0_m * denom_eta)
            + (2.0 * s_b.diam1_m * eta) / (s_b.diam0_m * denom_eta)
        )
        t_rphi2 = factor * (
            (2.0 * s_a.diam2_m * eta) / (s_a.diam0_m * denom_eta)
            + (2.0 * s_b.diam2_m * eta) / (s_b.diam0_m * denom_eta)
        )
        t_rphi1 = max(0.15, t_rphi1)
        t_rphi2 = max(0.15, t_rphi2)

        # -----------------------------
        # Case 1: Same Orientation
        # beta == beta', S2/S1 == S2'/S1', phi1/phi0 == phi1'/phi0', phi2/phi0 == phi2'/phi0'
        # -----------------------------
        res_beta1 = abs(s_a.angle_deg - s_b.angle_deg)
        if res_beta1 > 180.0:
            res_beta1 = 360.0 - res_beta1

        res_rs1 = abs(s_a.dist_ratio - s_b.dist_ratio)
        res_rphi1_1 = abs(s_a.diam_ratio1 - s_b.diam_ratio1)
        res_rphi2_1 = abs(s_a.diam_ratio2 - s_b.diam_ratio2)

        if (
            res_beta1 <= t_beta
            and res_rs1 <= t_rs
            and res_rphi1_1 <= t_rphi1
            and res_rphi2_1 <= t_rphi2
        ):
            return True, 1

        # -----------------------------
        # Case 2: Inverted Starting Edge
        # beta == 360 - beta', S2/S1 == S1'/S2', phi1/phi0 == phi2'/phi0', phi2/phi0 == phi1'/phi0'
        # -----------------------------
        res_beta2 = abs((s_a.angle_deg + s_b.angle_deg) - 360.0)
        inv_ratio_b = 1.0 / max(s_b.dist_ratio, 1e-4)
        res_rs2 = abs(s_a.dist_ratio - inv_ratio_b)
        res_rphi1_2 = abs(s_a.diam_ratio1 - s_b.diam_ratio2)
        res_rphi2_2 = abs(s_a.diam_ratio2 - s_b.diam_ratio1)

        if (
            res_beta2 <= t_beta
            and res_rs2 <= t_rs
            and res_rphi1_2 <= t_rphi1
            and res_rphi2_2 <= t_rphi2
        ):
            return True, 2

        return False, 0

    def _extract_structure_terms(self, desc: CNSFDescriptor) -> List[Tuple]:
        """Precompute per-structure tolerance terms to avoid redundant trig calls in matching."""
        delta = self.delta_m
        eta = self.eta_percent
        denom_eta = max(1.0 - eta, 0.1)

        def _safe_asin_deg(arg: float) -> float:
            clamped = min(1.0, max(-1.0, arg))
            return math.degrees(math.asin(clamped))

        terms = []
        for s in desc.angular_structures:
            t_b = _safe_asin_deg(delta / max(s.dist1_m, 1e-3)) + _safe_asin_deg(delta / max(s.dist2_m, 1e-3))
            denom = max(s.dist1_m - delta, 1.0)
            t_rs = (delta * (s.dist2_m + s.dist1_m)) / (s.dist1_m * denom)
            t_phi1 = (2.0 * s.diam1_m * eta) / (s.diam0_m * denom_eta)
            t_phi2 = (2.0 * s.diam2_m * eta) / (s.diam0_m * denom_eta)
            terms.append((s, s.angle_deg, s.dist_ratio, s.diam_ratio1, s.diam_ratio2, t_b, t_rs, t_phi1, t_phi2))
        return terms

    def compute_pairwise_cnsf_distance(
        self,
        desc_a: CNSFDescriptor,
        desc_b: CNSFDescriptor,
    ) -> Tuple[float, int, List[Tuple[Any, Any]]]:
        """Compute the CNSF distance between two descriptors using frequency voting and Equation 10.

        Args:
            desc_a: Source CNSF descriptor.
            desc_b: Target CNSF descriptor.

        Returns:
            Tuple of:
                - cnsf_distance: float in [0.0, 2.0] (or inf if rejected).
                - xi_actual: Number of mutually supporting corresponding NCs.
                - matched_nc_pairs: List of (id_a, id_b) corresponding NC pairs.
        """
        if desc_a.k < self.xi_min or desc_b.k < self.xi_min:
            return float("inf"), 0, []

        # Step 1: Compare all angular structures with cached precomputed terms
        terms_a = getattr(desc_a, "_cached_structure_terms", None)
        if terms_a is None or getattr(desc_a, "_cached_delta_m", None) != self.delta_m:
            terms_a = self._extract_structure_terms(desc_a)
            desc_a._cached_structure_terms = terms_a
            desc_a._cached_delta_m = self.delta_m

        terms_b = getattr(desc_b, "_cached_structure_terms", None)
        if terms_b is None or getattr(desc_b, "_cached_delta_m", None) != self.delta_m:
            terms_b = self._extract_structure_terms(desc_b)
            desc_b._cached_structure_terms = terms_b
            desc_b._cached_delta_m = self.delta_m

        factor = (1.0 / 3.0) if self.one_third_rule else 1.0
        votes: Counter = Counter()

        for sa, ang_a, rs_a, d1_a, d2_a, tb_a, trs_a, td1_a, td2_a in terms_a:
            for sb, ang_b, rs_b, d1_b, d2_b, tb_b, trs_b, td1_b, td2_b in terms_b:
                trs = max(0.1, factor * (trs_a + trs_b))
                diff_rs1 = abs(rs_a - rs_b)
                inv_rs_b = 1.0 / max(rs_b, 1e-4)
                diff_rs2 = abs(rs_a - inv_rs_b)
                if diff_rs1 > trs and diff_rs2 > trs:
                    continue

                tb = max(5.0, factor * (tb_a + tb_b))
                td1 = max(0.15, factor * (td1_a + td1_b))
                td2 = max(0.15, factor * (td2_a + td2_b))

                # Case 1: Same Orientation
                res_b1 = abs(ang_a - ang_b)
                if res_b1 > 180.0:
                    res_b1 = 360.0 - res_b1

                if (
                    res_b1 <= tb
                    and diff_rs1 <= trs
                    and abs(d1_a - d1_b) <= td1
                    and abs(d2_a - d2_b) <= td2
                ):
                    votes[(sa.nc1_id, sb.nc1_id)] += 1
                    votes[(sa.nc2_id, sb.nc2_id)] += 1
                    continue

                # Case 2: Inverted Starting Edge
                res_b2 = abs((ang_a + ang_b) - 360.0)
                if (
                    res_b2 <= tb
                    and diff_rs2 <= trs
                    and abs(d1_a - d2_b) <= td1
                    and abs(d2_a - d1_b) <= td2
                ):
                    votes[(sa.nc1_id, sb.nc2_id)] += 1
                    votes[(sa.nc2_id, sb.nc1_id)] += 1

        if not votes:
            return float("inf"), 0, []

        # Step 2: Frequency voting (Section 2.2.3)
        # Find mode of the frequency counts
        counts = list(votes.values())
        freq_counter = Counter(counts)
        mode_count, _ = freq_counter.most_common(1)[0]
        # In ideal case, each true corresponding NC appears in xi - 1 structures
        min_required_votes = max(1, mode_count)

        # Retain candidate pairs meeting mode threshold
        candidate_nc_pairs = [pair for pair, v in votes.items() if v >= min_required_votes]

        # Enforce 1-to-1 NC matching: pick pair with highest vote if collision occurs
        candidate_nc_pairs.sort(key=lambda p: votes[p], reverse=True)
        used_a: Set[Any] = set()
        used_b: Set[Any] = set()
        retained_pairs: List[Tuple[Any, Any]] = []

        for id_a, id_b in candidate_nc_pairs:
            if id_a not in used_a and id_b not in used_b:
                used_a.add(id_a)
                used_b.add(id_b)
                retained_pairs.append((id_a, id_b))

        xi_actual = len(retained_pairs)

        # Step 3: Check minimum NC threshold xi >= xi_min (Section 2.2.4 & 3.3.3)
        if xi_actual < self.xi_min:
            return float("inf"), xi_actual, retained_pairs

        # Step 4: CNSF distance measurement via cyclic cosine distance (Equation 10)
        # Construct cyclic feature vectors D and D' using retained corresponding NCs
        id_to_idx_a = {n.detection_id: idx for idx, n in enumerate(desc_a.neighbors)}
        id_to_idx_b = {n.detection_id: idx for idx, n in enumerate(desc_b.neighbors)}

        # Sort retained pairs by neighbor index in desc_a
        retained_pairs.sort(key=lambda p: id_to_idx_a.get(p[0], 0))

        vec_d: List[float] = []
        vec_d_prime: List[float] = []

        # Cyclic adjacent angular terms among corresponding NCs (dimension = xi_actual)
        for m in range(xi_actual):
            p1_a, p1_b = retained_pairs[m]
            p2_a, p2_b = retained_pairs[(m + 1) % xi_actual]

            idx1_a = id_to_idx_a[p1_a]
            idx2_a = id_to_idx_a[p2_a]
            idx1_b = id_to_idx_b[p1_b]
            idx2_b = id_to_idx_b[p2_b]

            s1_a = desc_a.normalized_distances[idx1_a]
            s2_a = desc_a.normalized_distances[idx2_a]
            s1_b = desc_b.normalized_distances[idx1_b]
            s2_b = desc_b.normalized_distances[idx2_b]

            # Cyclic clockwise angular difference between consecutive neighbors
            diff_a = (desc_a.azimuth_angles_deg[idx2_a] - desc_a.azimuth_angles_deg[idx1_a]) % 360.0
            diff_b = (desc_b.azimuth_angles_deg[idx2_b] - desc_b.azimuth_angles_deg[idx1_b]) % 360.0

            ang_a = math.radians(diff_a)
            ang_b = math.radians(diff_b)

            term_a = s1_a * s2_a * math.cos(ang_a)
            term_b = s1_b * s2_b * math.cos(ang_b)

            vec_d.append(term_a)
            vec_d_prime.append(term_b)

        arr_d = np.array(vec_d, dtype=np.float64)
        arr_dp = np.array(vec_d_prime, dtype=np.float64)

        norm_d = np.linalg.norm(arr_d)
        norm_dp = np.linalg.norm(arr_dp)

        if norm_d < 1e-7 or norm_dp < 1e-7:
            return 0.0, xi_actual, retained_pairs

        # Invariance to starting edge: test all xi_actual cyclic shifts of arr_dp
        best_cos_sim = -1.0
        for shift in range(xi_actual):
            sh_dp = np.roll(arr_dp, shift)
            sim = float(np.dot(arr_d, sh_dp) / (norm_d * norm_dp))
            if sim > best_cos_sim:
                best_cos_sim = sim

        # Also test reversed cyclic direction (inverted orientation case)
        arr_dp_rev = np.flip(arr_dp)
        for shift in range(xi_actual):
            sh_dp = np.roll(arr_dp_rev, shift)
            sim = float(np.dot(arr_d, sh_dp) / (norm_d * norm_dp))
            if sim > best_cos_sim:
                best_cos_sim = sim

        best_cos_sim = min(1.0, max(-1.0, best_cos_sim))
        d_cnsf = float(max(0.0, 1.0 - best_cos_sim))

        return d_cnsf, xi_actual, retained_pairs

    def match_descriptors(
        self,
        source_descriptors: List[CNSFDescriptor],
        target_descriptors: List[CNSFDescriptor],
        max_candidate_dist_m: Optional[float] = None,
        use_spatial_constraint: bool = False,
    ) -> List[CNSFMatch]:
        """Match source descriptors against target descriptors with frequency voting and NNDR.

        Args:
            source_descriptors: List of source CNSF descriptors (e.g. TMC-2).
            target_descriptors: List of target CNSF descriptors (e.g. coarse OHRC).
            max_candidate_dist_m: Optional spatial search radius in meters for candidate pairs.
            use_spatial_constraint: If True, uses max_candidate_dist_m to filter candidates.

        Returns:
            List of filtered CNSFMatch objects.
        """
        matches: List[CNSFMatch] = []

        if not source_descriptors or not target_descriptors:
            return matches

        for desc_a in source_descriptors:
            cc_a = desc_a.center_crater
            candidate_distances: List[Tuple[float, int, List[Tuple[Any, Any]], CNSFDescriptor]] = []

            for desc_b in target_descriptors:
                cc_b = desc_b.center_crater

                # Optional spatial bounding constraint
                if use_spatial_constraint and max_candidate_dist_m is not None:
                    dist_ground = compute_euclidean_distance_m(cc_a, cc_b)
                    if dist_ground > max_candidate_dist_m:
                        continue

                d_val, xi_act, nc_pairs = self.compute_pairwise_cnsf_distance(desc_a, desc_b)
                if not math.isinf(d_val):
                    candidate_distances.append((d_val, xi_act, nc_pairs, desc_b))

            if not candidate_distances:
                continue

            # Sort candidate matches by CNSF distance ascending
            candidate_distances.sort(key=lambda item: item[0])

            best_d, best_xi, best_pairs, best_desc_b = candidate_distances[0]

            # Compute NNDR ratio
            if len(candidate_distances) >= 2:
                second_best_d = candidate_distances[1][0]
                nndr_ratio = float(best_d / max(second_best_d, 1e-6))
            else:
                second_best_d = 1.0
                nndr_ratio = 0.0

            # Section 2.2.4 matching criteria:
            # 1. d_closest <= distance_cutoff
            # 2. d_closest / d_second_closest <= nndr_threshold
            passes_distance = bool(best_d <= self.distance_cutoff)
            passes_nndr = bool(nndr_ratio <= self.nndr_threshold or len(candidate_distances) == 1)

            if passes_distance and passes_nndr:
                best_cc_b = best_desc_b.center_crater
                d_min = min(cc_a.diameter_m, best_cc_b.diameter_m)
                d_max = max(cc_a.diameter_m, best_cc_b.diameter_m)
                diam_ratio = float(d_min / max(d_max, 1e-4))
                center_disp_m = compute_euclidean_distance_m(cc_a, best_cc_b)

                match_record = CNSFMatch(
                    source_cc_id=cc_a.detection_id,
                    target_cc_id=best_cc_b.detection_id,
                    source_x=cc_a.image_x,
                    source_y=cc_a.image_y,
                    target_x=best_cc_b.image_x,
                    target_y=best_cc_b.image_y,
                    source_diameter_m=cc_a.diameter_m,
                    target_diameter_m=best_cc_b.diameter_m,
                    diameter_ratio=diam_ratio,
                    center_distance_m=center_disp_m,
                    cnsf_distance=round(best_d, 5),
                    nndr_ratio=round(nndr_ratio, 5),
                    num_corresponding_ncs=best_xi,
                    k=desc_a.k,
                    source_sensor=cc_a.sensor,
                    target_sensor=best_cc_b.sensor,
                    corresponding_nc_pairs=best_pairs,
                )
                matches.append(match_record)

        return matches
