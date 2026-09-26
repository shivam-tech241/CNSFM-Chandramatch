"""Geometric transformation models for cross-sensor validation (Chunk 9).

Provides:
- GeometricModel: Abstract base class for 2D spatial transformations.
- SimilarityModel: 4-DOF conformal transformation (isotropic scale, rotation, translation).
- AffineModel: 6-DOF linear transformation (anisotropic scale, shear, rotation, translation).
- Model parameters, residual evaluation, and transformation sanity checks.
"""

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, Optional, Tuple
import numpy as np


class GeometricModel(ABC):
    """Abstract base class for 2D geometric transformations."""

    @abstractmethod
    def fit(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> bool:
        """Estimate model parameters from point correspondences.

        Args:
            src_pts: (N, 2) array of source coordinates (meters or pixels).
            dst_pts: (N, 2) array of target coordinates.

        Returns:
            True if estimation succeeded, False if degenerate/underdetermined.
        """
        pass

    @abstractmethod
    def transform(self, pts: np.ndarray) -> np.ndarray:
        """Apply the forward transformation to input points.

        Args:
            pts: (N, 2) array of points to transform.

        Returns:
            (N, 2) array of transformed points.
        """
        pass

    @abstractmethod
    def min_samples(self) -> int:
        """Minimum number of point correspondences required for minimal solver."""
        pass

    def compute_residuals(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        """Compute Euclidean transfer error (residuals) for each point pair.

        Args:
            src_pts: (N, 2) source points.
            dst_pts: (N, 2) target points.

        Returns:
            (N,) 1D array of Euclidean distances ||dst_i - transform(src_i)|| in metric units.
        """
        if len(src_pts) == 0:
            return np.zeros(0, dtype=np.float64)
        pred_dst = self.transform(src_pts)
        diff = dst_pts - pred_dst
        return np.hypot(diff[:, 0], diff[:, 1])

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return a dictionary of model parameters and diagnostics."""
        pass

    @abstractmethod
    def is_plausible(
        self,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
        max_rotation_deg: float = 45.0,
        max_condition_number: float = 5.0,
    ) -> Tuple[bool, str]:
        """Sanity check estimated transformation against physical expectations.

        Returns:
            Tuple of (is_plausible, rejection_reason).
        """
        pass


class SimilarityModel(GeometricModel):
    """2D Similarity transformation (4 Degrees of Freedom: scale, rotation, translation).

    Model:
        x' = s * cos(theta) * x - s * sin(theta) * y + t_x
        y' = s * sin(theta) * x + s * cos(theta) * y + t_y

    Estimated via the closed-form Umeyama / Procrustes least-squares algorithm.
    """

    def __init__(self):
        self.scale: float = 1.0
        self.rotation_rad: float = 0.0
        self.translation: np.ndarray = np.zeros(2, dtype=np.float64)
        self.matrix_2x2: np.ndarray = np.eye(2, dtype=np.float64)
        self.is_fitted: bool = False

    def min_samples(self) -> int:
        return 2

    def fit(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> bool:
        src = np.asarray(src_pts, dtype=np.float64)
        dst = np.asarray(dst_pts, dtype=np.float64)

        n = len(src)
        if n < self.min_samples() or len(dst) != n:
            self.is_fitted = False
            return False

        # Center point clouds
        mu_src = np.mean(src, axis=0)
        mu_dst = np.mean(dst, axis=0)
        src_centered = src - mu_src
        dst_centered = dst - mu_dst

        # Source variance
        var_src = np.mean(np.sum(src_centered**2, axis=1))
        if var_src < 1e-12:  # Degenerate source points (co-located)
            self.is_fitted = False
            return False

        # Covariance matrix H = (dst^T * src) / n
        h = np.dot(dst_centered.T, src_centered) / n

        # SVD: H = U * S * V^T
        try:
            u, s, vt = np.linalg.svd(h)
        except np.linalg.LinAlgError:
            self.is_fitted = False
            return False

        # Reflection correction: det(R) must be +1
        d = np.linalg.det(u) * np.linalg.det(vt)
        diag = np.array([1.0, d], dtype=np.float64)
        r = np.dot(u, np.dot(np.diag(diag), vt))

        # Scale factor
        scale = float(np.sum(s * diag) / var_src)
        if scale <= 1e-7 or math.isnan(scale):
            self.is_fitted = False
            return False

        # Translation: t = mu_dst - s * R * mu_src
        t = mu_dst - scale * np.dot(r, mu_src)

        self.scale = scale
        self.matrix_2x2 = scale * r
        self.rotation_rad = math.atan2(r[1, 0], r[0, 0])
        self.translation = t
        self.is_fitted = True
        return True

    def transform(self, pts: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("SimilarityModel must be fitted before transform can be called.")
        arr = np.asarray(pts, dtype=np.float64)
        return np.dot(arr, self.matrix_2x2.T) + self.translation

    @property
    def rotation_deg(self) -> float:
        return math.degrees(self.rotation_rad)

    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": "Similarity",
            "is_fitted": self.is_fitted,
            "scale": round(float(self.scale), 6),
            "rotation_deg": round(float(self.rotation_deg), 4),
            "translation_x_m": round(float(self.translation[0]), 3),
            "translation_y_m": round(float(self.translation[1]), 3),
            "determinant": round(float(np.linalg.det(self.matrix_2x2)), 6),
            "matrix_2x2": self.matrix_2x2.tolist(),
        }

    def is_plausible(
        self,
        min_scale: float = 0.7,
        max_scale: float = 1.3,
        max_rotation_deg: float = 30.0,
        max_condition_number: float = 3.0,
    ) -> Tuple[bool, str]:
        if not self.is_fitted:
            return False, "Model is not fitted."
        if not (min_scale <= self.scale <= max_scale):
            return False, f"Scale {self.scale:.4f} outside plausible range [{min_scale}, {max_scale}]."
        if abs(self.rotation_deg) > max_rotation_deg:
            return False, f"Rotation {self.rotation_deg:.2f} deg exceeds threshold {max_rotation_deg} deg."
        return True, "Plausible similarity model."


class AffineModel(GeometricModel):
    """2D Affine transformation (6 Degrees of Freedom: linear mapping + translation).

    Model:
        [x', y']^T = A * [x, y]^T + t

    Estimated via ordinary least-squares on point correspondences.
    """

    def __init__(self):
        self.matrix_2x2: np.ndarray = np.eye(2, dtype=np.float64)
        self.translation: np.ndarray = np.zeros(2, dtype=np.float64)
        self.singular_values: np.ndarray = np.ones(2, dtype=np.float64)
        self.condition_number: float = 1.0
        self.determinant: float = 1.0
        self.is_fitted: bool = False

    def min_samples(self) -> int:
        return 3

    def fit(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> bool:
        src = np.asarray(src_pts, dtype=np.float64)
        dst = np.asarray(dst_pts, dtype=np.float64)

        n = len(src)
        if n < self.min_samples() or len(dst) != n:
            self.is_fitted = False
            return False

        # Construct design matrix: [x, y, 1]
        x_des = np.column_stack([src, np.ones(n, dtype=np.float64)])

        # Solve for parameters: X * M = D => M = (X^T X)^-1 X^T D
        try:
            m, residuals, rank, s_vals = np.linalg.lstsq(x_des, dst, rcond=None)
        except np.linalg.LinAlgError:
            self.is_fitted = False
            return False

        if rank < 3:  # Degenerate collinear points
            self.is_fitted = False
            return False

        a = m[:2, :].T  # shape (2, 2)
        t = m[2, :]     # shape (2,)

        # Compute singular values and condition number of A
        u, s, vt = np.linalg.svd(a)
        det_a = float(np.linalg.det(a))
        cond = float(s[0] / max(s[1], 1e-12))

        self.matrix_2x2 = a
        self.translation = t
        self.singular_values = s
        self.condition_number = cond
        self.determinant = det_a
        self.is_fitted = True
        return True

    def transform(self, pts: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("AffineModel must be fitted before transform can be called.")
        arr = np.asarray(pts, dtype=np.float64)
        return np.dot(arr, self.matrix_2x2.T) + self.translation

    @property
    def scale_anisotropy(self) -> float:
        """Ratio of maximum to minimum singular value (1.0 for isotropic)."""
        return float(self.singular_values[0] / max(self.singular_values[1], 1e-12))

    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": "Affine",
            "is_fitted": self.is_fitted,
            "matrix_2x2": self.matrix_2x2.tolist(),
            "translation_x_m": round(float(self.translation[0]), 3),
            "translation_y_m": round(float(self.translation[1]), 3),
            "determinant": round(float(self.determinant), 6),
            "singular_values": [round(float(s), 6) for s in self.singular_values],
            "condition_number": round(float(self.condition_number), 4),
            "scale_anisotropy": round(float(self.scale_anisotropy), 4),
        }

    def is_plausible(
        self,
        min_det: float = 0.4,
        max_det: float = 2.5,
        max_condition_number: float = 3.0,
        max_rotation_deg: float = 35.0,
    ) -> Tuple[bool, str]:
        if not self.is_fitted:
            return False, "Model is not fitted."
        if self.determinant <= 0.0:
            return False, f"Non-positive determinant {self.determinant:.4f} (reflection/inversion)."
        if not (min_det <= self.determinant <= max_det):
            return False, f"Determinant {self.determinant:.4f} outside bounds [{min_det}, {max_det}]."
        if self.condition_number > max_condition_number:
            return False, f"Condition number {self.condition_number:.2f} exceeds limit {max_condition_number}."
        return True, "Plausible affine model."
