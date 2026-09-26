"""Ground-coordinate grid parser, cell-wise bilinear interpolation, and overlap extraction.

CHUNK 4: Exploits official ISRO ground coordinate grid files (_g_grd_d18.csv)
to establish high-precision, bidirectional mapping between geographic coordinates
(Longitude, Latitude) and image pixel coordinates (Pixel, Scan).
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Mean lunar radius in meters (IAU standard)
MOON_RADIUS_METERS: float = 1737400.0


@dataclass(frozen=True)
class GridSummary:
    """Statistical and structural summary of an ISRO ground coordinate grid."""

    sensor_name: str
    csv_path: str
    num_points: int
    n_scans: int
    n_pixels: int
    scan_min: int
    scan_max: int
    pixel_min: int
    pixel_max: int
    scan_step_regular: int
    pixel_step_regular: int
    is_regular_grid: bool
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    dlat_dscan_min: float
    dlat_dscan_max: float
    dlon_dpixel_min: float
    dlon_dpixel_max: float
    is_monotonic_lat: bool
    is_monotonic_lon: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationMetrics:
    """Evaluation metrics for grid interpolation accuracy and reversibility."""

    sensor_name: str
    strategy: str
    num_samples: int
    pixel_error_mean: float
    pixel_error_median: float
    pixel_error_max: float
    pixel_error_p95: float
    pixel_error_std: float
    scan_error_mean: float
    scan_error_median: float
    scan_error_max: float
    scan_error_p95: float
    scan_error_std: float
    geo_dist_m_mean: Optional[float] = None
    geo_dist_m_median: Optional[float] = None
    geo_dist_m_max: Optional[float] = None
    geo_dist_m_p95: Optional[float] = None
    geo_dist_m_std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OverlapResult:
    """Exact calculated overlap bounds between OHRC footprint and TMC-2 sensor."""

    ohrc_geo_bounds: Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    ohrc_image_shape: Tuple[int, int]  # (lines, samples)
    tmc2_image_shape: Tuple[int, int]  # (lines, samples)
    tmc2_scan_bounds_continuous: Tuple[float, float]  # (min_scan, max_scan)
    tmc2_pixel_bounds_continuous: Tuple[float, float]  # (min_pixel, max_pixel)
    tmc2_overlap_scan_start: int
    tmc2_overlap_scan_end: int
    tmc2_overlap_pixel_start: int
    tmc2_overlap_pixel_end: int
    num_tie_points_in_overlap: int
    mapped_boundary_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ohrc_geo_bounds": list(self.ohrc_geo_bounds),
            "ohrc_image_shape": list(self.ohrc_image_shape),
            "tmc2_image_shape": list(self.tmc2_image_shape),
            "tmc2_scan_bounds_continuous": list(self.tmc2_scan_bounds_continuous),
            "tmc2_pixel_bounds_continuous": list(self.tmc2_pixel_bounds_continuous),
            "tmc2_overlap_scan_start": self.tmc2_overlap_scan_start,
            "tmc2_overlap_scan_end": self.tmc2_overlap_scan_end,
            "tmc2_overlap_pixel_start": self.tmc2_overlap_pixel_start,
            "tmc2_overlap_pixel_end": self.tmc2_overlap_pixel_end,
            "num_tie_points_in_overlap": self.num_tie_points_in_overlap,
            "mapped_boundary_samples": self.mapped_boundary_samples,
        }


def _solve_inverse_bilinear_cell(
    p: np.ndarray,
    p00: np.ndarray,
    p10: np.ndarray,
    p01: np.ndarray,
    p11: np.ndarray,
    tol: float = 1e-11,
    max_iter: int = 15,
) -> np.ndarray:
    """Solve p = (1-u)(1-v)p00 + u(1-v)p10 + (1-u)v p01 + uv p11 for (u, v) using Newton-Raphson.

    Args:
        p: Target geographic coordinate [lon, lat].
        p00: Corner (0, 0) coordinate [lon, lat].
        p10: Corner (1, 0) coordinate [lon, lat] (increment in pixel/column).
        p01: Corner (0, 1) coordinate [lon, lat] (increment in scan/row).
        p11: Corner (1, 1) coordinate [lon, lat] (increment in both).
        tol: Convergence tolerance for residual norm.
        max_iter: Maximum iterations (typically converges in 2-3 iterations).

    Returns:
        np.ndarray [u, v] coordinates in unit square [0, 1]^2.
    """
    uv = np.array([0.5, 0.5], dtype=np.float64)
    for _ in range(max_iter):
        u, v = uv[0], uv[1]
        w00 = (1.0 - u) * (1.0 - v)
        w10 = u * (1.0 - v)
        w01 = (1.0 - u) * v
        w11 = u * v
        F = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11 - p
        if np.dot(F, F) < tol * tol:
            break
        dF_du = (1.0 - v) * (p10 - p00) + v * (p11 - p01)
        dF_dv = (1.0 - u) * (p01 - p00) + u * (p11 - p10)
        det = dF_du[0] * dF_dv[1] - dF_du[1] * dF_dv[0]
        if abs(det) < 1e-15:
            break
        du = (dF_dv[1] * F[0] - dF_dv[0] * F[1]) / det
        dv = (-dF_du[1] * F[0] + dF_du[0] * F[1]) / det
        uv[0] -= du
        uv[1] -= dv
        if du * du + dv * dv < tol * tol:
            break
    return uv


class GroundGrid:
    """ISRO Ground Coordinate Grid Abstraction.

    Loads the official sparse grid of tie points relating image pixel/scan coordinates
    to lunar selenographic longitude/latitude, verifies grid regularity and monotonicity,
    and performs exact cell-wise bilinear and inverse-bilinear interpolations.
    """

    def __init__(self, csv_path: Path | str, sensor_name: str = "GENERIC"):
        self.csv_path = Path(csv_path)
        self.sensor_name = sensor_name.upper()

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"ISRO Ground Grid CSV not found: {self.csv_path}")

        # Load CSV
        df = pd.read_csv(self.csv_path)
        required_cols = {"Longitude", "Latitude", "Pixel", "Scan"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Ground Grid CSV {self.csv_path} is missing required columns: {missing}")

        self.num_points: int = len(df)
        self.u_scans: np.ndarray = np.unique(df["Scan"].values)
        self.u_pixels: np.ndarray = np.unique(df["Pixel"].values)
        self.n_scans: int = len(self.u_scans)
        self.n_pixels: int = len(self.u_pixels)

        if self.n_scans * self.n_pixels != self.num_points:
            raise ValueError(
                f"Ground Grid has non-rectangular point count: "
                f"{self.n_scans} scans x {self.n_pixels} pixels != {self.num_points} rows"
            )

        # Reshape into 2D grids (n_scans, n_pixels)
        self.lons_grid: np.ndarray = df["Longitude"].values.reshape(self.n_scans, self.n_pixels).astype(np.float64)
        self.lats_grid: np.ndarray = df["Latitude"].values.reshape(self.n_scans, self.n_pixels).astype(np.float64)

        # Coordinate bounds
        self.scan_min: int = int(self.u_scans[0])
        self.scan_max: int = int(self.u_scans[-1])
        self.pixel_min: int = int(self.u_pixels[0])
        self.pixel_max: int = int(self.u_pixels[-1])

        self.lon_min: float = float(self.lons_grid.min())
        self.lon_max: float = float(self.lons_grid.max())
        self.lat_min: float = float(self.lats_grid.min())
        self.lat_max: float = float(self.lats_grid.max())

        # Step sizes and regularity
        scan_diffs = np.diff(self.u_scans)
        pixel_diffs = np.diff(self.u_pixels)
        self.scan_step_regular: int = int(scan_diffs[0])
        self.pixel_step_regular: int = int(pixel_diffs[0])

        # Regular except possibly the final boundary step
        self.is_regular_grid: bool = bool(
            np.all(scan_diffs[:-1] == self.scan_step_regular)
            and np.all(pixel_diffs[:-1] == self.pixel_step_regular)
            and scan_diffs[-1] <= self.scan_step_regular
            and pixel_diffs[-1] <= self.pixel_step_regular
        )

        # Precompute row-level bounds for fast binary-search spatial indexing
        # Row min and max latitudes across pixel columns
        self._row_lat_min: np.ndarray = self.lats_grid.min(axis=1)
        self._row_lat_max: np.ndarray = self.lats_grid.max(axis=1)
        self._row_lon_min: np.ndarray = self.lons_grid.min(axis=1)
        self._row_lon_max: np.ndarray = self.lons_grid.max(axis=1)

        # Monotonicity check
        # Latitude strictly decreases with scan index (North to South orbital motion)
        self.is_monotonic_lat: bool = bool(
            np.all(np.diff(self._row_lat_min) < 0) and np.all(np.diff(self._row_lat_max) < 0)
        )
        # Check pixel longitude monotonicity
        dlon_dpixel = np.diff(self.lons_grid, axis=1)
        self.is_monotonic_lon: bool = bool(np.all(dlon_dpixel > 0) or np.all(dlon_dpixel < 0))

        # Check derivatives
        dlat_dscan = np.diff(self.lats_grid, axis=0)
        self._dlat_dscan_min = float(dlat_dscan.min())
        self._dlat_dscan_max = float(dlat_dscan.max())
        self._dlon_dpixel_min = float(dlon_dpixel.min())
        self._dlon_dpixel_max = float(dlon_dpixel.max())

    def get_summary(self) -> GridSummary:
        """Return comprehensive metadata summary of this ground grid."""
        return GridSummary(
            sensor_name=self.sensor_name,
            csv_path=str(self.csv_path),
            num_points=self.num_points,
            n_scans=self.n_scans,
            n_pixels=self.n_pixels,
            scan_min=self.scan_min,
            scan_max=self.scan_max,
            pixel_min=self.pixel_min,
            pixel_max=self.pixel_max,
            scan_step_regular=self.scan_step_regular,
            pixel_step_regular=self.pixel_step_regular,
            is_regular_grid=self.is_regular_grid,
            lon_min=self.lon_min,
            lon_max=self.lon_max,
            lat_min=self.lat_min,
            lat_max=self.lat_max,
            dlat_dscan_min=self._dlat_dscan_min,
            dlat_dscan_max=self._dlat_dscan_max,
            dlon_dpixel_min=self._dlon_dpixel_min,
            dlon_dpixel_max=self._dlon_dpixel_max,
            is_monotonic_lat=self.is_monotonic_lat,
            is_monotonic_lon=self.is_monotonic_lon,
        )

    def contains_pixel(self, pixel: float, scan: float) -> bool:
        """Check whether image coordinates fall within the grid boundaries."""
        return bool(
            self.pixel_min <= pixel <= self.pixel_max
            and self.scan_min <= scan <= self.scan_max
        )

    def contains_geo(self, lon: float, lat: float) -> bool:
        """Check whether geographic coordinates fall within the grid geographic bounding box."""
        return bool(
            self.lon_min <= lon <= self.lon_max
            and self.lat_min <= lat <= self.lat_max
        )

    def pixel_to_geo(
        self,
        pixel: Union[float, int, np.ndarray, List[float]],
        scan: Union[float, int, np.ndarray, List[float]],
        check_bounds: bool = True,
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Map image coordinates (pixel, scan) to selenographic coordinates (longitude, latitude).

        Uses cell-wise bilinear interpolation within the sparse rectilinear tie-point grid.

        Args:
            pixel: Image sample coordinate (0-indexed).
            scan: Image line coordinate (0-indexed).
            check_bounds: If True, raises ValueError if coordinate is outside grid bounds.
                          If False, returns NaN for out-of-bounds queries.

        Returns:
            Tuple of (longitude, latitude) in degrees.
        """
        is_scalar = np.isscalar(pixel) and np.isscalar(scan)
        px_arr = np.asarray(pixel, dtype=np.float64)
        sc_arr = np.asarray(scan, dtype=np.float64)

        if px_arr.shape != sc_arr.shape:
            raise ValueError(f"Pixel and Scan arrays must have the same shape: {px_arr.shape} vs {sc_arr.shape}")

        orig_shape = px_arr.shape
        px_flat = px_arr.ravel()
        sc_flat = sc_arr.ravel()
        n = len(px_flat)

        out_lon = np.full(n, np.nan, dtype=np.float64)
        out_lat = np.full(n, np.nan, dtype=np.float64)

        # Check validity
        valid_mask = (
            (px_flat >= self.pixel_min)
            & (px_flat <= self.pixel_max)
            & (sc_flat >= self.scan_min)
            & (sc_flat <= self.scan_max)
        )

        if check_bounds and not np.all(valid_mask):
            invalid_idx = np.where(~valid_mask)[0][0]
            raise ValueError(
                f"Image coordinate (Pixel={px_flat[invalid_idx]}, Scan={sc_flat[invalid_idx]}) "
                f"is out of bounds for sensor {self.sensor_name} grid "
                f"[Pixel: {self.pixel_min}..{self.pixel_max}, Scan: {self.scan_min}..{self.scan_max}]."
            )

        if np.any(valid_mask):
            v_px = px_flat[valid_mask]
            v_sc = sc_flat[valid_mask]

            # Fast searchsorted to locate cell row i and column j
            i = np.searchsorted(self.u_scans, v_sc, side="right") - 1
            j = np.searchsorted(self.u_pixels, v_px, side="right") - 1

            # Clamp right/bottom boundary points
            i = np.clip(i, 0, self.n_scans - 2)
            j = np.clip(j, 0, self.n_pixels - 2)

            # Local normalized cell coordinates (u, v) in [0, 1]
            p0 = self.u_pixels[j]
            p1 = self.u_pixels[j + 1]
            s0 = self.u_scans[i]
            s1 = self.u_scans[i + 1]

            u = (v_px - p0) / (p1 - p0)
            v = (v_sc - s0) / (s1 - s0)

            # Grid corner values
            lon00 = self.lons_grid[i, j]
            lon10 = self.lons_grid[i, j + 1]
            lon01 = self.lons_grid[i + 1, j]
            lon11 = self.lons_grid[i + 1, j + 1]

            lat00 = self.lats_grid[i, j]
            lat10 = self.lats_grid[i, j + 1]
            lat01 = self.lats_grid[i + 1, j]
            lat11 = self.lats_grid[i + 1, j + 1]

            # Bilinear formula
            w00 = (1.0 - u) * (1.0 - v)
            w10 = u * (1.0 - v)
            w01 = (1.0 - u) * v
            w11 = u * v

            out_lon[valid_mask] = w00 * lon00 + w10 * lon10 + w01 * lon01 + w11 * lon11
            out_lat[valid_mask] = w00 * lat00 + w10 * lat10 + w01 * lat01 + w11 * lat11

        if is_scalar:
            return float(out_lon[0]), float(out_lat[0])
        return out_lon.reshape(orig_shape), out_lat.reshape(orig_shape)

    def geo_to_pixel(
        self,
        longitude: Union[float, int, np.ndarray, List[float]],
        latitude: Union[float, int, np.ndarray, List[float]],
        check_bounds: bool = True,
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Map selenographic coordinates (longitude, latitude) to image coordinates (pixel, scan).

        Locates the enclosing quad cell using monotonic spatial indexing and solves the
        inverse-bilinear mapping equation via 2D Newton-Raphson iteration.

        Args:
            longitude: Selenographic East longitude in degrees.
            latitude: Selenographic North latitude in degrees.
            check_bounds: If True, raises ValueError if coordinate is outside grid bounds.
                          If False, returns NaN for out-of-bounds queries.

        Returns:
            Tuple of (pixel, scan) image coordinates.
        """
        is_scalar = np.isscalar(longitude) and np.isscalar(latitude)
        lon_arr = np.asarray(longitude, dtype=np.float64)
        lat_arr = np.asarray(latitude, dtype=np.float64)

        if lon_arr.shape != lat_arr.shape:
            raise ValueError(f"Longitude and Latitude arrays must match shape: {lon_arr.shape} vs {lat_arr.shape}")

        orig_shape = lon_arr.shape
        lon_flat = lon_arr.ravel()
        lat_flat = lat_arr.ravel()
        n = len(lon_flat)

        out_px = np.full(n, np.nan, dtype=np.float64)
        out_sc = np.full(n, np.nan, dtype=np.float64)

        eps = 1e-5  # boundary inclusion tolerance

        for idx in range(n):
            target_lon = lon_flat[idx]
            target_lat = lat_flat[idx]

            # Fast bounding box rejection
            if not (self.lon_min - 0.05 <= target_lon <= self.lon_max + 0.05 and
                    self.lat_min - 0.05 <= target_lat <= self.lat_max + 0.05):
                if check_bounds:
                    raise ValueError(
                        f"Geographic coordinate (Lon={target_lon:.6f}, Lat={target_lat:.6f}) "
                        f"is outside {self.sensor_name} ground grid bounding box "
                        f"[Lon: {self.lon_min:.4f}..{self.lon_max:.4f}, Lat: {self.lat_min:.4f}..{self.lat_max:.4f}]."
                    )
                continue

            # Binary search candidate rows
            # Since latitude decreases monotonically with row index:
            # -_row_lat_min is increasing with row index
            i_min = max(0, int(np.searchsorted(-self._row_lat_min, -target_lat, side="left")) - 1)
            i_max = min(self.n_scans - 2, int(np.searchsorted(-self._row_lat_max, -target_lat, side="right")) + 1)

            p_target = np.array([target_lon, target_lat], dtype=np.float64)
            found = False

            for i in range(i_min, i_max + 1):
                row_lons = self.lons_grid[i]
                row_min_lon = min(row_lons[0], row_lons[-1]) - 0.02
                row_max_lon = max(row_lons[0], row_lons[-1]) + 0.02

                if not (row_min_lon <= target_lon <= row_max_lon):
                    continue

                # Binary search column
                if row_lons[-1] > row_lons[0]:
                    # Increasing (OHRC)
                    j_cand = int(np.searchsorted(row_lons, target_lon, side="right")) - 1
                else:
                    # Decreasing (TMC-2)
                    j_cand = int(np.searchsorted(-row_lons, -target_lon, side="right")) - 1

                j_cand_min = max(0, j_cand - 1)
                j_cand_max = min(self.n_pixels - 2, j_cand + 1)

                for j in range(j_cand_min, j_cand_max + 1):
                    p00 = np.array([self.lons_grid[i, j], self.lats_grid[i, j]])
                    p10 = np.array([self.lons_grid[i, j + 1], self.lats_grid[i, j + 1]])
                    p01 = np.array([self.lons_grid[i + 1, j], self.lats_grid[i + 1, j]])
                    p11 = np.array([self.lons_grid[i + 1, j + 1], self.lats_grid[i + 1, j + 1]])

                    uv = _solve_inverse_bilinear_cell(p_target, p00, p10, p01, p11)

                    if -eps <= uv[0] <= 1.0 + eps and -eps <= uv[1] <= 1.0 + eps:
                        u_c = float(np.clip(uv[0], 0.0, 1.0))
                        v_c = float(np.clip(uv[1], 0.0, 1.0))
                        px_val = self.u_pixels[j] + u_c * (self.u_pixels[j + 1] - self.u_pixels[j])
                        sc_val = self.u_scans[i] + v_c * (self.u_scans[i + 1] - self.u_scans[i])
                        out_px[idx] = px_val
                        out_sc[idx] = sc_val
                        found = True
                        break
                if found:
                    break

            if not found and check_bounds:
                raise ValueError(
                    f"Geographic coordinate (Lon={target_lon:.6f}, Lat={target_lat:.6f}) "
                    f"could not be mapped into any valid quad cell of {self.sensor_name} grid."
                )

        if is_scalar:
            return float(out_px[0]), float(out_sc[0])
        return out_px.reshape(orig_shape), out_sc.reshape(orig_shape)

    def validate_held_out(
        self,
        stride: int = 2,
        max_scans: Optional[int] = None,
    ) -> ValidationMetrics:
        """Validate genuine interpolation accuracy by evaluating against held-out grid points.

        A coarse subgrid (with spacing stride * 100) is used to predict the coordinates
        of the held-out central points, comparing the prediction directly against the
        official ISRO ground-grid tie point ground truth.

        Args:
            stride: Subsampling stride (default 2 -> 200x200 pixel cells with midpoint evaluation).
            max_scans: Optional scan limit to restrict execution time.

        Returns:
            ValidationMetrics instance containing errors in pixel, scan, and meters.
        """
        scans_to_use = self.u_scans if max_scans is None else self.u_scans[:max_scans]
        n_sc = len(scans_to_use)

        sub_lons = self.lons_grid[:n_sc, :]
        sub_lats = self.lats_grid[:n_sc, :]

        # Training grid: even rows and columns
        train_scans = scans_to_use[::stride]
        train_pixels = self.u_pixels[::stride]
        train_lons = sub_lons[::stride, ::stride]
        train_lats = sub_lats[::stride, ::stride]

        # Test points: odd rows and columns (central points of 2x2 cells)
        test_scans = scans_to_use[1::stride]
        test_pixels = self.u_pixels[1::stride]

        pixel_errs: List[float] = []
        scan_errs: List[float] = []
        geo_dist_m: List[float] = []

        m_per_deg_lat = MOON_RADIUS_METERS * np.pi / 180.0

        n_test_r = min(len(test_scans), train_lons.shape[0] - 1)
        n_test_c = min(len(test_pixels), train_lons.shape[1] - 1)

        for r in range(n_test_r):
            true_sc = test_scans[r]
            for c in range(n_test_c):
                true_px = test_pixels[c]

                # True official coordinate at held-out tie point
                true_lon = sub_lons[stride * r + 1, stride * c + 1]
                true_lat = sub_lats[stride * r + 1, stride * c + 1]
                p_true = np.array([true_lon, true_lat], dtype=np.float64)

                # Four corners in training subgrid
                p00 = np.array([train_lons[r, c], train_lats[r, c]])
                p10 = np.array([train_lons[r, c + 1], train_lats[r, c + 1]])
                p01 = np.array([train_lons[r + 1, c], train_lats[r + 1, c]])
                p11 = np.array([train_lons[r + 1, c + 1], train_lats[r + 1, c + 1]])

                # Forward interpolation error in geographic coordinates (lon, lat at true pixel/scan)
                # Midpoint normalized coordinate in coarse cell
                u_mid = (true_px - train_pixels[c]) / (train_pixels[c + 1] - train_pixels[c])
                v_mid = (true_sc - train_scans[r]) / (train_scans[r + 1] - train_scans[r])

                w00 = (1.0 - u_mid) * (1.0 - v_mid)
                w10 = u_mid * (1.0 - v_mid)
                w01 = (1.0 - u_mid) * v_mid
                w11 = u_mid * v_mid
                pred_geo = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11

                dlon_deg = abs(pred_geo[0] - true_lon)
                dlat_deg = abs(pred_geo[1] - true_lat)
                dx_m = dlon_deg * m_per_deg_lat * np.cos(np.radians(true_lat))
                dy_m = dlat_deg * m_per_deg_lat
                dist_m = float(np.hypot(dx_m, dy_m))
                geo_dist_m.append(dist_m)

                # Reverse prediction: predict pixel and scan from true geographic coordinate
                uv = _solve_inverse_bilinear_cell(p_true, p00, p10, p01, p11)
                pred_px = train_pixels[c] + uv[0] * (train_pixels[c + 1] - train_pixels[c])
                pred_sc = train_scans[r] + uv[1] * (train_scans[r + 1] - train_scans[r])

                pixel_errs.append(float(abs(pred_px - true_px)))
                scan_errs.append(float(abs(pred_sc - true_sc)))

        px_a = np.array(pixel_errs, dtype=np.float64)
        sc_a = np.array(scan_errs, dtype=np.float64)
        dist_a = np.array(geo_dist_m, dtype=np.float64)

        return ValidationMetrics(
            sensor_name=self.sensor_name,
            strategy=f"held_out_midpoints_stride_{stride}",
            num_samples=len(px_a),
            pixel_error_mean=float(px_a.mean()),
            pixel_error_median=float(np.median(px_a)),
            pixel_error_max=float(px_a.max()),
            pixel_error_p95=float(np.percentile(px_a, 95)),
            pixel_error_std=float(px_a.std()),
            scan_error_mean=float(sc_a.mean()),
            scan_error_median=float(np.median(sc_a)),
            scan_error_max=float(sc_a.max()),
            scan_error_p95=float(np.percentile(sc_a, 95)),
            scan_error_std=float(sc_a.std()),
            geo_dist_m_mean=float(dist_a.mean()),
            geo_dist_m_median=float(np.median(dist_a)),
            geo_dist_m_max=float(dist_a.max()),
            geo_dist_m_p95=float(np.percentile(dist_a, 95)),
            geo_dist_m_std=float(dist_a.std()),
        )

    def validate_round_trip(
        self,
        n_samples: int = 500,
        random_seed: int = 42,
    ) -> ValidationMetrics:
        """Validate mathematical round-trip consistency: (pixel, scan) -> (lon, lat) -> (pixel, scan).

        Tests arbitrary continuous non-grid points across the interior of the sensor footprint.

        Args:
            n_samples: Number of random interior query points.
            random_seed: Seed for reproducibility.

        Returns:
            ValidationMetrics instance.
        """
        rng = np.random.default_rng(random_seed)
        test_px = rng.uniform(self.pixel_min + 50.0, self.pixel_max - 50.0, n_samples)
        test_sc = rng.uniform(self.scan_min + 50.0, self.scan_max - 50.0, n_samples)

        # 1. Forward: pixel, scan -> lon, lat
        lons, lats = self.pixel_to_geo(test_px, test_sc, check_bounds=True)

        # 2. Backward: lon, lat -> pixel, scan
        rec_px, rec_sc = self.geo_to_pixel(lons, lats, check_bounds=True)

        px_err = np.abs(np.asarray(rec_px) - test_px)
        sc_err = np.abs(np.asarray(rec_sc) - test_sc)

        return ValidationMetrics(
            sensor_name=self.sensor_name,
            strategy="round_trip_arbitrary_interior_points",
            num_samples=n_samples,
            pixel_error_mean=float(px_err.mean()),
            pixel_error_median=float(np.median(px_err)),
            pixel_error_max=float(px_err.max()),
            pixel_error_p95=float(np.percentile(px_err, 95)),
            pixel_error_std=float(px_err.std()),
            scan_error_mean=float(sc_err.mean()),
            scan_error_median=float(np.median(sc_err)),
            scan_error_max=float(sc_err.max()),
            scan_error_p95=float(np.percentile(sc_err, 95)),
            scan_error_std=float(sc_err.std()),
        )

    def compute_global_affine_diagnostic(self) -> ValidationMetrics:
        """Fit a single global affine transformation for diagnostic comparison.

        Demonstrates the residual errors that would occur if using a simple global affine
        model instead of the true ISRO ground-grid cell-wise bilinear interpolation.

        Returns:
            ValidationMetrics instance reporting affine model residual errors.
        """
        # Linear system: [lon, lat, 1] @ M = [pixel, scan]
        flat_lons = self.lons_grid.ravel()
        flat_lats = self.lats_grid.ravel()
        grid_sc, grid_px = np.meshgrid(self.u_scans, self.u_pixels, indexing="ij")
        flat_sc = grid_sc.ravel()
        flat_px = grid_px.ravel()

        X = np.column_stack([flat_lons, flat_lats, np.ones_like(flat_lons)])
        Y = np.column_stack([flat_px, flat_sc])

        M, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        pred_Y = X @ M
        residuals = np.abs(Y - pred_Y)

        px_res = residuals[:, 0]
        sc_res = residuals[:, 1]

        return ValidationMetrics(
            sensor_name=self.sensor_name,
            strategy="diagnostic_global_affine_fit",
            num_samples=len(X),
            pixel_error_mean=float(px_res.mean()),
            pixel_error_median=float(np.median(px_res)),
            pixel_error_max=float(px_res.max()),
            pixel_error_p95=float(np.percentile(px_res, 95)),
            pixel_error_std=float(px_res.std()),
            scan_error_mean=float(sc_res.mean()),
            scan_error_median=float(np.median(sc_res)),
            scan_error_max=float(sc_res.max()),
            scan_error_p95=float(np.percentile(sc_res, 95)),
            scan_error_std=float(sc_res.std()),
        )


def compute_sensor_overlap(
    ohrc_grid: GroundGrid,
    tmc2_grid: GroundGrid,
    num_boundary_samples_per_edge: int = 50,
) -> Tuple[OverlapResult, Dict[str, np.ndarray]]:
    """Compute exact pixel overlap of the OHRC footprint inside the TMC-2 sensor frame.

    Samples the OHRC image boundary at dense intervals, converts boundary points to
    geographic coordinates, and maps each point into TMC-2 image pixel coordinates.

    Args:
        ohrc_grid: Loaded GroundGrid for OHRC sensor.
        tmc2_grid: Loaded GroundGrid for TMC-2 sensor.
        num_boundary_samples_per_edge: Sampling density along each edge.

    Returns:
        Tuple of (OverlapResult, dict containing sampled boundary points).
    """
    # Sample OHRC boundary edges
    # Top edge: scan = scan_min, pixel varies from min to max
    top_px = np.linspace(ohrc_grid.pixel_min, ohrc_grid.pixel_max, num_boundary_samples_per_edge)
    top_sc = np.full_like(top_px, ohrc_grid.scan_min)

    # Bottom edge: scan = scan_max, pixel varies from max to min
    bot_px = np.linspace(ohrc_grid.pixel_max, ohrc_grid.pixel_min, num_boundary_samples_per_edge)
    bot_sc = np.full_like(bot_px, ohrc_grid.scan_max)

    # Right edge: pixel = pixel_max, scan varies from min to max
    rgt_sc = np.linspace(ohrc_grid.scan_min, ohrc_grid.scan_max, num_boundary_samples_per_edge)
    rgt_px = np.full_like(rgt_sc, ohrc_grid.pixel_max)

    # Left edge: pixel = pixel_min, scan varies from max to min
    lft_sc = np.linspace(ohrc_grid.scan_max, ohrc_grid.scan_min, num_boundary_samples_per_edge)
    lft_px = np.full_like(lft_sc, ohrc_grid.pixel_min)

    boundary_px = np.concatenate([top_px, rgt_px, bot_px, lft_px])
    boundary_sc = np.concatenate([top_sc, rgt_sc, bot_sc, lft_sc])

    # Convert OHRC pixels -> Geographic (lon, lat)
    b_lons, b_lats = ohrc_grid.pixel_to_geo(boundary_px, boundary_sc, check_bounds=True)

    # Convert Geographic -> TMC-2 pixels (pixel, scan)
    tmc_px, tmc_sc = tmc2_grid.geo_to_pixel(b_lons, b_lats, check_bounds=True)

    min_sc = float(np.min(tmc_sc))
    max_sc = float(np.max(tmc_sc))
    min_px = float(np.min(tmc_px))
    max_px = float(np.max(tmc_px))

    # Integer bounds covering the full region
    scan_start = int(np.floor(min_sc))
    scan_end = int(np.ceil(max_sc))
    pixel_start = int(np.floor(min_px))
    pixel_end = int(np.ceil(max_px))

    # Official tie points within this overlap box
    sub_scans = tmc2_grid.u_scans[
        (tmc2_grid.u_scans >= scan_start) & (tmc2_grid.u_scans <= scan_end)
    ]
    sub_pixels = tmc2_grid.u_pixels[
        (tmc2_grid.u_pixels >= pixel_start) & (tmc2_grid.u_pixels <= pixel_end)
    ]
    num_tie_pts = len(sub_scans) * len(sub_pixels)

    res = OverlapResult(
        ohrc_geo_bounds=(ohrc_grid.lon_min, ohrc_grid.lat_min, ohrc_grid.lon_max, ohrc_grid.lat_max),
        ohrc_image_shape=(ohrc_grid.scan_max + 1, ohrc_grid.pixel_max + 1),
        tmc2_image_shape=(tmc2_grid.scan_max + 1, tmc2_grid.pixel_max + 1),
        tmc2_scan_bounds_continuous=(min_sc, max_sc),
        tmc2_pixel_bounds_continuous=(min_px, max_px),
        tmc2_overlap_scan_start=scan_start,
        tmc2_overlap_scan_end=scan_end,
        tmc2_overlap_pixel_start=pixel_start,
        tmc2_overlap_pixel_end=pixel_end,
        num_tie_points_in_overlap=num_tie_pts,
        mapped_boundary_samples=len(boundary_px),
    )

    boundary_data = {
        "ohrc_pixel": boundary_px,
        "ohrc_scan": boundary_sc,
        "longitude": b_lons,
        "latitude": b_lats,
        "tmc2_pixel": tmc_px,
        "tmc2_scan": tmc_sc,
    }

    return res, boundary_data


# Top-level functional interfaces conceptually requested
def geo_to_pixel(
    longitude: Union[float, np.ndarray],
    latitude: Union[float, np.ndarray],
    sensor: Union[str, GroundGrid],
    root_dir: Optional[Path | str] = None,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Convert geographic coordinates (longitude, latitude) to (pixel, scan) for the specified sensor."""
    if isinstance(sensor, GroundGrid):
        grid = sensor
    else:
        # Load from default dataset paths
        s_upper = sensor.upper()
        root = Path(root_dir) if root_dir else Path(r"D:\ChandraData\triplet_1")
        if s_upper == "OHRC":
            csv_path = root / "OHRC" / "geometry" / "calibrated" / "20210405" / "ch2_ohr_ncp_20210405T1606536730_g_grd_d18.csv"
        elif s_upper in ("TMC", "TMC2", "TMC-2"):
            csv_path = root / "TMC" / "geometry" / "calibrated" / "20250807" / "ch2_tmc_ncf_20250807T1904346039_g_grd_d18.csv"
        else:
            raise ValueError(f"Unknown sensor: {sensor}")
        grid = GroundGrid(csv_path, sensor_name=s_upper)

    return grid.geo_to_pixel(longitude, latitude, check_bounds=True)


def pixel_to_geo(
    pixel: Union[float, np.ndarray],
    scan: Union[float, np.ndarray],
    sensor: Union[str, GroundGrid],
    root_dir: Optional[Path | str] = None,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Convert image pixel coordinates (pixel, scan) to (longitude, latitude) for the specified sensor."""
    if isinstance(sensor, GroundGrid):
        grid = sensor
    else:
        s_upper = sensor.upper()
        root = Path(root_dir) if root_dir else Path(r"D:\ChandraData\triplet_1")
        if s_upper == "OHRC":
            csv_path = root / "OHRC" / "geometry" / "calibrated" / "20210405" / "ch2_ohr_ncp_20210405T1606536730_g_grd_d18.csv"
        elif s_upper in ("TMC", "TMC2", "TMC-2"):
            csv_path = root / "TMC" / "geometry" / "calibrated" / "20250807" / "ch2_tmc_ncf_20250807T1904346039_g_grd_d18.csv"
        else:
            raise ValueError(f"Unknown sensor: {sensor}")
        grid = GroundGrid(csv_path, sensor_name=s_upper)

    return grid.pixel_to_geo(pixel, scan, check_bounds=True)
