"""Baseline lunar crater detector based on circular Hough transform and photometric rim verification.

This detector is a reproducible, deterministic baseline for lunar terrain:
1. Identifies circular depression candidates via Circular Hough Transform on preprocessed gradients.
2. Performs vectorized photometric dipole verification: evaluates the contrast between the sun-illuminated
   rim arc and shadowed rim arc/floor.
3. Computes radial gradient edge support along the candidate rim circumference.
4. Produces a calibrated confidence score in [0.0, 1.0].
5. Rejects non-crater terrain noise and outputs standardized CraterDetection objects.
"""

from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from src.crater_detection.base import BaseCraterDetector, CraterDetection


class BaselineCraterDetector(BaseCraterDetector):
    """Deterministic photometric and gradient-based circular crater detector."""

    def __init__(
        self,
        name: str = "BaselineRimDetector",
        version: str = "1.0.0",
        min_radius_px: float = 6.0,
        max_radius_px: float = 100.0,
        min_dist_px: float = 16.0,
        canny_param1: float = 70.0,
        accumulator_param2: float = 50.0,
        confidence_threshold: float = 0.30,
        num_sample_points: int = 24,
        dp: float = 1.5,
    ):
        super().__init__(
            name=name,
            version=version,
            min_radius_px=min_radius_px,
            max_radius_px=max_radius_px,
            confidence_threshold=confidence_threshold,
        )
        self.min_dist_px = min_dist_px
        self.canny_param1 = canny_param1
        self.accumulator_param2 = accumulator_param2
        self.num_sample_points = num_sample_points
        self.dp = dp

    def detect(
        self,
        image: np.ndarray,
        gsd_m: float,
        source_sensor: str = "unknown",
        image_region: str = "unknown",
        preprocessing_representation: str = "clahe",
        tile_id: Optional[str] = None,
    ) -> List[CraterDetection]:
        """Detect craters in a 2D grayscale image array (tile).

        Args:
            image: 2D uint8 numpy array (ideally CLAHE or percentile-normalized).
            gsd_m: Ground sample distance in meters/pixel.
            source_sensor: "OHRC" or "TMC-2".
            image_region: Region identifier.
            preprocessing_representation: "clahe", "percentile_normalized", etc.
            tile_id: Tile identifier if operating on a tile.

        Returns:
            List of CraterDetection objects detected in this image/tile.
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

        # Ensure image is uint8
        if image.dtype != np.uint8:
            img_min, img_max = float(np.min(image)), float(np.max(image))
            if img_max > img_min:
                img_u8 = ((image.astype(np.float32) - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
            else:
                img_u8 = np.zeros_like(image, dtype=np.uint8)
        else:
            img_u8 = image

        h, w = img_u8.shape
        if h < 16 or w < 16:
            return []

        # 1. Compute spatial Sobel gradients for edge support verification
        gx = cv2.Sobel(img_u8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_u8, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(gx, gy)

        # 2. Run Hough Circle Transform
        circles = cv2.HoughCircles(
            img_u8,
            cv2.HOUGH_GRADIENT,
            dp=float(self.dp),
            minDist=float(self.min_dist_px),
            param1=float(self.canny_param1),
            param2=float(self.accumulator_param2),
            minRadius=int(self.min_radius_px),
            maxRadius=int(self.max_radius_px),
        )

        if circles is None or len(circles) == 0:
            return []

        circ_arr = circles[0]
        cxs = circ_arr[:, 0]
        cys = circ_arr[:, 1]
        radii = circ_arr[:, 2]

        # Valid in-bounds mask
        in_bounds = (
            (cxs - radii >= 1) & (cxs + radii < w - 1) &
            (cys - radii >= 1) & (cys + radii < h - 1)
        )
        if not np.any(in_bounds):
            return []

        cxs = cxs[in_bounds]
        cys = cys[in_bounds]
        radii = radii[in_bounds]
        num_cand = len(cxs)

        # 3. Vectorized Photometric and Gradient Sampling along rim
        angles = np.linspace(0.0, 2.0 * np.pi, self.num_sample_points, endpoint=False)
        cos_a = np.cos(angles)[None, :]  # (1, K)
        sin_a = np.sin(angles)[None, :]  # (1, K)

        xs = np.clip(np.round(cxs[:, None] + radii[:, None] * cos_a).astype(int), 0, w - 1)
        ys = np.clip(np.round(cys[:, None] + radii[:, None] * sin_a).astype(int), 0, h - 1)

        rim_intensities = img_u8[ys, xs]  # (N, K)
        rim_gradients = grad_mag[ys, xs]  # (N, K)

        # 4. Vectorized Photometric Dipole Verification
        quarter_samples = max(2, self.num_sample_points // 4)
        sorted_rim = np.sort(rim_intensities, axis=1)
        bright_arc = np.mean(sorted_rim[:, -quarter_samples:], axis=1)
        dark_arc = np.mean(sorted_rim[:, :quarter_samples], axis=1)
        dipole_contrast = bright_arc - dark_arc  # Range: [0, 255]

        # 5. Gradient Saliency along Rim
        mean_grad = np.mean(rim_gradients, axis=1)
        edge_support_ratio = np.mean(rim_gradients > 25.0, axis=1)

        # 6. Radial Gradient Symmetry
        g_sample_x = gx[ys, xs]
        g_sample_y = gy[ys, xs]
        radial_alignment = np.mean(
            np.abs((g_sample_x * cos_a + g_sample_y * sin_a) / (rim_gradients + 1e-6)),
            axis=1,
        )

        # 7. Composite Calibrated Confidence Score
        scores = (
            (dipole_contrast / 255.0) * 0.40
            + (np.minimum(mean_grad, 150.0) / 150.0) * 0.25
            + edge_support_ratio * 0.20
            + radial_alignment * 0.15
        )
        scores = np.clip(scores, 0.0, 1.0)

        # Filter by confidence threshold
        valid_mask = scores >= self.confidence_threshold
        if not np.any(valid_mask):
            return []

        detections: List[CraterDetection] = []
        valid_indices = np.where(valid_mask)[0]

        for idx in valid_indices:
            cx_val = float(cxs[idx])
            cy_val = float(cys[idx])
            r_val = float(radii[idx])
            score_val = float(scores[idx])
            d_px = 2.0 * r_val
            d_m = d_px * gsd_m

            det = CraterDetection(
                center_x=cx_val,
                center_y=cy_val,
                radius_px=r_val,
                diameter_px=d_px,
                diameter_m=d_m,
                confidence=score_val,
                source_sensor=source_sensor,
                image_region=image_region,
                detector_name=self.name,
                detector_version=self.version,
                preprocessing_representation=preprocessing_representation,
                inference_parameters={
                    "dp": self.dp,
                    "canny_param1": self.canny_param1,
                    "accumulator_param2": self.accumulator_param2,
                    "min_radius_px": self.min_radius_px,
                    "max_radius_px": self.max_radius_px,
                    "min_dist_px": self.min_dist_px,
                    "confidence_threshold": self.confidence_threshold,
                    "dipole_contrast": float(round(float(dipole_contrast[idx]), 2)),
                    "mean_rim_gradient": float(round(float(mean_grad[idx]), 2)),
                    "edge_support_ratio": float(round(float(edge_support_ratio[idx]), 3)),
                    "radial_alignment": float(round(float(radial_alignment[idx]), 3)),
                },
                tile_id=tile_id,
            )
            detections.append(det)

        return detections
