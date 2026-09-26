"""YOLOv9-C crater detector implementation using the author's official trained checkpoint.

Wraps the paper's exact YOLOv9-C model (YOLOv9Best.pt) trained on LROC NAC multi-illumination
imagery (MiLOIs) for crater detection.
"""

import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import torch

# Ensure external/yolov9 is importable
YOLOV9_ROOT = Path(__file__).resolve().parent.parent.parent / "external" / "yolov9"
if str(YOLOV9_ROOT) not in sys.path and YOLOV9_ROOT.is_dir():
    sys.path.insert(0, str(YOLOV9_ROOT))

from src.crater_detection.base import BaseCraterDetector, CraterDetection


class YOLOv9CraterDetector(BaseCraterDetector):
    """Crater detector utilizing the author's official trained YOLOv9-C checkpoint."""

    def __init__(
        self,
        weights_path: Union[str, Path] = "data/models/YOLOv9Best.pt",
        name: str = "paper_yolov9c",
        version: str = "1.0.0",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        imgsz: int = 640,
    ):
        super().__init__(
            name=name,
            version=version,
            min_radius_px=3.0,
            max_radius_px=320.0,
            confidence_threshold=confidence_threshold,
        )
        self.weights_path = Path(weights_path)
        self.iou_threshold = iou_threshold
        self.device = torch.device(device)
        self.imgsz = imgsz
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the PyTorch model checkpoint into memory."""
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"YOLOv9 weights file not found at: {self.weights_path}")

        # Ensure external/yolov9 is in sys.path
        if str(YOLOV9_ROOT) not in sys.path and YOLOV9_ROOT.is_dir():
            sys.path.insert(0, str(YOLOV9_ROOT))

        ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        self.model = ckpt["model"].float().to(self.device).eval()

    def detect(
        self,
        image: np.ndarray,
        gsd_m: float,
        source_sensor: str = "unknown",
        image_region: str = "unknown",
        preprocessing_representation: str = "clahe",
        tile_id: Optional[str] = None,
    ) -> List[CraterDetection]:
        """Run YOLOv9-C detection on an image patch (typically 640x640).

        Args:
            image: 2D uint8 numpy array.
            gsd_m: Ground sample distance in meters.
            source_sensor: Sensor identifier (OHRC, TMC-2).
            image_region: Region label.
            preprocessing_representation: Preprocessing method applied.
            tile_id: Tile identifier if tiled.

        Returns:
            List of detected CraterDetection instances.
        """
        if self.model is None:
            self._load_model()

        h_orig, w_orig = image.shape[:2]
        if h_orig < 16 or w_orig < 16:
            return []

        # Resize/pad to self.imgsz if dimensions differ
        if h_orig != self.imgsz or w_orig != self.imgsz:
            # Letterbox or direct resize to imgsz
            scale_x = float(w_orig) / float(self.imgsz)
            scale_y = float(h_orig) / float(self.imgsz)
            input_img = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        else:
            scale_x = 1.0
            scale_y = 1.0
            input_img = image

        # Convert to 3-channel RGB float tensor (1, 3, imgsz, imgsz) in [0, 1]
        if input_img.ndim == 2:
            rgb = np.stack([input_img] * 3, axis=2)
        elif input_img.shape[2] == 1:
            rgb = np.repeat(input_img, 3, axis=2)
        else:
            rgb = input_img

        tensor_x = torch.from_numpy(rgb).permute(2, 0, 1).float().to(self.device) / 255.0
        tensor_x = tensor_x.unsqueeze(0)

        from utils.general import non_max_suppression

        with torch.no_grad():
            preds = self.model(tensor_x)
            if isinstance(preds, tuple):
                pred_out = preds[0]
            else:
                pred_out = preds

            nms_results = non_max_suppression(
                pred_out,
                conf_thres=self.confidence_threshold,
                iou_thres=self.iou_threshold,
            )

        detections: List[CraterDetection] = []
        if not nms_results or len(nms_results[0]) == 0:
            return []

        boxes = nms_results[0].cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = box[:6]
            # Rescale box back to original patch coordinate frame
            x1_orig = x1 * scale_x
            x2_orig = x2 * scale_x
            y1_orig = y1 * scale_y
            y2_orig = y2 * scale_y

            cx = float((x1_orig + x2_orig) / 2.0)
            cy = float((y1_orig + y2_orig) / 2.0)
            w = float(x2_orig - x1_orig)
            h = float(y2_orig - y1_orig)
            diameter_px = float((w + h) / 2.0)
            radius_px = float(diameter_px / 2.0)
            diameter_m = float(diameter_px * gsd_m)

            det = CraterDetection(
                center_x=cx,
                center_y=cy,
                radius_px=radius_px,
                diameter_px=diameter_px,
                diameter_m=diameter_m,
                confidence=float(conf),
                source_sensor=source_sensor,
                image_region=image_region,
                detector_name=self.name,
                detector_version=self.version,
                preprocessing_representation=preprocessing_representation,
                inference_parameters={
                    "weights_path": str(self.weights_path),
                    "confidence_threshold": self.confidence_threshold,
                    "iou_threshold": self.iou_threshold,
                    "imgsz": self.imgsz,
                    "box_w": round(w, 2),
                    "box_h": round(h, 2),
                },
                tile_id=tile_id,
            )
            detections.append(det)

        return detections
