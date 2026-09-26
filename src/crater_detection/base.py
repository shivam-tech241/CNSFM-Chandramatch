"""Base definitions and standardized representation for crater detections.

Provides:
- CraterDetection dataclass with complete spatial, physical, and sensor metadata.
- I/O functions for CSV, JSON, and the official CNSFM '.craters' format.
- BaseCraterDetector abstract class for pluggable detector backends.
- ExternalFileCraterDetector for ingesting precomputed / external detections.
"""

from abc import ABC, abstractmethod
import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


@dataclass
class CraterDetection:
    """Standardized representation of a detected lunar crater."""

    center_x: float  # Subpixel column in image coordinates
    center_y: float  # Subpixel row in image coordinates
    radius_px: float  # Radius in pixels
    diameter_px: float  # Diameter in pixels (= 2 * radius_px)
    diameter_m: float  # Physical diameter in meters (= diameter_px * GSD)
    confidence: float  # Confidence score in [0.0, 1.0]
    source_sensor: str  # Sensor name, e.g. "OHRC" or "TMC-2"
    image_region: str  # Region label, e.g. "OHRC_Benchmark" or "TMC-2_Overlap"
    detector_name: str  # Detector identifier, e.g. "BaselineRimDetector"
    detector_version: str = "1.0.0"
    preprocessing_representation: str = "clahe"
    inference_parameters: Dict[str, Any] = field(default_factory=dict)
    tile_id: Optional[str] = None
    crater_id: Optional[int] = None

    def __post_init__(self):
        # Enforce consistency between radius and diameter
        if self.diameter_px is None or self.diameter_px <= 0.0:
            self.diameter_px = 2.0 * float(self.radius_px)
        if self.radius_px is None or self.radius_px <= 0.0:
            self.radius_px = float(self.diameter_px) / 2.0

        # Confidence clipping
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to a Python dictionary."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CraterDetection":
        """Create CraterDetection from a dictionary."""
        d_copy = dict(d)
        return cls(**d_copy)

    def to_craters_line(self, idx: Optional[int] = None) -> str:
        """Format detection in the official CNSFM '.craters' format.

        Format: center_x, center_y, width, height, confidence, id;
        Note: For circular craters, width = height = diameter_px.
        """
        cid = self.crater_id if idx is None else idx
        if cid is None:
            cid = 0
        w = self.diameter_px
        h = self.diameter_px
        return f"{self.center_x:.2f}, {self.center_y:.2f}, {w:.2f}, {h:.2f}, {self.confidence:.4f}, {cid};"

    @classmethod
    def from_craters_line(
        cls,
        line: str,
        source_sensor: str,
        gsd_m: float,
        image_region: str = "unknown",
        detector_name: str = "CNSFM_YOLOv9",
        detector_version: str = "paper_checkpoint",
        preprocessing_representation: str = "raw",
    ) -> "CraterDetection":
        """Parse a single line from a CNSFM '.craters' file."""
        cleaned = line.strip().rstrip(";").strip()
        parts = [p.strip() for p in cleaned.split(",")]
        if len(parts) < 6:
            raise ValueError(f"Malformed '.craters' line: '{line}' (expected 6 values)")

        cx = float(parts[0])
        cy = float(parts[1])
        w = float(parts[2])
        h = float(parts[3])
        conf = float(parts[4])
        cid = int(parts[5])

        # Effective diameter is mean of width and height
        diameter_px = (w + h) / 2.0
        radius_px = diameter_px / 2.0
        diameter_m = diameter_px * gsd_m

        return cls(
            center_x=cx,
            center_y=cy,
            radius_px=radius_px,
            diameter_px=diameter_px,
            diameter_m=diameter_m,
            confidence=conf,
            source_sensor=source_sensor,
            image_region=image_region,
            detector_name=detector_name,
            detector_version=detector_version,
            preprocessing_representation=preprocessing_representation,
            crater_id=cid,
        )


def save_detections_csv(detections: List[CraterDetection], filepath: Union[str, Path]) -> None:
    """Save crater detections to CSV."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "crater_id",
        "source_sensor",
        "image_region",
        "global_x",
        "global_y",
        "radius_px",
        "diameter_px",
        "diameter_m",
        "confidence",
        "tile_id",
        "detector_name",
        "detector_version",
        "preprocessing_representation",
    ]

    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, det in enumerate(detections):
            cid = det.crater_id if det.crater_id is not None else idx
            row = {
                "crater_id": cid,
                "source_sensor": det.source_sensor,
                "image_region": det.image_region,
                "global_x": f"{det.center_x:.3f}",
                "global_y": f"{det.center_y:.3f}",
                "radius_px": f"{det.radius_px:.3f}",
                "diameter_px": f"{det.diameter_px:.3f}",
                "diameter_m": f"{det.diameter_m:.3f}",
                "confidence": f"{det.confidence:.4f}",
                "tile_id": det.tile_id or "",
                "detector_name": det.detector_name,
                "detector_version": det.detector_version,
                "preprocessing_representation": det.preprocessing_representation,
            }
            writer.writerow(row)


def load_detections_csv(filepath: Union[str, Path]) -> List[CraterDetection]:
    """Load crater detections from CSV."""
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"Detections CSV not found: {p}")

    detections = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            det = CraterDetection(
                center_x=float(row["global_x"]),
                center_y=float(row["global_y"]),
                radius_px=float(row["radius_px"]),
                diameter_px=float(row["diameter_px"]),
                diameter_m=float(row["diameter_m"]),
                confidence=float(row["confidence"]),
                source_sensor=row["source_sensor"],
                image_region=row["image_region"],
                detector_name=row["detector_name"],
                detector_version=row.get("detector_version", "1.0.0"),
                preprocessing_representation=row.get("preprocessing_representation", "clahe"),
                tile_id=row.get("tile_id") if row.get("tile_id") else None,
                crater_id=int(row["crater_id"]) if row.get("crater_id") else None,
            )
            detections.append(det)
    return detections


def save_detections_json(detections: List[CraterDetection], filepath: Union[str, Path]) -> None:
    """Save crater detections to JSON."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [d.to_dict() for d in detections]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_detections_json(filepath: Union[str, Path]) -> List[CraterDetection]:
    """Load crater detections from JSON."""
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"Detections JSON not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [CraterDetection.from_dict(d) for d in data]


def save_detections_craters(detections: List[CraterDetection], filepath: Union[str, Path]) -> None:
    """Save crater detections in the official CNSFM '.craters' format."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for idx, det in enumerate(detections):
            cid = det.crater_id if det.crater_id is not None else idx
            f.write(det.to_craters_line(cid) + "\n")


def load_detections_craters(
    filepath: Union[str, Path],
    source_sensor: str,
    gsd_m: float,
    image_region: str = "unknown",
) -> List[CraterDetection]:
    """Load crater detections from a CNSFM '.craters' file."""
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"Detections '.craters' file not found: {p}")

    detections = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            det = CraterDetection.from_craters_line(
                line=line_str,
                source_sensor=source_sensor,
                gsd_m=gsd_m,
                image_region=image_region,
            )
            detections.append(det)
    return detections


class BaseCraterDetector(ABC):
    """Abstract base class for crater detectors."""

    def __init__(
        self,
        name: str = "BaseDetector",
        version: str = "1.0.0",
        min_radius_px: float = 5.0,
        max_radius_px: float = 150.0,
        confidence_threshold: float = 0.3,
    ):
        if min_radius_px <= 0 or max_radius_px <= min_radius_px:
            raise ValueError(f"Invalid radius range: [{min_radius_px}, {max_radius_px}]")
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(f"Confidence threshold must be in [0, 1], got {confidence_threshold}")

        self.name = name
        self.version = version
        self.min_radius_px = min_radius_px
        self.max_radius_px = max_radius_px
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        gsd_m: float,
        source_sensor: str = "unknown",
        image_region: str = "unknown",
        preprocessing_representation: str = "clahe",
        tile_id: Optional[str] = None,
    ) -> List[CraterDetection]:
        """Detect craters in a 2D image array."""
        pass


class ExternalFileCraterDetector(BaseCraterDetector):
    """Crater detector that loads detections from an external file (.craters, .csv, or .json)."""

    def __init__(
        self,
        filepath: Union[str, Path],
        name: str = "ExternalDetector",
        version: str = "1.0.0",
        confidence_threshold: float = 0.0,
    ):
        super().__init__(name=name, version=version, confidence_threshold=confidence_threshold)
        self.filepath = Path(filepath)

    def detect(
        self,
        image: np.ndarray,
        gsd_m: float,
        source_sensor: str = "unknown",
        image_region: str = "unknown",
        preprocessing_representation: str = "raw",
        tile_id: Optional[str] = None,
    ) -> List[CraterDetection]:
        """Load detections from file and filter by confidence."""
        suffix = self.filepath.suffix.lower()
        if suffix == ".craters":
            detections = load_detections_craters(
                self.filepath,
                source_sensor=source_sensor,
                gsd_m=gsd_m,
                image_region=image_region,
            )
        elif suffix == ".csv":
            detections = load_detections_csv(self.filepath)
        elif suffix == ".json":
            detections = load_detections_json(self.filepath)
        else:
            raise ValueError(f"Unsupported file format for external detections: {suffix}")

        filtered = [d for d in detections if d.confidence >= self.confidence_threshold]
        return filtered
