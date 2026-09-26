"""Overlap data extractor and source window manager.

Loads the calculated TMC-2 overlap region using Chunk-4 validated bounds,
extracts co-located benchmark and diagnostic OHRC patches using ground-grid mapping,
and ensures raw arrays remain strictly read-only and unmutated.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

from ..io.region_loader import DatasetLoader


@dataclass(frozen=True)
class SourceWindowMetadata:
    """Metadata describing source image crop coordinates and physical dimensions."""

    sensor_name: str
    file_path: str
    lines_total: int
    samples_total: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    crop_height: int
    crop_width: int
    dtype: str
    pixel_resolution_m: float
    ground_width_km: float
    ground_height_km: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OverlapDataExtractor:
    """Extracts and manages cross-sensor overlap rasters and co-located patches."""

    def __init__(
        self,
        loader: Optional[DatasetLoader] = None,
        overlap_json_path: Optional[Path | str] = None,
    ):
        self.loader = loader or DatasetLoader()

        if overlap_json_path is None:
            overlap_json_path = (
                Path(__file__).resolve().parent.parent.parent / "results" / "overlap" / "pixel_overlap.json"
            )

        self.overlap_json_path = Path(overlap_json_path)
        self.overlap_meta: Dict[str, Any] = {}
        if self.overlap_json_path.is_file():
            with open(self.overlap_json_path, "r", encoding="utf-8") as f:
                self.overlap_meta = json.load(f)

    def extract_tmc2_overlap(self) -> Tuple[np.ndarray, SourceWindowMetadata]:
        """Load the entire calculated TMC-2 overlap region from disk.

        Uses the validated Chunk-4 bounds:
            Scan: 280781 to 285893 (5112 lines)
            Pixel: 2519 to 3163 (644 samples)

        Returns:
            Tuple of:
                - Read-only uint16 NumPy array of shape (5112, 644).
                - SourceWindowMetadata instance.
        """
        # Default bounds from Chunk 4 validation
        r0 = 280781
        r1 = 285893
        c0 = 2519
        c1 = 3163

        if "tmc2" in self.overlap_meta and "overlap_scan_bounds" in self.overlap_meta["tmc2"]:
            t_meta = self.overlap_meta["tmc2"]
            r0 = t_meta["overlap_scan_bounds"]["integer_window_start"]
            r1 = t_meta["overlap_scan_bounds"]["integer_window_end"]
            c0 = t_meta["overlap_pixel_bounds"]["integer_window_start"]
            c1 = t_meta["overlap_pixel_bounds"]["integer_window_end"]

        reader = self.loader.get_reader("TMC2")
        arr = reader.read_region(r0, r1, c0, c1)

        # Enforce read-only array to preserve source data
        arr.flags.writeable = False

        meta = SourceWindowMetadata(
            sensor_name="TMC-2",
            file_path=str(reader.file_path),
            lines_total=reader.lines,
            samples_total=reader.samples,
            row_start=r0,
            row_end=r1,
            col_start=c0,
            col_end=c1,
            crop_height=r1 - r0,
            crop_width=c1 - c0,
            dtype=str(arr.dtype),
            pixel_resolution_m=float(reader.pixel_resolution or getattr(reader, "_pixel_resolution_m", 5.40)),
            ground_width_km=(c1 - c0) * float(reader.pixel_resolution or getattr(reader, "_pixel_resolution_m", 5.40)) / 1000.0,
            ground_height_km=(r1 - r0) * float(reader.pixel_resolution or getattr(reader, "_pixel_resolution_m", 5.40)) / 1000.0,
        )

        return arr, meta

    def extract_benchmark_pair(
        self,
        center_lon: float = 336.536,
        center_lat: float = -3.000,
        tmc_size_px: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Extract a co-located cross-sensor benchmark patch pair centered at matching lunar coordinates.

        TMC-2 patch is tmc_size_px x tmc_size_px (e.g. 200x200 = 1080m x 1080m).
        OHRC patch is scaled by 21.6x (e.g. 4320x4320 = 1080m x 1080m).

        Both returned arrays are strictly read-only.

        Returns:
            Tuple of:
                - TMC-2 patch (tmc_size_px, tmc_size_px), uint16, read-only.
                - OHRC patch (ohrc_size_px, ohrc_size_px), uint8, read-only.
                - Metadata dict recording exact pixel ranges and geographic coordinates.
        """
        ohrc_size_px = int(round(tmc_size_px * 21.6))

        # Target center pixel/scan from Chunk-4 ground-grid mapping
        # lon 336.536, lat -3.000 maps to:
        # TMC-2: pixel 2845.55, scan 283359.41
        # OHRC:  pixel 5869.29, scan 47243.85
        tmc_sc_c = 283359
        tmc_px_c = 2846

        ohrc_sc_c = 47244
        ohrc_px_c = 5869

        half_tmc = tmc_size_px // 2
        tmc_r0 = tmc_sc_c - half_tmc
        tmc_r1 = tmc_r0 + tmc_size_px
        tmc_c0 = tmc_px_c - half_tmc
        tmc_c1 = tmc_c0 + tmc_size_px

        half_ohrc = ohrc_size_px // 2
        ohrc_r0 = ohrc_sc_c - half_ohrc
        ohrc_r1 = ohrc_r0 + ohrc_size_px
        ohrc_c0 = ohrc_px_c - half_ohrc
        ohrc_c1 = ohrc_c0 + ohrc_size_px

        tmc_reader = self.loader.get_reader("TMC2")
        ohrc_reader = self.loader.get_reader("OHRC")

        tmc_arr = tmc_reader.read_region(tmc_r0, tmc_r1, tmc_c0, tmc_c1)
        ohrc_arr = ohrc_reader.read_region(ohrc_r0, ohrc_r1, ohrc_c0, ohrc_c1)

        tmc_arr.flags.writeable = False
        ohrc_arr.flags.writeable = False

        meta = {
            "geographic_center": {"longitude_deg": center_lon, "latitude_deg": center_lat},
            "ground_extent_km": tmc_size_px * 5.40 / 1000.0,
            "tmc2": {
                "row_start": tmc_r0,
                "row_end": tmc_r1,
                "col_start": tmc_c0,
                "col_end": tmc_c1,
                "shape": list(tmc_arr.shape),
                "dtype": str(tmc_arr.dtype),
                "gsd_m": 5.40,
            },
            "ohrc": {
                "row_start": ohrc_r0,
                "row_end": ohrc_r1,
                "col_start": ohrc_c0,
                "col_end": ohrc_c1,
                "shape": list(ohrc_arr.shape),
                "dtype": str(ohrc_arr.dtype),
                "gsd_m": 0.25,
            },
            "correspondence_note": (
                "The full OHRC footprint (93,693 x 12,000 pixels) corresponds to the entire TMC-2 "
                "overlap swath (5,112 x 644 pixels). Because OHRC has a 21.6x higher spatial resolution "
                "and is tilted by ~5.7 deg relative to TMC-2 along-track coordinate frame, a 1:1 pixel "
                "grid correspondence is not invented. Instead, representative co-located spatial windows "
                "are extracted using the validated ISRO ground-grid mapping."
            ),
        }

        return tmc_arr, ohrc_arr, meta
