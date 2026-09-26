"""Memory-efficient, region-based raster readers for Chandrayaan-2 lunar imagery."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from .metadata import (
    PDS4ImageMetadata,
    parse_pds4_xml,
    validate_file_size,
    check_metadata_discrepancies,
)


class BaseRasterReader(ABC):
    """Abstract base class for memory-mapped, region-based raster reading."""

    def __init__(
        self,
        file_path: Path | str,
        lines: int,
        samples: int,
        dtype: np.dtype,
        byte_order: str,
        offset_bytes: int = 0,
        pixel_resolution_m: Optional[float] = None,
        xml_path: Optional[Path | str] = None,
        sensor_name: str = "GENERIC",
        validate_size_on_init: bool = True,
        validate_xml_on_init: bool = True,
    ):
        self.file_path = Path(file_path)
        self.xml_path = Path(xml_path) if xml_path is not None else None
        self._lines = int(lines)
        self._samples = int(samples)
        self._dtype = np.dtype(dtype)
        self._byte_order = byte_order
        self._offset_bytes = int(offset_bytes)
        self._pixel_resolution_m = pixel_resolution_m
        self._sensor_name = sensor_name

        if validate_size_on_init:
            self._file_size = validate_file_size(
                file_path=self.file_path,
                expected_lines=self._lines,
                expected_samples=self._samples,
                bytes_per_pixel=self._dtype.itemsize,
                offset_bytes=self._offset_bytes,
            )
        else:
            self._file_size = self.file_path.stat().st_size if self.file_path.is_file() else 0

        # XML cross-validation
        if validate_xml_on_init and self.xml_path is not None and self.xml_path.is_file():
            xml_meta = parse_pds4_xml(self.xml_path)
            discrepancies = check_metadata_discrepancies(
                {
                    "lines": self._lines,
                    "samples": self._samples,
                    "pixel_resolution_m": self._pixel_resolution_m,
                },
                xml_meta,
            )
            if discrepancies:
                raise ValueError(
                    f"Metadata discrepancy detected between config and XML for {self._sensor_name}:\n"
                    + "\n".join(f"  - {d}" for d in discrepancies)
                )

        self._mmap: Optional[np.memmap] = None

    @property
    def width(self) -> int:
        """Width (number of samples/columns)."""
        return self._samples

    @property
    def samples(self) -> int:
        """Number of samples (columns)."""
        return self._samples

    @property
    def height(self) -> int:
        """Height (number of lines/rows)."""
        return self._lines

    @property
    def lines(self) -> int:
        """Number of lines (rows)."""
        return self._lines

    @property
    def dtype(self) -> np.dtype:
        """Data type of the raster elements."""
        return self._dtype

    @property
    def byte_order(self) -> str:
        """Byte order ('little', 'big', 'none')."""
        return self._byte_order

    @property
    def file_size(self) -> int:
        """File size in bytes."""
        return self._file_size

    @property
    def pixel_resolution(self) -> Optional[float]:
        """Ground sample distance in meters/pixel."""
        return self._pixel_resolution_m

    @property
    def sensor_name(self) -> str:
        """Sensor identifier."""
        return self._sensor_name

    @property
    def metadata(self) -> PDS4ImageMetadata:
        """Return structured metadata representation."""
        return PDS4ImageMetadata(
            sensor_name=self._sensor_name,
            file_path=self.file_path,
            xml_path=self.xml_path,
            lines=self._lines,
            samples=self._samples,
            dtype=self._dtype,
            byte_order=self._byte_order,
            file_size_bytes=self._file_size,
            offset_bytes=self._offset_bytes,
            pixel_resolution_m=self._pixel_resolution_m,
        )

    def _get_mmap(self) -> np.memmap:
        """Get or lazily initialize the read-only memory map."""
        if self._mmap is None or self._mmap._mmap.closed:  # type: ignore[attr-defined]
            self._mmap = np.memmap(
                filename=str(self.file_path),
                dtype=self._dtype,
                mode="r",
                offset=self._offset_bytes,
                shape=(self._lines, self._samples),
            )
        return self._mmap

    def validate_region_bounds(
        self, row_start: int, row_end: int, col_start: int, col_end: int
    ) -> None:
        """Validate rectangular region boundaries strictly.

        Raises:
            ValueError: If boundaries are invalid, inverted, or out of bounds.
        """
        if row_start < 0:
            raise ValueError(f"Invalid row_start={row_start}: must be non-negative (>= 0)")
        if col_start < 0:
            raise ValueError(f"Invalid col_start={col_start}: must be non-negative (>= 0)")
        if row_start >= row_end:
            raise ValueError(
                f"Invalid row range [{row_start}, {row_end}): row_start must be strictly less than row_end"
            )
        if col_start >= col_end:
            raise ValueError(
                f"Invalid col range [{col_start}, {col_end}): col_start must be strictly less than col_end"
            )
        if row_end > self._lines:
            raise ValueError(
                f"Requested row_end={row_end} exceeds image height (lines={self._lines})"
            )
        if col_end > self._samples:
            raise ValueError(
                f"Requested col_end={col_end} exceeds image width (samples={self._samples})"
            )

    def read_region(
        self, row_start: int, row_end: int, col_start: int, col_end: int
    ) -> np.ndarray:
        """Read a rectangular region from the image using read-only memory mapping.

        Args:
            row_start: Starting line index (inclusive, 0-indexed).
            row_end: Ending line index (exclusive).
            col_start: Starting sample index (inclusive, 0-indexed).
            col_end: Ending sample index (exclusive).

        Returns:
            Normal NumPy array copy of the requested region.
        """
        self.validate_region_bounds(row_start, row_end, col_start, col_end)
        mmap = self._get_mmap()
        # Return an explicit, independent in-memory copy
        crop = np.array(mmap[row_start:row_end, col_start:col_end], copy=True)
        return crop

    def close(self) -> None:
        """Release memory map resources."""
        if self._mmap is not None:
            try:
                # In Windows, deleting the memmap reference releases the file handle
                del self._mmap
            except Exception:
                pass
            self._mmap = None

    def __enter__(self) -> "BaseRasterReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class OHRCRasterReader(BaseRasterReader):
    """Memory-mapped region reader for Chandrayaan-2 OHRC imagery.

    Default specifications:
    - lines: 93693
    - samples: 12000
    - dtype: uint8 (1 byte per element)
    - pixel_resolution: 0.25 m/pixel
    - expected file size: 1,124,316,000 bytes
    """

    DEFAULT_LINES: int = 93693
    DEFAULT_SAMPLES: int = 12000
    DEFAULT_DTYPE: np.dtype = np.dtype(np.uint8)
    DEFAULT_PIXEL_RES: float = 0.25

    def __init__(
        self,
        file_path: Path | str,
        xml_path: Optional[Path | str] = None,
        lines: int = DEFAULT_LINES,
        samples: int = DEFAULT_SAMPLES,
        pixel_resolution_m: float = DEFAULT_PIXEL_RES,
        validate_size_on_init: bool = True,
        validate_xml_on_init: bool = True,
    ):
        super().__init__(
            file_path=file_path,
            lines=lines,
            samples=samples,
            dtype=self.DEFAULT_DTYPE,
            byte_order="none",
            offset_bytes=0,
            pixel_resolution_m=pixel_resolution_m,
            xml_path=xml_path,
            sensor_name="OHRC",
            validate_size_on_init=validate_size_on_init,
            validate_xml_on_init=validate_xml_on_init,
        )


class TMC2RasterReader(BaseRasterReader):
    """Memory-mapped region reader for Chandrayaan-2 TMC-2 imagery.

    Default specifications:
    - lines: 295234
    - samples: 4000
    - dtype: uint16 little-endian ('<u2', 2 bytes per element)
    - pixel_resolution: 5.40 m/pixel
    - expected file size: 2,361,872,000 bytes
    """

    DEFAULT_LINES: int = 295234
    DEFAULT_SAMPLES: int = 4000
    DEFAULT_DTYPE: np.dtype = np.dtype("<u2")
    DEFAULT_PIXEL_RES: float = 5.40

    def __init__(
        self,
        file_path: Path | str,
        xml_path: Optional[Path | str] = None,
        lines: int = DEFAULT_LINES,
        samples: int = DEFAULT_SAMPLES,
        pixel_resolution_m: float = DEFAULT_PIXEL_RES,
        validate_size_on_init: bool = True,
        validate_xml_on_init: bool = True,
    ):
        super().__init__(
            file_path=file_path,
            lines=lines,
            samples=samples,
            dtype=self.DEFAULT_DTYPE,
            byte_order="little",
            offset_bytes=0,
            pixel_resolution_m=pixel_resolution_m,
            xml_path=xml_path,
            sensor_name="TMC-2",
            validate_size_on_init=validate_size_on_init,
            validate_xml_on_init=validate_xml_on_init,
        )
