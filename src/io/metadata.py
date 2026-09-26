"""Metadata representation and PDS4 XML validation for lunar orbiter imagery."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET
import numpy as np


@dataclass(frozen=True)
class PDS4ImageMetadata:
    """Immutable representation of PDS4 2D Image metadata."""

    sensor_name: str
    file_path: Path
    xml_path: Optional[Path]
    lines: int
    samples: int
    dtype: np.dtype
    byte_order: str
    file_size_bytes: int
    offset_bytes: int = 0
    pixel_resolution_m: Optional[float] = None

    @property
    def width(self) -> int:
        """Alias for samples (columns)."""
        return self.samples

    @property
    def height(self) -> int:
        """Alias for lines (rows)."""
        return self.lines

    @property
    def expected_file_size(self) -> int:
        """Calculates expected file size based on dimensions and data type."""
        bytes_per_elem = self.dtype.itemsize
        return self.offset_bytes + (self.lines * self.samples * bytes_per_elem)


# Mapping from PDS4 data_type string to numpy dtype
PDS4_DATA_TYPE_MAP: Dict[str, np.dtype] = {
    "UnsignedByte": np.dtype(np.uint8),
    "SignedByte": np.dtype(np.int8),
    "UnsignedLSB2": np.dtype("<u2"),
    "SignedLSB2": np.dtype("<i2"),
    "UnsignedMSB2": np.dtype(">u2"),
    "SignedMSB2": np.dtype(">i2"),
    "UnsignedLSB4": np.dtype("<u4"),
    "SignedLSB4": np.dtype("<i4"),
    "IEEE754LSBSingle": np.dtype("<f4"),
    "IEEE754MSBSingle": np.dtype(">f4"),
    "IEEE754LSBDouble": np.dtype("<f8"),
    "IEEE754MSBDouble": np.dtype(">f8"),
}


def parse_pds4_xml(xml_path: Path | str) -> Dict[str, Any]:
    """Parse a PDS4 XML label file and extract image array metadata.

    Args:
        xml_path: Path to the PDS4 .xml label.

    Returns:
        Dictionary containing extracted metadata fields.

    Raises:
        FileNotFoundError: If the XML file does not exist.
        ValueError: If required PDS4 image elements are missing.
    """
    path = Path(xml_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDS4 XML label not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    # Strip XML namespaces for straightforward element lookup
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    array_elem = root.find(".//Array_2D_Image")
    if array_elem is None:
        raise ValueError(f"No <Array_2D_Image> found in {path}")

    offset_elem = array_elem.find("offset")
    offset = int(offset_elem.text.strip()) if offset_elem is not None and offset_elem.text else 0

    data_type_elem = array_elem.find(".//data_type")
    if data_type_elem is None or not data_type_elem.text:
        raise ValueError(f"No <data_type> element found in {path}")
    raw_data_type = data_type_elem.text.strip()
    np_dtype = PDS4_DATA_TYPE_MAP.get(raw_data_type)
    if np_dtype is None:
        raise ValueError(f"Unrecognized PDS4 data_type '{raw_data_type}' in {path}")

    # Extract axes (Line and Sample)
    lines: Optional[int] = None
    samples: Optional[int] = None
    for axis in array_elem.findall(".//Axis_Array"):
        name_elem = axis.find("axis_name")
        elems_elem = axis.find("elements")
        if name_elem is not None and elems_elem is not None and name_elem.text and elems_elem.text:
            axis_name = name_elem.text.strip().lower()
            if axis_name == "line":
                lines = int(elems_elem.text.strip())
            elif axis_name == "sample":
                samples = int(elems_elem.text.strip())

    if lines is None or samples is None:
        raise ValueError(f"Failed to find both Line and Sample axes in {path}")

    # File size from label
    file_size_elem = root.find(".//file_size")
    file_size = int(file_size_elem.text.strip()) if file_size_elem is not None and file_size_elem.text else None

    # Pixel resolution (GSD)
    res_elem = root.find(".//pixel_resolution")
    pixel_res = float(res_elem.text.strip()) if res_elem is not None and res_elem.text else None

    return {
        "lines": lines,
        "samples": samples,
        "offset": offset,
        "raw_data_type": raw_data_type,
        "dtype": np_dtype,
        "file_size": file_size,
        "pixel_resolution_m": pixel_res,
    }


def validate_file_size(
    file_path: Path | str,
    expected_lines: int,
    expected_samples: int,
    bytes_per_pixel: int,
    offset_bytes: int = 0,
) -> int:
    """Validate that the actual file size matches the expected binary dimensions.

    Args:
        file_path: Path to the binary .img file.
        expected_lines: Number of lines (height).
        expected_samples: Number of samples (width).
        bytes_per_pixel: Bytes per data element (1 for uint8, 2 for uint16, etc.).
        offset_bytes: Header or prefix byte offset.

    Returns:
        Actual file size in bytes if valid.

    Raises:
        FileNotFoundError: If the binary file does not exist.
        ValueError: If the actual file size differs from the expected size.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Binary raster file not found: {path}")

    actual_size = path.stat().st_size
    expected_size = offset_bytes + (expected_lines * expected_samples * bytes_per_pixel)

    if actual_size != expected_size:
        raise ValueError(
            f"File size mismatch for {path.name}:\n"
            f"  Expected: {expected_size:,} bytes ({expected_lines:,} lines × {expected_samples:,} samples × {bytes_per_pixel} bytes + {offset_bytes} offset)\n"
            f"  Actual:   {actual_size:,} bytes\n"
            f"  Difference: {actual_size - expected_size:+,} bytes"
        )

    return actual_size


def check_metadata_discrepancies(
    config_dict: Dict[str, Any], xml_dict: Dict[str, Any]
) -> List[str]:
    """Cross-check configuration values against parsed PDS4 XML metadata.

    Returns:
        List of discrepancy descriptions (empty if perfectly consistent).
    """
    discrepancies: List[str] = []

    for key in ["lines", "samples"]:
        cfg_val = config_dict.get(key)
        xml_val = xml_dict.get(key)
        if cfg_val is not None and xml_val is not None and cfg_val != xml_val:
            discrepancies.append(
                f"Discrepancy for '{key}': Config={cfg_val} vs XML={xml_val}"
            )

    if "pixel_resolution_m" in config_dict and "pixel_resolution_m" in xml_dict:
        cfg_res = config_dict["pixel_resolution_m"]
        xml_res = xml_dict["pixel_resolution_m"]
        if cfg_res is not None and xml_res is not None and abs(cfg_res - xml_res) > 1e-4:
            discrepancies.append(
                f"Discrepancy for 'pixel_resolution_m': Config={cfg_res} vs XML={xml_res}"
            )

    return discrepancies
