"""Generic region-reading interface and DatasetLoader for Chandrayaan-2 imagery."""

from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np

from ..utils.config import AppConfig, load_config
from .raster_reader import BaseRasterReader, OHRCRasterReader, TMC2RasterReader


class DatasetLoader:
    """Manages memory-efficient readers for a Chandrayaan-2 pair."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        config_path: Optional[Path | str] = None,
        validate_xml: bool = True,
    ):
        if config is None:
            if config_path is None:
                # Default path
                config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
            self.config = load_config(config_path)
        else:
            self.config = config

        self._root_dir = self.config.dataset.root_dir
        self._validate_xml = validate_xml
        self._readers: Dict[str, BaseRasterReader] = {}

    def get_reader(self, sensor: str) -> BaseRasterReader:
        """Get or lazily instantiate the reader for the specified sensor."""
        sensor_key = self._normalize_sensor_name(sensor)
        if sensor_key not in self._readers:
            if sensor_key == "OHRC":
                cfg = self.config.dataset.ohrc
                img_path = cfg.get_img_path(self._root_dir)
                xml_path = cfg.get_xml_path(self._root_dir)
                reader = OHRCRasterReader(
                    file_path=img_path,
                    xml_path=xml_path,
                    lines=cfg.lines,
                    samples=cfg.samples,
                    pixel_resolution_m=cfg.pixel_resolution_m,
                    validate_size_on_init=True,
                    validate_xml_on_init=self._validate_xml,
                )
            elif sensor_key == "TMC2":
                cfg = self.config.dataset.tmc2
                img_path = cfg.get_img_path(self._root_dir)
                xml_path = cfg.get_xml_path(self._root_dir)
                reader = TMC2RasterReader(
                    file_path=img_path,
                    xml_path=xml_path,
                    lines=cfg.lines,
                    samples=cfg.samples,
                    pixel_resolution_m=cfg.pixel_resolution_m,
                    validate_size_on_init=True,
                    validate_xml_on_init=self._validate_xml,
                )
            else:
                raise ValueError(
                    f"Unsupported sensor '{sensor}'. Supported sensors: 'OHRC', 'TMC2' (or 'TMC')"
                )
            self._readers[sensor_key] = reader

        return self._readers[sensor_key]

    def read_region(
        self,
        sensor: str,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> np.ndarray:
        """Read a rectangular region from the requested sensor image.

        Args:
            sensor: Sensor identifier ('OHRC' or 'TMC2' / 'TMC').
            row_start: Starting line index (inclusive).
            row_end: Ending line index (exclusive).
            col_start: Starting sample index (inclusive).
            col_end: Ending sample index (exclusive).

        Returns:
            Normal NumPy array copy of the region.
        """
        reader = self.get_reader(sensor)
        return reader.read_region(
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end,
        )

    def close(self) -> None:
        """Close all open raster readers."""
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> "DatasetLoader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    def _normalize_sensor_name(sensor: str) -> str:
        s = sensor.strip().upper().replace("-", "")
        if s in ["OHRC"]:
            return "OHRC"
        elif s in ["TMC", "TMC2"]:
            return "TMC2"
        raise ValueError(
            f"Unknown sensor: '{sensor}'. Must be 'OHRC', 'TMC', or 'TMC-2'"
        )


# Global default loader instance (lazily initialized)
_DEFAULT_LOADER: Optional[DatasetLoader] = None


def get_default_loader(config_path: Optional[Path | str] = None) -> DatasetLoader:
    """Retrieve or initialize the global DatasetLoader."""
    global _DEFAULT_LOADER
    if _DEFAULT_LOADER is None:
        _DEFAULT_LOADER = DatasetLoader(config_path=config_path)
    return _DEFAULT_LOADER


def read_region(
    sensor: str,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    config: Optional[AppConfig] = None,
    config_path: Optional[Path | str] = None,
) -> np.ndarray:
    """Convenience function to read a region from a specified sensor.

    Example:
        crop = read_region("OHRC", 1000, 1500, 2000, 2500)
    """
    if config is not None or config_path is not None:
        loader = DatasetLoader(config=config, config_path=config_path)
        with loader:
            return loader.read_region(sensor, row_start, row_end, col_start, col_end)
    else:
        loader = get_default_loader()
        return loader.read_region(sensor, row_start, row_end, col_start, col_end)
