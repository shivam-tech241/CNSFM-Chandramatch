"""Configuration system for CNSFM-ChandraMatch."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


@dataclass
class SensorConfig:
    data_dir: str
    img_filename: str
    xml_filename: str
    lines: int
    samples: int
    pixel_resolution_m: float
    dtype: str
    endian: str
    sun_azimuth_deg: float
    solar_incidence_deg: float

    def get_img_path(self, root_dir: Path) -> Path:
        return root_dir / self.data_dir / self.img_filename

    def get_xml_path(self, root_dir: Path) -> Path:
        return root_dir / self.data_dir / self.xml_filename


@dataclass
class DatasetConfig:
    root_dir: Path
    ohrc: SensorConfig
    tmc2: SensorConfig


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "results/logs"


@dataclass
class AppConfig:
    dataset: DatasetConfig
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    raw_config: Dict[str, Any] = field(default_factory=dict)


def load_config(config_path: Path | str) -> AppConfig:
    """Load and parse YAML experiment configuration.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        AppConfig populated instance.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    ds_raw = data.get("dataset", {})
    root_dir = Path(ds_raw.get("root_dir", ""))

    ohrc_raw = ds_raw.get("ohrc", {})
    ohrc_cfg = SensorConfig(
        data_dir=ohrc_raw.get("data_dir", ""),
        img_filename=ohrc_raw.get("img_filename", ""),
        xml_filename=ohrc_raw.get("xml_filename", ""),
        lines=int(ohrc_raw.get("lines", 0)),
        samples=int(ohrc_raw.get("samples", 0)),
        pixel_resolution_m=float(ohrc_raw.get("pixel_resolution_m", 0.0)),
        dtype=ohrc_raw.get("dtype", "uint8"),
        endian=ohrc_raw.get("endian", "none"),
        sun_azimuth_deg=float(ohrc_raw.get("sun_azimuth_deg", 0.0)),
        solar_incidence_deg=float(ohrc_raw.get("solar_incidence_deg", 0.0)),
    )

    tmc2_raw = ds_raw.get("tmc2", {})
    tmc2_cfg = SensorConfig(
        data_dir=tmc2_raw.get("data_dir", ""),
        img_filename=tmc2_raw.get("img_filename", ""),
        xml_filename=tmc2_raw.get("xml_filename", ""),
        lines=int(tmc2_raw.get("lines", 0)),
        samples=int(tmc2_raw.get("samples", 0)),
        pixel_resolution_m=float(tmc2_raw.get("pixel_resolution_m", 0.0)),
        dtype=tmc2_raw.get("dtype", "uint16"),
        endian=tmc2_raw.get("endian", "little"),
        sun_azimuth_deg=float(tmc2_raw.get("sun_azimuth_deg", 0.0)),
        solar_incidence_deg=float(tmc2_raw.get("solar_incidence_deg", 0.0)),
    )

    dataset_cfg = DatasetConfig(root_dir=root_dir, ohrc=ohrc_cfg, tmc2=tmc2_cfg)

    log_raw = data.get("logging", {})
    logging_cfg = LoggingConfig(
        level=log_raw.get("level", "INFO"),
        log_to_file=bool(log_raw.get("log_to_file", True)),
        log_dir=log_raw.get("log_dir", "results/logs"),
    )

    return AppConfig(
        dataset=dataset_cfg,
        logging=logging_cfg,
        raw_config=data,
    )
