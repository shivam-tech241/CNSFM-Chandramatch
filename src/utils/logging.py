"""Logging utility for CNSFM-ChandraMatch."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "CNSFM",
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_filename: str = "cnsfm.log",
) -> logging.Logger:
    """Set up and configure a structured logger.

    Args:
        name: Name of the logger.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory where log file will be saved. If None, only logs to console.
        log_filename: Name of the log file.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / log_filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def close_logger(logger: logging.Logger) -> None:
    """Close and remove all handlers from a logger to release file locks."""
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
