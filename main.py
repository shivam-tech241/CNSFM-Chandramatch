"""Main CLI entry point for CNSFM-ChandraMatch.

Chunk 1: Project Skeleton & Configuration Verification.
"""

import argparse
from pathlib import Path
import sys

from src.utils.config import load_config
from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CNSFM-ChandraMatch: Crater Neighborhood Structure Feature Matching Reproduction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.is_file():
        print(f"Error: Config file not found at {config_path}")
        return 1

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    logger = setup_logger(
        name="CNSFM-Main",
        level=config.logging.level,
        log_dir=Path(config.logging.log_dir) if config.logging.log_to_file else None,
    )

    logger.info("==================================================")
    logger.info("CNSFM-ChandraMatch Initialized (Chunk 1 Skeleton)")
    logger.info("==================================================")
    logger.info(f"Loaded config from: {config_path.resolve()}")
    logger.info(f"Dataset root: {config.dataset.root_dir}")

    # Verify external paths exist without loading heavy data
    root_exists = config.dataset.root_dir.is_dir()
    logger.info(f"Dataset root exists: {root_exists}")

    ohrc_img = config.dataset.ohrc.get_img_path(config.dataset.root_dir)
    tmc2_img = config.dataset.tmc2.get_img_path(config.dataset.root_dir)

    logger.info(f"OHRC img target: {ohrc_img} (exists: {ohrc_img.is_file()})")
    logger.info(f"TMC-2 img target: {tmc2_img} (exists: {tmc2_img.is_file()})")

    logger.info("Chunk 1 skeleton verified successfully.")
    logger.info("Pipeline stages (I/O, Detection, CNSF, Matching, MCR, Registration, Evaluation) awaiting subsequent chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
