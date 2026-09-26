"""Unit tests for logger setup."""

import logging
from pathlib import Path
import sys
import tempfile
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import setup_logger, close_logger


class TestLogging(unittest.TestCase):
    def test_setup_logger_console_only(self):
        logger = setup_logger(name="TestLoggerConsole", level="DEBUG")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.level, logging.DEBUG)
        self.assertGreaterEqual(len(logger.handlers), 1)
        close_logger(logger)

    def test_setup_logger_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir) / "logs"
            logger = setup_logger(
                name="TestLoggerFile",
                level="INFO",
                log_dir=log_dir,
                log_filename="test.log",
            )
            logger.info("Test log message")
            log_file = log_dir / "test.log"
            self.assertTrue(log_file.is_file())
            content = log_file.read_text(encoding="utf-8")
            self.assertIn("Test log message", content)
            close_logger(logger)


if __name__ == "__main__":
    unittest.main()
