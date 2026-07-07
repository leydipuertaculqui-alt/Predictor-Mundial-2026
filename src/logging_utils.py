import logging
import sys
from pathlib import Path

from src import config as cfg


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    cfg.ensure_directories()
    logger = logging.getLogger("mundial2026_predictor")
    if logger.handlers:
        logger.setLevel(level)
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    log_file = cfg.LOGS / "pipeline.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"mundial2026_predictor.{name}")
