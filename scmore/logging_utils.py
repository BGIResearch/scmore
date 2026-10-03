from __future__ import annotations

from contextlib import contextmanager
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import time


LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | dataset=%(dataset)s | "
    "pid=%(process)d | %(name)s | %(message)s"
)


class _DatasetFilter(logging.Filter):
    def __init__(self, dataset: str):
        super().__init__()
        self.dataset = dataset

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "dataset"):
            record.dataset = self.dataset
        return True


def configure_logging(
    log_file: Path,
    dataset: str = "-",
    level: str = "INFO",
    console: bool = True,
) -> logging.LoggerAdapter:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("scmore_v2")
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(LOG_FORMAT)
    dataset_filter = _DatasetFilter(dataset)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(dataset_filter)
    logger.addHandler(file_handler)
    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        stream.addFilter(dataset_filter)
        logger.addHandler(stream)
    return logging.LoggerAdapter(logger, {"dataset": dataset})


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"scmore_v2.{name}")


@contextmanager
def logged_stage(logger, stage: str):
    started = time.perf_counter()
    logger.info("stage_start stage=%s", stage)
    try:
        yield
    except BaseException:
        logger.exception(
            "stage_failed stage=%s elapsed_seconds=%.3f",
            stage,
            time.perf_counter() - started,
        )
        raise
    else:
        logger.info(
            "stage_complete stage=%s elapsed_seconds=%.3f",
            stage,
            time.perf_counter() - started,
        )


def run_stage(logger, stage: str, function, *args, **kwargs):
    with logged_stage(logger, stage):
        return function(*args, **kwargs)
