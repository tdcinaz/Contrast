from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(threadName)s] %(name)s: %(message)s"

# Modules can be imported by tests or utility scripts before main() configures output.
logging.getLogger("contrast").addHandler(logging.NullHandler())


def configure_logging(level_name: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the Contrast logger without changing unrelated library loggers."""
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported log level: {level_name}")

    logger = logging.getLogger("contrast")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def install_exception_logging() -> None:
    original_hook = sys.excepthook

    def log_uncaught_exception(
        exception_type: type[BaseException],
        exception: BaseException,
        traceback: object,
    ) -> None:
        if issubclass(exception_type, KeyboardInterrupt):
            original_hook(exception_type, exception, traceback)
            return
        logging.getLogger("contrast").critical("Unhandled exception", exc_info=(exception_type, exception, traceback))

    sys.excepthook = log_uncaught_exception