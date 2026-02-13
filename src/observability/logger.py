"""Minimal logging setup.

Stage A3 only needs a basic logger that writes to stderr.
This will be expanded later during observability work.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str = "modular-rag", level: Optional[str] = None) -> logging.Logger:
    """Create (or return) a configured logger.

    Args:
        name: Logger name.
        level: Optional log level string (e.g. "INFO"). If omitted, keeps existing.

    Returns:
        Configured logger writing to stderr.
    """

    logger = logging.getLogger(name)
    logger.propagate = False

    if level is not None:
        logger.setLevel(level.upper())
    elif logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(
            logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)

    return logger
