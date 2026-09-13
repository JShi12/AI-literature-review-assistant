"""Central logging configuration for the application."""

from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    """Configure root logging once, controlled by the LOG_LEVEL environment variable."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
