"""
Dataset loading module

In real system:
- reads Sentinel-2 data
- constructs time series
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("map.dataset.loader")


def load_dataset(config: Any) -> str:
    """Load training dataset"""
    logger.info("Loading AMD dataset")
    return 'raw_data'


def load_new_data(config: Any) -> str:
    """Load inference dataset"""
    logger.info("Loading new AMD data")
    return 'new_data'
