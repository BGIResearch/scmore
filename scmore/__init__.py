"""scMORE v2 multi-dataset processing pipeline."""

import os

# Shared installations can be read-only; numba/scanpy otherwise fail at import.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/scmore_v2_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/scmore_v2_mpl_cache")

from .config import PipelineConfig, Thresholds

__all__ = ["PipelineConfig", "Thresholds"]
