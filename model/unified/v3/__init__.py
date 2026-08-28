"""Leak-free unified rugby supermodel v3.

The v3 package leaves production incumbents untouched.  It owns a versioned
evaluation harness, data audits, exposure-aware raw-event models, bounded model
search, retrospective benchmarking, and immutable prospective shadow outputs.
"""

from .config import GBDTV3Config, NeuralV3Config
from .gbdt import ExposureRateGBDT
from .neural import ExposureRateNeural

__all__ = [
    "ExposureRateGBDT",
    "ExposureRateNeural",
    "GBDTV3Config",
    "NeuralV3Config",
]
