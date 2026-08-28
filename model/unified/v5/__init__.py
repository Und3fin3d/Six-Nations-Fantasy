"""Unified rugby supermodel v5 — shrunk-form direct-totals GBDT.

The v5 package is additive: it leaves the v1 universal control, the v3
package, both specialist incumbents, `contracts.py` and `scoring.py`
untouched. It implements Family A of `data/unified/v5/RESEARCH_PLAN.md`:
direct per-event totals (the v1 target structure) with an empirical-Bayes
shrunk feature layer, rare-event guards, and config-gated global
attribution-quality data transforms.
"""

from .config import V5Config
from .data_quality import attach_ncr_potm, mask_events_globally
from .metrics import (
    ablated_expected_points, first_cap_club_rich_mask, low_history_mask,
    paired_bootstrap_ci, position_bias, subgroup_mae,
)
from .model import ShrunkFormGBDT, V5FeatureEncoder, training_weights
from .shrinkage import ShrunkTables, add_shrunk_features, fit_tables

__all__ = [
    "ShrunkFormGBDT",
    "ShrunkTables",
    "V5Config",
    "V5FeatureEncoder",
    "ablated_expected_points",
    "add_shrunk_features",
    "attach_ncr_potm",
    "first_cap_club_rich_mask",
    "fit_tables",
    "low_history_mask",
    "mask_events_globally",
    "paired_bootstrap_ci",
    "position_bias",
    "subgroup_mae",
    "training_weights",
]
