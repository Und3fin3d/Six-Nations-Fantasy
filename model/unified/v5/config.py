"""Frozen configuration contracts for unified model v5.

Every constant here is fixed a priori per the v5 research plan
(`data/unified/v5/RESEARCH_PLAN.md` §4, ledger L8/L9): tree settings come from
the v3 search-winning baseline (`data/unified/v3/search/best_baseline.json`);
shrinkage constants are incumbent-inherited and sensitivity-checked, never
tuned against folds one value at a time.

Corrective-cycle update (see `data/unified/v5/corrective_cycle.md`): the
frozen v3-protocol benchmark fired kill-switch K1 (the EB shrunk-rate layer
delivered zero low-history-slice gain: 11.67 vs baseline 11.66) and showed the
EB + slot layers compress the NCR top-10 boundary against try-scoring backs
(top-10 capture 44.6% vs baseline 56.4% at equal MAE/Spearman). Per the plan's
precommitted ladder, the candidate reverts to **terminal rung T = v1 + guards
+ attribution masking**: the EB and slot layers are now OFF by default (still
config-gated for ablation forensics), and `lineouts_won` masking is ON by
default on the row-level evidence documented in the corrective-cycle note.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class V5Config:
    """One immutable configuration for :class:`ShrunkFormGBDT`."""

    # --- LightGBM head settings (frozen from the v3 search-winning baseline).
    learning_rate: float = 0.045
    n_estimators: int = 180
    num_leaves: int = 23
    min_child_samples: int = 35
    reg_lambda: float = 4.0
    subsample: float = 0.85
    colsample_bytree: float = 0.8
    weighting: str = "natural"
    time_half_life_days: float | None = None
    seed: int = 17

    # --- Empirical-Bayes shrinkage layer (a-priori constants; plan §4).
    shrinkage_minutes_k: float = 220.0
    club_confidence: float = 0.55
    ewm_halflife_matches: float = 4.0
    cal_clip_lo: float = 0.5
    cal_clip_hi: float = 1.5
    cal_shrink_minutes: float = 4000.0
    dual_min_minutes: float = 160.0
    prior_shrink_minutes: float = 5000.0

    # --- Rare-event guards (fixed a priori; plan ledger L8).
    min_valid_rows: int = 20
    min_positives: int = 5

    # --- Feature switches. Post-benchmark default = terminal rung T: the EB
    # shrunk-rate layer (K1: no low-history gain; NCR top-10 boundary harm)
    # and the slot-minutes layer are OFF unless explicitly re-enabled for
    # ablation. `keep_legacy_form_features` stays on (v1 parity).
    keep_legacy_form_features: bool = True
    use_eb_features: bool = False
    use_slot_minutes: bool = False
    context_blocks: tuple[str, ...] = ()

    # --- Data-quality switches. `mask_attribution_corrupted` now defaults on:
    # the row-level NCR evidence (hooker-credited `lineouts_won` inflating
    # predicted top-10s; the incumbent drops the same artifact and leads
    # capture by ~6pp over every unified variant) satisfies the plan's E1
    # support threshold for the terminal rung. Config-gated; recorded in every
    # artifact manifest via `to_dict`.
    mask_attribution_corrupted: bool = True
    attribution_events: tuple[str, ...] = ("lineouts_won",)
    attach_potm_labels: bool = False

    def excluded_events(self) -> tuple[str, ...]:
        if not self.mask_attribution_corrupted:
            return ()
        return tuple(self.attribution_events)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["context_blocks"] = list(self.context_blocks)
        payload["attribution_events"] = list(self.attribution_events)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "V5Config":
        data = dict(payload)
        data["context_blocks"] = tuple(data.get("context_blocks", ()))
        data["attribution_events"] = tuple(
            data.get("attribution_events", ("lineouts_won",))
        )
        return cls(**data)
