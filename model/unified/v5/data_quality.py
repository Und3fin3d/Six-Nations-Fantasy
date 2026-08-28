"""Global data-quality transforms for the unified store (v5).

These are *global* operations justified by feed-attribution evidence, not
competition-specific calibrators:

* :func:`mask_events_globally` — mark events whose feed attribution is
  position-corrupted (candidate: ``lineouts_won``, hooker-credited in the
  underlying API feed; see `data/unified/v5/phase1_failure_diagnosis.md` F7)
  as unavailable for every row. With the label unavailable, no head is fit
  and predictions omit the event; the deterministic scorers then read zero
  via the existing ``_event`` default. The 6N scorer never used
  ``lineouts_won``, so the change is NCR-neutral-to-positive and 6N-inert.
* :func:`attach_ncr_potm` — attach the existing ``data/ncr/ncr_potm.csv``
  player-of-the-match labels to the canonical store. One winner per fixture
  makes every other row in a covered fixture a true zero. This is a
  reconstruction-fidelity (diagnostic) improvement, plan ledger L6: with six
  GW2 positives it cannot move the potm head materially.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data import ROOT

DATA = ROOT / "data"
DEFAULT_POTM = DATA / "ncr" / "ncr_potm.csv"


def mask_events_globally(frame: pd.DataFrame, events: tuple[str, ...]) -> pd.DataFrame:
    """Mark ``events`` unavailable on every row (returns a copy)."""
    out = frame.copy()
    for event in events:
        if event in out:
            out[event] = np.nan
        out[f"available__{event}"] = False
    return out


def attach_ncr_potm(
    frame: pd.DataFrame, path: Path = DEFAULT_POTM,
) -> tuple[pd.DataFrame, dict]:
    """Attach POTM labels from the existing NCR potm file.

    Winner rows get ``potm = 1``; every other row in a covered fixture gets
    ``potm = 0``; rows in fixtures absent from the file are untouched.
    Returns the updated frame and a small audit payload.
    """
    out = frame.copy()
    if "potm" not in out:
        out["potm"] = np.nan
    if "available__potm" not in out:
        out["available__potm"] = False
    report = {"winners": 0, "covered_fixtures": 0, "rows_labelled": 0,
              "winner_rows_unmatched": 0, "path": str(path)}
    if not path.exists():
        return out, report
    potm = pd.read_csv(path)
    potm["fixture_id"] = potm["fixture_id"].astype(str)
    potm["player_id"] = potm["player_id"].astype(str)
    out["fixture_id"] = out["fixture_id"].astype(str)
    out["player_id"] = out["player_id"].astype(str)

    winners = set(zip(potm["fixture_id"], potm["player_id"]))
    covered_fixtures = set(potm["fixture_id"])
    in_covered = out["fixture_id"].isin(covered_fixtures)
    keys = list(zip(out["fixture_id"], out["player_id"]))
    is_winner = pd.Series([key in winners for key in keys], index=out.index)

    matched_winners = in_covered & is_winner
    report["winners"] = int(matched_winners.sum())
    report["winner_rows_unmatched"] = int(len(winners) - matched_winners.sum())
    report["covered_fixtures"] = int(len(covered_fixtures))
    labelled = in_covered
    out.loc[labelled, "potm"] = np.where(is_winner[labelled], 1.0, 0.0)
    out.loc[labelled, "available__potm"] = True
    report["rows_labelled"] = int(labelled.sum())
    return out, report
