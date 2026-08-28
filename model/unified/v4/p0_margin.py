"""P0.3 — pinned picker-sensitivity protocol for the non-inferiority margin.

Precommitted (red-team correction #4) and frozen once computed:
- Fixed squad: the frozen v3 baseline predictions on the Six Nations 2025
  selection folds (data/unified/v3/benchmark/predictions.csv).
- Perturbation: zero-mean Gaussian noise scaled so its expected absolute value
  equals sigma, for sigma on the predeclared grid {0.1, 0.2, ..., 1.0} points.
- 200 seeded replications per sigma (seeds 0..199).
- A replication "changes the squad" when any round's quota-picked XV differs by
  at least one player from the unperturbed XV.
- Margin = the smallest sigma whose change rate is >= 50%. MAE differences
  below this margin cannot alter squad selection often enough to matter, so
  they are decision-irrelevant for tier-2 non-inferiority.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..data import ROOT
from model.evaluate import XV_QUOTA

DATA = ROOT / "data"
OUT = DATA / "unified" / "v4"
BENCH = DATA / "unified" / "v3" / "benchmark"

SIGMA_GRID = tuple(round(s, 1) for s in np.arange(0.1, 1.01, 0.1))
REPS = 200
CANONICAL = {
    "Prop": "Prop", "Hooker": "Hooker", "Lock": "Second-row",
    "Second-row": "Second-row", "Back-row": "Back-row",
    "Loose-forward": "Back-row", "Flanker": "Back-row", "Number-8": "Back-row",
    "Scrum-half": "Scrum-half", "Fly-half": "Fly-half",
    "Centre": "Centre", "Back-three": "Back-three",
}


def _fixed_squad() -> pd.DataFrame:
    predictions = pd.read_csv(BENCH / "predictions.csv")
    squad = predictions[
        predictions["engine"].eq("baseline") & predictions["phase"].eq("selection")
    ].copy()
    squad["canonical_pos"] = squad["position"].map(CANONICAL)
    unmapped = squad[squad["canonical_pos"].isna()]["position"].unique()
    if len(unmapped):
        raise ValueError(f"unmapped positions: {unmapped}")
    return squad


def _picked_ids(block: pd.DataFrame, score_col: str) -> frozenset:
    picked = []
    for pos, quota in XV_QUOTA.items():
        group = block[block["canonical_pos"] == pos]
        if not group.empty:
            picked.extend(group.nlargest(min(quota, len(group)), score_col)["key_player"])
    return frozenset(picked)


def main() -> None:
    squad = _fixed_squad()
    rounds = {r: block.reset_index(drop=True) for r, block in squad.groupby("round")}
    base_picks = {r: _picked_ids(block, "predicted_points") for r, block in rounds.items()}
    rows = []
    margin = None
    noise_scale = float(np.sqrt(np.pi / 2.0))  # E|N(0, s)| = s*sqrt(2/pi)
    for sigma in SIGMA_GRID:
        changed = 0
        for seed in range(REPS):
            rng = np.random.default_rng(1_000_000 * int(sigma * 10) + seed)
            rep_changed = False
            for r, block in rounds.items():
                noisy = block.copy()
                noisy["noisy_points"] = (
                    noisy["predicted_points"]
                    + rng.normal(0.0, sigma * noise_scale, len(noisy))
                )
                if _picked_ids(noisy, "noisy_points") != base_picks[r]:
                    rep_changed = True
                    break
            changed += rep_changed
        rate = changed / REPS
        rows.append({"sigma_mae_equivalent": sigma, "change_rate": rate})
        if margin is None and rate >= 0.5:
            margin = sigma
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "p0_margin_sensitivity.csv", index=False)
    payload = {
        "protocol": "fixed 6N-2025 baseline squad; Gaussian noise with E|noise|=sigma; "
                    "grid 0.1..1.0; 200 seeded reps; margin = smallest sigma with "
                    ">=50% squad-change rate",
        "sensitivity": rows,
        "derived_margin_mae_points": margin,
        "frozen": True,
    }
    (OUT / "p0_margin.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
