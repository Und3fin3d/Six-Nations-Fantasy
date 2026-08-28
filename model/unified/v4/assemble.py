"""Deployable gbdt_v4 artifact: fit through an exclusive cutoff and activate.

Usage (pinned env):

  # Fit the admitted configuration through a pre-lock cutoff.
  python -m model.unified.v4.assemble fit \
      --config A --cutoff 2026-11-06T18:00:00Z \
      --output data/unified/v4/models/gbdt_v4_ncr_gw4.pkl

  # Register it as an ADDITIONAL active shadow next to the v3 baseline.
  python -m model.unified.v4.assemble activate \
      --model data/unified/v4/models/gbdt_v4_ncr_gw4.pkl --target-gw 4

`activate` converts data/unified/v3/shadow_active.json to a list when needed
and never removes existing entries, so the v3 baseline keeps shadowing the
same gameweeks. The freeze itself stays on the v3 write-once path
(`model/unified/v3/shadow.py`), which now knows the gbdt_v4 engine and applies
the stored per-event K recipe at prediction time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..data import ROOT
from ..features import build_pit_features
from ..v3.harness import attach_match_timestamps
from .experiments import HURDLE_EVENTS
from .features import add_v4_base_stats, apply_eb_features, fit_shrinkage_k
from .gbdt import V4GBDT

DATA = ROOT / "data"
OUT = DATA / "unified" / "v4"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(name: str) -> V4GBDT:
    if name == "A":
        return V4GBDT(weighting="natural")
    if name == "B":
        return V4GBDT(weighting="natural", pool_player_id=True, player_effects=True)
    if name == "A+hurdle":
        return V4GBDT(weighting="natural", hurdle_events=HURDLE_EVENTS)
    if name == "B+hurdle":
        return V4GBDT(weighting="natural", pool_player_id=True, player_effects=True,
                      hurdle_events=HURDLE_EVENTS)
    raise ValueError(name)


def fit(args) -> None:
    cutoff = pd.Timestamp(args.cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    raw = pd.read_csv(DATA / "unified" / "player_match.csv", low_memory=False,
                      parse_dates=["date"])
    store = add_v4_base_stats(build_pit_features(attach_match_timestamps(raw)))
    match_at = pd.to_datetime(store["match_at"], utc=True)
    train = store[match_at < cutoff].copy()
    if train.empty:
        raise ValueError(f"no training rows before {cutoff}")
    uses_eb = args.config.startswith("A")
    k_by_event = fit_shrinkage_k(train) if uses_eb else None
    if uses_eb:
        train = apply_eb_features(train, k_by_event)
    model = build_config(args.config)
    print(f"fitting gbdt_v4[{args.config}] on {len(train):,} rows "
          f"(max match {match_at.max()})", flush=True)
    model.fit(train)
    if k_by_event:
        model.k_by_event = k_by_event
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    manifest = {
        "schema_version": 1, "engine": "gbdt_v4", "config": args.config,
        "exclusive_cutoff": cutoff.isoformat(),
        "training_rows": int(len(train)),
        "max_training_match_at": pd.to_datetime(train["match_at"], utc=True).max().isoformat(),
        "k_by_event": k_by_event,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": _sha(args.output),
        "plan": "data/unified/v4/RESEARCH_PLAN.md",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def activate(args) -> None:
    path = DATA / "unified" / "v3" / "shadow_active.json"
    existing = json.loads(path.read_text()) if path.exists() else []
    specs = existing if isinstance(existing, list) else [existing]
    manifest = json.loads(
        args.model.with_suffix(args.model.suffix + ".json").read_text())
    entry = {
        "engine": "gbdt_v4", "model": str(args.model),
        "exclusive_cutoff": manifest["exclusive_cutoff"],
        "artifact_sha256": manifest["artifact_sha256"],
        "target_gw": int(args.target_gw),
    }
    specs = [s for s in specs
             if not (s.get("engine") == "gbdt_v4"
                     and s.get("target_gw") == entry["target_gw"])]
    specs.append(entry)
    path.write_text(json.dumps(specs, indent=2, sort_keys=True) + "\n")
    print(json.dumps(specs, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("fit")
    p.add_argument("--config", choices=("A", "B", "A+hurdle", "B+hurdle"), required=True)
    p.add_argument("--cutoff", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=fit)
    p = sub.add_parser("activate")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--target-gw", type=int, required=True)
    p.set_defaults(func=activate)
    return ap


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
