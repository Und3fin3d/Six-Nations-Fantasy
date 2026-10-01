"""World Rugby / Pulselive match-centre API (api.wr-rims-prod.pulselive.com/rugby/v3).

Per-match endpoints used by public open-source clients (no key):
    /match/{matchAltId}/summary   teams[i].teamList.list -> 23-man lineups (number, player id, name, captain)
    /match/{matchAltId}/timeline  timeline[] events with secs + type ("Sub On", "Sub Off", "IR", "HIAO",
                                  tries, cards, kicks) -> on/off times incl. HIA and blood replacements
    /match/{matchAltId}/stats     team (and per-player in recent tests) stat blocks
The repo already uses /rankings and /match lists from this host (build_wr.py, build_intl_results.py).

Modes
-----
  fetch   --alt-id ID --out DIR : live (blocked in the 2026-10 cloud session; works where egress allows)
  compare --minutes-glob GLOB   : offline check of Pulselive-derived minutes (github.com/bpcsaund/
                                  rugby-analytics data/raw/sheets_export/*_minutes.csv, built from
                                  summary+timeline) against our store minutes and official 6N Min.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, agreement, load_store, surname_key  # noqa: E402

BASE = "https://api.wr-rims-prod.pulselive.com/rugby/v3"
TEAM_FROM_FILE = {"new_zealand": "New Zealand", "south_africa": "South Africa"}
TOGGLES = ("Sub On", "Sub Off", "IR", "HIAO")


def minutes_from_timeline(summary: dict, timeline: dict, team_idx: int, full=80) -> pd.DataFrame:
    """Lineup + on/off seconds -> minutes, with replacement entry times (the bench-timing signal)."""
    players = [p for p in summary["teams"][team_idx]["teamList"]["list"] if p.get("number")]
    evs = {}
    for e in timeline.get("timeline", []):
        if e.get("type") in TOGGLES and e.get("playerId") is not None:
            evs.setdefault(str(e["playerId"]), []).append((e.get("time", {}).get("secs", 0), e["type"]))
    rows = []
    for p in players:
        pid = str(p["player"]["id"])
        started = int(p["number"]) <= 15
        state, last, total, first_on = ("on", 0, 0, 0) if started else ("off", None, 0, None)
        for secs, typ in sorted(evs.get(pid, [])):
            go_off = typ == "Sub Off" or (typ in ("IR", "HIAO") and state == "on")
            if go_off and state == "on":
                total += secs - last
                state = "off"
            elif not go_off and state == "off":
                state, last = "on", secs
                first_on = secs if first_on is None else first_on
        if state == "on":
            total += full * 60 - last
        rows.append({"player_id": pid, "name": p["player"].get("name", {}).get("display"),
                     "shirt": int(p["number"]), "started": started,
                     "first_on_min": None if first_on is None else round(first_on / 60),
                     "minutes": round(total / 60)})
    return pd.DataFrame(rows)


def fetch(alt_id: str, out: str) -> None:
    import requests

    os.makedirs(out, exist_ok=True)
    got = {}
    for part in ("summary", "timeline", "stats"):
        r = requests.get(f"{BASE}/match/{alt_id}/{part}", params={"language": "en"}, timeout=30)
        r.raise_for_status()
        Path(out, f"pulselive_{alt_id}_{part}.json").write_text(r.text)
        got[part] = r.json()
    for i in (0, 1):
        print(minutes_from_timeline(got["summary"], got["timeline"], i).to_string())


def compare(minutes_glob: str, store_path: str | None) -> dict:
    frames = []
    for f in sorted(glob.glob(minutes_glob)):
        t = os.path.basename(f).replace("_minutes.csv", "")
        d = pd.read_csv(f)
        d["team"] = TEAM_FROM_FILE.get(t, t.title())
        frames.append(d)
    pl = pd.concat(frames, ignore_index=True)
    pl["date"] = pd.to_datetime(pl.date, format="%d/%m/%Y").dt.date
    pl["sur"] = pl.player.map(surname_key)
    store = load_store(store_path)
    store = store[store.team.isin(pl.team.unique())]
    s_u = store.groupby(["date", "team", "sur"]).filter(lambda g: len(g) == 1)
    p_u = pl.groupby(["date", "team", "sur"]).filter(lambda g: len(g) == 1)
    j = p_u.merge(s_u, on=["date", "team", "sur"], suffixes=("_wr", ""))
    res = {"wr_rows": len(pl), "wr_dates": int(pl.date.nunique()), "joined": len(j),
           "minutes_vs_store_all": agreement(j.minutes_wr, j.minutes)}
    bench = j[j.started_wr.eq("n") & (j.minutes_wr > 0)]
    res["minutes_vs_store_bench_used"] = agreement(bench.minutes_wr, bench.minutes)
    res["abs_diff_gt5_share"] = round(float(((j.minutes_wr - j.minutes).abs() > 5).mean()), 3)
    # vs official Six Nations Min
    off = pd.read_csv(REPO / "data" / "official_player_match.csv")
    off["sur"] = off.name.map(surname_key)
    six = j[j.competition == "Six Nations"].copy()
    six["season"] = six.season.astype(int)
    six["round"] = six["round"].astype(int)
    o_u = off.groupby(["season", "round", "team", "sur"]).filter(lambda g: len(g) == 1)
    k = six.merge(o_u[["season", "round", "team", "sur", "Min"]], on=["season", "round", "team", "sur"])
    res["six_nations_official_rows"] = len(k)
    res["wr_minutes~official_Min"] = agreement(k.minutes_wr, k.Min)
    res["store_minutes~official_Min"] = agreement(k.minutes, k.Min)
    kb = k[k.started_wr.eq("n") & (k.Min > 0)]
    res["bench_wr_minutes~official_Min"] = agreement(kb.minutes_wr, kb.Min)
    res["bench_store_minutes~official_Min"] = agreement(kb.minutes, kb.Min)
    res["bench_mae_wr_vs_official"] = round(float((kb.minutes_wr - kb.Min).abs().mean()), 2)
    res["bench_mae_store_vs_official"] = round(float((kb.minutes - kb.Min).abs().mean()), 2)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--alt-id", required=True)
    f.add_argument("--out", default="pulselive_out")
    c = sub.add_parser("compare")
    c.add_argument("--minutes-glob", required=True)
    c.add_argument("--store")
    c.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch(a.alt_id, a.out)
    else:
        res = compare(a.minutes_glob, a.store)
        print(json.dumps(res, indent=2, default=str))
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
