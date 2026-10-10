from __future__ import annotations
import re, unicodedata
import numpy as np
import pandas as pd

from model.history import past_seasons, utc_cutoff


ATT = {"tries": 12, "try_assists": 5, "defenders_beaten": 2, "offloads": 2, "clean_breaks": 3}
DEF = {"tackles": 1, "missed_tackles": -1, "turnovers_won": 4}
DISC = {"turnovers_conceded": -1, "penalties_conceded": -1, "yellow_cards": -5, "red_cards": -10}
HALFLIFE_YEARS = 1.6


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s.lower())).strip()


def _season_end_year(s):
    yrs = re.findall(r"\d{4}", str(s))
    return int(yrs[-1]) if yrs else 2026


def rp_rates(asof_year=2026, min_minutes=120, *, asof=None):
    cs = pd.read_csv("data/rp_compstats.csv")
    if cs["slug"].nunique() < 600:
        raise RuntimeError(
            f"rp_compstats.csv looks stale ({cs['slug'].nunique()} slugs < 600); "
            "rerun `python -m pipeline.sources.build_rugbypass_tables` to restore the NCR backfill"
        )
    if asof is not None:
        cs = past_seasons(cs, asof)
        asof_year = utc_cutoff(asof).year
    cs["w"] = 0.5 ** ((asof_year - cs["season"].map(_season_end_year)).clip(lower=0) / HALFLIFE_YEARS)
    cs = cs[cs["w"] > 0.15]
    rows = []
    for slug, g in cs.groupby("slug"):
        wm = (g.w * g.minutes).sum()
        if wm < min_minutes:
            continue
        r = {e: (g.w * g[e]).sum() / wm * 80 for e in {**ATT, **DEF, **DISC}}
        rows.append(
            dict(
                slug=slug, att=sum(r[e] * v for e, v in ATT.items()), dfn=sum(r[e] * v for e, v in DEF.items()), dsc=sum(r[e] * v for e, v in DISC.items()),
                rp_min=wm,
            )
        )
    out = pd.DataFrame(rows, columns=["slug", "att", "dfn", "dsc", "rp_min"])
    out["name_key"] = out.slug.str.replace("-", " ").map(_norm)
    return out


def calibrate(rp, api_players):
    m = rp.merge(api_players, on="name_key", suffixes=("_rp", "_api"))
    m = m[(m.rp_min > 300)]
    fac = {}
    for comp in ("att", "dfn"):
        a, b = m[f"{comp}_api"], m[f"{comp}_rp"]
        good = (b.abs() > 1) & np.isfinite(a) & np.isfinite(b)
        fac[comp] = float(np.clip(np.median(a[good] / b[good]), 0.5, 1.2)) if good.sum() >= 8 else 1.0
    rp = rp.copy()
    rp["att"] *= fac["att"]
    rp["dfn"] *= fac["dfn"]
    return rp, fac


if __name__ == "__main__":
    r = rp_rates()
    print(f"{len(r)} RugbyPass players with ≥120 weighted min")
    print(r.sort_values("att", ascending=False).head(12)[["slug", "att", "dfn", "dsc", "rp_min"]].round(1).to_string(index=False))
