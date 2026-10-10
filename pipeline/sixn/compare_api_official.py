from __future__ import annotations

import os
import re
import unicodedata
import pandas as pd


OFFICIAL = {
    2023: "data/official/2023/Six_Nations_Data_2023.xlsx",
    2025: "data/official/2025/sixnationsmatch.xlsx",
    2026: "data/official/2026/sixnationmatch.xlsx",
}

STAT_MAP = {
    "Min":  "minutes",
    "T":    "tries",
    "As":   "try_assists",
    "C":    "conversion_goals",
    "Pen":  "penalty_goals",
    "MC":   "metres",
    "DB":   "defenders_beaten",
    "Ta":   "tackles",
    "CPen": "penalties_conceded",
    "DG":   "drop_goals_converted",
    "OF":   "offload",
    "YC":   "yellow_cards",
    "RC":   "red_cards",
    "BS":   "tackle_turnover",
}
API_BLIND = ["50-22", "KR", "LS", "POTM", "SW"]


def norm_key(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.strip()
    parts = s.split()
    if not parts:
        return ""
    initial = parts[0][0].lower()
    surname = "".join(parts[1:]) if len(parts) > 1 else parts[0]
    surname = re.sub(r"[^a-z]", "", surname.lower())
    return f"{initial}|{surname}"


def parse_official(path: str, season: int) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    out = []
    for r_idx, sheet in enumerate(xl.sheet_names, start=1):
        raw = pd.read_excel(xl, sheet_name=sheet, header=None)
        labels = None
        team = None
        for _, row in raw.iterrows():
            vals = list(row.values)
            cell0 = vals[0]
            rowset = {str(v) for v in vals if isinstance(v, str)}
            if "Min" in rowset and "Pts" in rowset:
                labels = [str(v) for v in vals]
                team = str(cell0).strip()
                continue
            if labels is None or not isinstance(cell0, str) or not cell0.strip():
                continue
            rec = {"season": season, "round": r_idx, "team": team,
                   "name": cell0.strip(), "key": norm_key(cell0)}
            for lab, v in zip(labels[1:], vals[1:]):
                rec[lab] = v
            out.append(rec)
    df = pd.DataFrame(out)
    for lab in set(STAT_MAP) | set(API_BLIND) | {"Pts"}:
        if lab in df.columns:
            df[lab] = pd.to_numeric(df[lab], errors="coerce").fillna(0)
    return df


def load_api() -> pd.DataFrame:
    df = pd.read_csv("data/6n_player_match.csv")
    df["key"] = df.player_name.map(norm_key)
    return df


def compare():
    api = load_api()
    off = pd.concat([parse_official(p, s) for s, p in OFFICIAL.items()
                     if os.path.exists(p)], ignore_index=True)

    keys4 = ["season", "round", "team", "key"]
    api_u = api.drop_duplicates(keys4)
    j = off.merge(api_u, on=keys4, how="inner", suffixes=("_off", "_api"))
    chk = off.merge(api_u[keys4].assign(_hit=1), on=keys4, how="left")
    unmatched = chk[chk["_hit"].isna()]
    print("=" * 74)
    print(f"JOIN  | official rows: {len(off)}   matched: {len(j)} "
          f"({100*len(j)/len(off):.0f}%)   unmatched: {len(unmatched)}")
    if len(unmatched):
        ex = unmatched.groupby("season").size().to_dict()
        print(f"      | unmatched by season: {ex}")
        print(f"      | e.g. {unmatched.name.head(10).tolist()}")
    print("=" * 74)

    hdr = f"{'stat':<14}{'n':>5}{'corr':>7}{'exact%':>8}{'off→api ratio':>15}{'mean Δ(o-a)':>13}{'MAE':>7}"
    print("\nPER-STAT AGREEMENT  (official vs API, matched player-matches)")
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    rows_report = []
    for lab, apicol in STAT_MAP.items():
        oc = lab if lab in j.columns else None
        if oc is None or apicol not in j.columns:
            continue
        o = pd.to_numeric(j[oc], errors="coerce").fillna(0).astype(float)
        a = pd.to_numeric(j[apicol], errors="coerce").fillna(0).astype(float)
        mask = ~(o.isna() | a.isna())
        o, a = o[mask], a[mask]
        n = len(o)
        corr = o.corr(a) if o.std() and a.std() else float("nan")
        exact = 100 * (o == a).mean()
        ratio = a.sum() / o.sum() if o.sum() else float("nan")
        meand = (o - a).mean()
        mae = (o - a).abs().mean()
        rows_report.append((lab, apicol, n, corr, exact, ratio, meand, mae))
        print(f"{lab:<14}{n:>5}{corr:>7.3f}{exact:>7.0f}%"
              f"{ratio:>14.2f}x{meand:>13.2f}{mae:>7.2f}")
    print("-" * len(hdr))
    print("ratio  = API total / official total  (1.00 = identical volume)")
    print("mean Δ = official minus API per player-match (+ = API undercounts)")

    o = pd.to_numeric(j["MC"], errors="coerce").fillna(0)
    a = pd.to_numeric(j["metres"], errors="coerce").fillna(0)
    nz = (o > 0) & (a > 0)
    print("\nMETRES detail (rows where both > 0): "
          f"median ratio API/official = {(a[nz]/o[nz]).median():.3f}, "
          f"official is {100*(o[nz].sum()/a[nz].sum()-1):.1f}% higher in total")

    print("\nFANTASY POINTS  official Pts  vs  fantasy_pts_recon")
    o = j["Pts"].astype(float)
    a = j["fantasy_pts_recon"].astype(float)
    print(f"  overall   n={len(j)}  corr={o.corr(a):.3f}  "
          f"recon captures {100*a.sum()/o.sum():.0f}% of total points  "
          f"mean gap(o-a)={ (o-a).mean():.1f}")
    by = (j.assign(off=o, rec=a, gap=o - a)
            .groupby("canonical_pos")
            .agg(n=("gap", "size"), corr=("off", lambda s: s.corr(a.loc[s.index])),
                 off_mean=("off", "mean"), rec_mean=("rec", "mean"),
                 gap_mean=("gap", "mean"))
            .sort_values("gap_mean", ascending=False))
    print(by.round(2).to_string())

    j["_gap"] = (j["Pts"].astype(float) - j["fantasy_pts_recon"].astype(float))
    worst = j.reindex(j["_gap"].abs().sort_values(ascending=False).index).head(12)
    print("\nLARGEST per-match point gaps (official − recon):")
    for _, r in worst.iterrows():
        extras = []
        for lab in API_BLIND:
            v = r.get(lab, 0)
            if lab in j.columns and pd.notna(v) and v:
                extras.append(f"{lab}={int(v)}")
        print(f"  {r['name']:<22} {int(r['season'])} R{int(r['round'])} "
              f"{r['canonical_pos'] or '?':<11} off={int(r['Pts']):>3} "
              f"recon={int(r['fantasy_pts_recon']):>3} gap={int(r['_gap']):>4}"
              f"   {' '.join(extras)}")

    print("\nAPI BLIND SPOTS (official-only events, total over matched rows):")
    for lab in API_BLIND:
        if lab in j.columns:
            tot = pd.to_numeric(j[lab], errors="coerce").fillna(0).sum()
            nz = (pd.to_numeric(j[lab], errors="coerce").fillna(0) > 0).sum()
            print(f"  {lab:<7} total={int(tot):>4} over {nz} player-matches")

    return j


if __name__ == "__main__":
    compare()
