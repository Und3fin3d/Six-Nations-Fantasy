from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.sixn.compare_api_official import norm_key
from pipeline.sources.compare_three_way import SLUG_TO_NAME, slug_key

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data"

SIX_NATIONS = {"England", "France", "Ireland", "Italy", "Scotland", "Wales"}


def surname_of(key: str) -> str:
    if not isinstance(key, str) or "|" not in key:
        return ""
    return key.split("|", 1)[1]


def last_token(name: str) -> str:
    import re
    import unicodedata
    if not isinstance(name, str) or not name.strip():
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    parts = s.replace("-", " ").split()
    if not parts:
        return ""
    return re.sub(r"[^a-z]", "", parts[-1].lower())


def load_api() -> pd.DataFrame:
    pm = pd.read_csv(DATA / "api_player_match.csv")
    pm["key"] = pm["player_name"].map(norm_key)

    six_n = pm[pm["comp_name"].str.contains("Six Nations", case=False, na=False)]
    api = (six_n[["player_id", "player_name", "key"]]
           .dropna(subset=["player_id"])
           .drop_duplicates(subset=["player_id"]))
    api["player_id"] = api["player_id"].astype("Int64")

    ref = pd.read_csv(DATA / "6n_players.csv")
    pos_6n = {int(p): c for p, c in zip(ref["player_id"], ref["canonical_pos"])
              if pd.notna(p)}
    pos_api = {int(p): c for p, c in
               (pm.dropna(subset=["canonical_pos"])
                  .drop_duplicates("player_id")[["player_id", "canonical_pos"]]
                  .itertuples(index=False, name=None))
               if pd.notna(p)}

    def canon_pos(pid):
        p = pos_6n.get(pid)
        return p if isinstance(p, str) and p else pos_api.get(pid)

    api["canonical_pos"] = api["player_id"].map(
        lambda x: canon_pos(int(x)) if pd.notna(x) else None)

    api = api.rename(columns={"player_name": "api_name",
                              "player_id": "api_player_id"})
    return api[["key", "api_player_id", "api_name", "canonical_pos"]]


def load_rp() -> pd.DataFrame:
    rp = pd.read_csv(DATA / "rp_bio.csv")
    rp = rp.dropna(subset=["key"]).drop_duplicates(subset=["slug"])
    rp["rp_name"] = rp["slug"].map(
        lambda s: SLUG_TO_NAME.get(s, s.replace("-", " ").title()))
    rp = rp.rename(columns={"slug": "rp_slug"})
    return rp[["key", "rp_slug", "rp_name", "nationality", "position"]]


def load_official() -> pd.DataFrame:
    off = pd.read_csv(DATA / "official_player_match.csv")
    off = off.dropna(subset=["key"])
    off = (off.groupby("key")
              .agg(official_name=("name", lambda s: s.mode().iat[0]),
                   official_team=("team", lambda s: s.mode().iat[0]))
              .reset_index())
    return off


def build() -> pd.DataFrame:
    api = load_api()
    rp = load_rp()
    off = load_official()

    coll_counts = api.groupby("key")["api_player_id"].nunique()
    collision_keys = set(coll_counts[coll_counts > 1].index)

    cw = api.merge(rp, on="key", how="outer").merge(off, on="key", how="outer")

    cw["in_api"] = cw["api_player_id"].notna()
    cw["in_rp"] = cw["rp_slug"].notna()
    cw["in_official"] = cw["official_name"].notna()
    cw["key_collision"] = cw["key"].isin(collision_keys)
    cw["match_method"] = "key"

    cw = _surname_pass(cw)

    cols = ["key", "api_player_id", "api_name", "rp_slug", "rp_name",
            "official_name", "nationality", "canonical_pos",
            "in_api", "in_rp", "in_official", "key_collision", "match_method"]
    cw = cw[cols].sort_values(["key", "api_player_id"]).reset_index(drop=True)
    return cw


def _nat_team_compatible(nat, team) -> bool:
    if not isinstance(nat, str) or nat not in SIX_NATIONS:
        return True
    if not isinstance(team, str) or team not in SIX_NATIONS:
        return True
    return nat == team


def _row_source(row) -> str:
    if row["in_api"]:
        return "api"
    if row["in_rp"]:
        return "rp"
    return "official"


def _surname_candidates(row) -> set:
    cands = set()
    s = surname_of(row.get("key"))
    if s:
        cands.add(s)
    src = _row_source(row)
    name = {"api": row.get("api_name"),
            "rp": row.get("rp_slug"),
            "official": row.get("official_name")}[src]
    lt = last_token(name)
    if lt:
        cands.add(lt)
    return cands


def _surname_pass(cw: pd.DataFrame) -> pd.DataFrame:
    cw = cw.copy()
    lonely_n = cw[["in_api", "in_rp", "in_official"]].sum(axis=1)
    lonely = cw[lonely_n == 1]

    by_src = {"api": [], "rp": [], "official": []}
    info = {}
    for idx, row in lonely.iterrows():
        src = _row_source(row)
        by_src[src].append(idx)
        info[idx] = (src,
                     _surname_candidates(row),
                     row.get("nationality"),
                     row.get("official_team"))

    drop_idx: set = set()
    merged = 0
    for src_a, src_b in (("api", "rp"), ("api", "official"), ("official", "rp")):
        cand_ab: dict = {}
        cand_ba: dict = {}
        for a_idx in by_src[src_a]:
            _, a_sn, a_nat, a_team = info[a_idx]
            for b_idx in by_src[src_b]:
                _, b_sn, b_nat, b_team = info[b_idx]
                if not (a_sn & b_sn):
                    continue
                nat = a_nat if isinstance(a_nat, str) and a_nat else b_nat
                team = a_team if isinstance(a_team, str) and a_team else b_team
                if not _nat_team_compatible(nat, team):
                    continue
                cand_ab.setdefault(a_idx, []).append(b_idx)
                cand_ba.setdefault(b_idx, []).append(a_idx)
        for a_idx, bs in cand_ab.items():
            if a_idx in drop_idx or len(bs) != 1:
                continue
            b_idx = bs[0]
            if b_idx in drop_idx or len(cand_ba.get(b_idx, [])) != 1:
                continue
            _fold(cw, a_idx, b_idx)
            drop_idx.add(b_idx)
            merged += 1
            for pool in by_src.values():
                if b_idx in pool:
                    pool.remove(b_idx)

    cw = cw.drop(index=drop_idx)

    merged += _consolidate_name_variants(cw, drop_idx)
    cw = cw.drop(index=[i for i in drop_idx if i in cw.index])
    cw.attrs["surname_links"] = merged
    return cw


def _consolidate_name_variants(cw: pd.DataFrame, drop_idx: set) -> int:
    incomplete = cw[(cw[["in_api", "in_rp", "in_official"]].sum(axis=1) < 3)
                    & cw["in_official"]]
    by_ln: dict = {}
    for idx, row in incomplete.iterrows():
        ln = last_token(row.get("official_name"))
        if ln:
            by_ln.setdefault(ln, []).append(idx)

    merged = 0
    for ln, idxs in by_ln.items():
        if len(idxs) != 2:
            continue
        i, jx = idxs
        if i in drop_idx or jx in drop_idx:
            continue
        ri, rj = cw.loc[i], cw.loc[jx]
        if bool(ri["in_api"]) and bool(rj["in_api"]):
            continue
        if bool(ri["in_rp"]) and bool(rj["in_rp"]):
            continue
        nat = ri.get("nationality") if isinstance(ri.get("nationality"), str) \
            else rj.get("nationality")
        team = ri.get("official_team") if isinstance(ri.get("official_team"), str) \
            else rj.get("official_team")
        if not _nat_team_compatible(nat, team):
            continue
        _fold(cw, i, jx)
        drop_idx.add(jx)
        merged += 1
    return merged


def _fold(cw: pd.DataFrame, keep: int, drop: int) -> None:
    fill_cols = ["api_player_id", "api_name", "rp_slug", "rp_name",
                 "official_name", "nationality", "canonical_pos",
                 "official_team"]
    for c in fill_cols:
        if c not in cw.columns:
            continue
        if pd.isna(cw.at[keep, c]) and pd.notna(cw.at[drop, c]):
            cw.at[keep, c] = cw.at[drop, c]
    for flag in ("in_api", "in_rp", "in_official"):
        cw.at[keep, flag] = bool(cw.at[keep, flag]) or bool(cw.at[drop, flag])
    cw.at[keep, "match_method"] = "surname"


def report(cw: pd.DataFrame) -> None:
    n_total = len(cw)
    print("=" * 74)
    print(f"PLAYER CROSSWALK  |  distinct players: {n_total}")
    print("=" * 74)

    in_api = cw["in_api"].sum()
    in_rp = cw["in_rp"].sum()
    in_off = cw["in_official"].sum()
    print(f"  present in API:      {in_api}")
    print(f"  present in RugbyPass:{in_rp:>5}")
    print(f"  present in official: {in_off}")

    api_rows = cw[cw["in_api"]]
    api_rp = api_rows[api_rows["in_rp"]]
    pct_api_rp = 100 * len(api_rp) / len(api_rows) if len(api_rows) else float("nan")
    print("-" * 74)
    print(f"API -> RugbyPass match: {len(api_rp)}/{len(api_rows)} "
          f"= {pct_api_rp:.1f}%   (target >99%)")
    unmatched_api = api_rows[~api_rows["in_rp"]]
    print(f"  unmatched API players ({len(unmatched_api)}):")
    for _, r in unmatched_api.iterrows():
        print(f"    {str(r['api_name']):<28} key={r['key']:<22} "
              f"pos={r['canonical_pos']}")

    off_rows = cw[cw["in_official"]]
    off_api = off_rows[off_rows["in_api"]]
    pct_off_api = 100 * len(off_api) / len(off_rows) if len(off_rows) else float("nan")
    print("-" * 74)
    print(f"Official -> API match:  {len(off_api)}/{len(off_rows)} "
          f"= {pct_off_api:.1f}%")
    unmatched_off = off_rows[~off_rows["in_api"]]
    if len(unmatched_off):
        print(f"  unmatched official players ({len(unmatched_off)}): "
              f"{unmatched_off['official_name'].head(15).tolist()}")

    n_coll = int(cw["key_collision"].sum())
    n_surn = int((cw["match_method"] == "surname").sum())
    print("-" * 74)
    print(f"key_collision rows:    {n_coll}")
    print(f"surname-method links:  {n_surn}")
    if n_coll:
        print("  collision keys:",
              sorted(cw[cw["key_collision"]]["key"].unique().tolist()))

    print("-" * 74)
    print("SPOT-CHECK (10 linked rows; surname-method first):")
    linked = cw[(cw["in_api"].astype(int) + cw["in_rp"].astype(int) +
                 cw["in_official"].astype(int)) >= 2]
    surn = linked[linked["match_method"] == "surname"]
    rest = linked[linked["match_method"] != "surname"]
    sample = pd.concat([surn, rest.head(max(0, 10 - len(surn)))]).head(10)
    for _, r in sample.iterrows():
        print(f"  [{r['match_method']:<7}] key={r['key']:<20} "
              f"api={str(r['api_name'])[:20]:<20} "
              f"rp={str(r['rp_slug'])[:20]:<20} "
              f"off={str(r['official_name'])[:18]:<18} "
              f"nat={r['nationality']}")
    print("=" * 74)


def main() -> None:
    cw = build()
    out = DATA / "player_crosswalk.csv"
    cw.to_csv(out, index=False)
    report(cw)
    print(f"\nwrote {out}  ({len(cw)} rows)")


if __name__ == "__main__":
    main()
