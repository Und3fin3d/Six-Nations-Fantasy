#!/usr/bin/env python3
"""
build_crosswalk.py  — Phase 3 of DATA_EXTRACTION_PLAN.md
========================================================
Produce ONE crosswalk table linking a player across the three sources:

    norm_key  <->  RapidAPI player_id  <->  RugbyPass slug  <->  official name

by outer-joining the three source tables on `key` (= norm_key of the player's
name), then a SECONDARY surname-only pass that recovers the obvious initial
mismatches the plan calls out (RugbyPass "Ignacio" vs official "J." Brex —
full first name vs abbreviated initial yield different norm_keys but the same
surname).

Sources
-------
  data/api_player_match.csv       API side  : (player_id, player_name) -> key
  data/6n_players.csv             API ref   : canonical_pos, fallback names
  data/rp_bio.csv                 RugbyPass : key, slug, nationality, position
  data/official_player_match.csv  Official  : key, name, team

Output
------
  data/player_crosswalk.csv  — one row per distinct player, columns:
    key, api_player_id, api_name, rp_slug, rp_name, official_name,
    nationality, canonical_pos, in_api, in_rp, in_official,
    key_collision, match_method

`match_method` in {"key","surname"}; "surname" rows were linked by the
secondary surname pass rather than an exact norm_key match.

Usage:  python build_crosswalk.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# reuse, do NOT reinvent
from compare_api_official import norm_key  # noqa: F401  (key already in files)
from compare_three_way import SLUG_TO_NAME, slug_key  # noqa: F401

BASE = Path(__file__).parent
DATA = BASE / "data"

# RugbyPass `nationality` is a citizenship, not necessarily the 6N team a player
# turns out for (project players, dual nationals).  For the surname tiebreaker
# we only trust nationality when it IS one of the six 6N nations; otherwise we
# treat it as "unknown" and fall back to surname-uniqueness alone.
SIX_NATIONS = {"England", "France", "Ireland", "Italy", "Scotland", "Wales"}


def surname_of(key: str) -> str:
    """'j|brex' -> 'brex'.  The part after the '|' in a norm_key."""
    if not isinstance(key, str) or "|" not in key:
        return ""
    return key.split("|", 1)[1]


def last_token(name: str) -> str:
    """Real final whitespace token of a name, ascii-folded, letters only.

    'Juan Ignacio Brex' -> 'brex'   (norm_key surname is 'ignaciobrex', so the
    norm_key surname alone can't link Brex; the true last name can).
    'juan-ignacio-brex' (slug) is normalised by the caller before this.
    """
    import re
    import unicodedata
    if not isinstance(name, str) or not name.strip():
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    parts = s.replace("-", " ").split()
    if not parts:
        return ""
    return re.sub(r"[^a-z]", "", parts[-1].lower())


# --------------------------------------------------------------------------- #
#  Load each source, reduced to its distinct players keyed by norm_key.
# --------------------------------------------------------------------------- #
def load_api() -> pd.DataFrame:
    """Distinct (player_id, player_name) for every SIX NATIONS player.

    Source is the per-match long table api_player_match.csv, filtered to rows
    with a Six Nations appearance (the file also holds club competitions from the
    backfill, which are NOT 6N players and must not enter the crosswalk).
    canonical_pos mirrors build_features.py:canon_pos (6n_players.csv modal start,
    api_player_match fallback).  Key = norm_key(player_name); the file may already
    carry a `key`, but we recompute to be safe.
    """
    pm = pd.read_csv(DATA / "api_player_match.csv")
    pm["key"] = pm["player_name"].map(norm_key)

    # The crosswalk links SIX NATIONS players only (its stated contract). The
    # api_player_match.csv backfill also carries club competitions (Top 14, URC,
    # Premiership, Champions/Challenge Cup), so restrict the player SET to those
    # with >=1 Six Nations appearance before reducing to one row per player_id.
    # (The canonical_pos fallback below deliberately still reads the FULL table —
    # a bench-only 6N player's modal start can come from a club match, matching
    # build_features.py:canon_pos exactly.)
    six_n = pm[pm["comp_name"].str.contains("Six Nations", case=False, na=False)]
    # one row per (player_id, name); prefer the player's 6N-appearance name
    api = (six_n[["player_id", "player_name", "key"]]
           .dropna(subset=["player_id"])
           .drop_duplicates(subset=["player_id"]))
    api["player_id"] = api["player_id"].astype("Int64")

    # canonical_pos — IDENTICAL derivation to build_features.py:canon_pos (the
    # one reliable position source): the fixed 6N-start modal position keyed by
    # player_id (data/6n_players.csv), falling back to the all-comp modal-start
    # jersey in api_player_match.csv for bench-only players absent from the 6N
    # reference. Do NOT join 6n_players.canonical_pos alone — it is blank for
    # never-started players (e.g. Gus McCarthy), which is why the api fallback
    # exists. Any consumer needing canonical position should match THIS logic.
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
    # rp_name: prefer the human name behind the slug, else humanise the slug
    rp["rp_name"] = rp["slug"].map(
        lambda s: SLUG_TO_NAME.get(s, s.replace("-", " ").title()))
    rp = rp.rename(columns={"slug": "rp_slug"})
    return rp[["key", "rp_slug", "rp_name", "nationality", "position"]]


def load_official() -> pd.DataFrame:
    off = pd.read_csv(DATA / "official_player_match.csv")
    off = off.dropna(subset=["key"])
    # one row per distinct (key, name, team); a key can recur across teams only
    # via collision, handled later — here keep the most frequent name/team.
    off = (off.groupby("key")
              .agg(official_name=("name", lambda s: s.mode().iat[0]),
                   official_team=("team", lambda s: s.mode().iat[0]))
              .reset_index())
    return off


# --------------------------------------------------------------------------- #
#  Outer-join on key, then mark collisions and the secondary surname pass.
# --------------------------------------------------------------------------- #
def build() -> pd.DataFrame:
    api = load_api()
    rp = load_rp()
    off = load_official()

    # A key collides when >1 distinct api_player_id share it (e.g. two J. Smith).
    coll_counts = api.groupby("key")["api_player_id"].nunique()
    collision_keys = set(coll_counts[coll_counts > 1].index)

    # Outer-join all three on key.  API may have multiple rows for a collision
    # key — keep them all (merge preserves the duplication).
    cw = api.merge(rp, on="key", how="outer").merge(off, on="key", how="outer")

    cw["in_api"] = cw["api_player_id"].notna()
    cw["in_rp"] = cw["rp_slug"].notna()
    cw["in_official"] = cw["official_name"].notna()
    cw["key_collision"] = cw["key"].isin(collision_keys)
    cw["match_method"] = "key"

    # --- SECONDARY surname pass --------------------------------------------- #
    # Target: keys that exist in ONLY ONE source (an unmatched singleton) and
    # whose surname uniquely identifies a counterpart in another source that is
    # ALSO an unmatched singleton.  This recovers initial-vs-fullname mismatches
    # (Ignacio Brex `i|brex`  vs  J. Brex `j|brex`) without touching rows that
    # already matched on key.
    cw = _surname_pass(cw)

    # tidy column order
    cols = ["key", "api_player_id", "api_name", "rp_slug", "rp_name",
            "official_name", "nationality", "canonical_pos",
            "in_api", "in_rp", "in_official", "key_collision", "match_method"]
    cw = cw[cols].sort_values(["key", "api_player_id"]).reset_index(drop=True)
    return cw


def _nat_team_compatible(nat, team) -> bool:
    """True if RP nationality and API/official team don't actively conflict."""
    if not isinstance(nat, str) or nat not in SIX_NATIONS:
        return True               # unknown / non-6N citizenship -> don't block
    if not isinstance(team, str) or team not in SIX_NATIONS:
        return True
    return nat == team


def _row_source(row) -> str:
    """Single source a lonely row belongs to: 'api' | 'rp' | 'official'."""
    if row["in_api"]:
        return "api"
    if row["in_rp"]:
        return "rp"
    return "official"


def _surname_candidates(row) -> set:
    """Surname forms by which this lonely row may be matched.

    Combines the norm_key surname (handles initial-vs-firstname) with the REAL
    last-name token of the row's own name string (handles multi-first-name
    players whose norm_key surname is polluted, e.g. API 'Juan Ignacio Brex' ->
    norm_key surname 'ignaciobrex' but true last name 'brex').
    """
    cands = set()
    s = surname_of(row.get("key"))
    if s:
        cands.add(s)
    src = _row_source(row)
    name = {"api": row.get("api_name"),
            "rp": row.get("rp_slug"),          # slug -> last segment
            "official": row.get("official_name")}[src]
    lt = last_token(name)
    if lt:
        cands.add(lt)
    return cands


def _surname_pass(cw: pd.DataFrame) -> pd.DataFrame:
    """Recover cross-source singletons that share a unique surname.

    A 'lonely' row is present in exactly one source after the key join.  Two
    lonely rows from *different* sources are linkable when their surname-form
    sets intersect (norm_key surname OR real last-name token) and their RP
    nationality / 6N team don't conflict.  We link a pair only when the match
    is mutually unique — each side has exactly one compatible counterpart in
    the other source — so ambiguous surnames (multiple 'williams') are skipped.
    """
    cw = cw.copy()
    lonely_n = cw[["in_api", "in_rp", "in_official"]].sum(axis=1)
    lonely = cw[lonely_n == 1]

    # bucket lonely rows by source
    by_src = {"api": [], "rp": [], "official": []}
    info = {}  # idx -> (source, surname-set, nationality, team)
    for idx, row in lonely.iterrows():
        src = _row_source(row)
        by_src[src].append(idx)
        info[idx] = (src,
                     _surname_candidates(row),
                     row.get("nationality"),
                     row.get("official_team"))

    # candidate pairs across the three ordered source-pairs
    drop_idx: set = set()
    merged = 0
    for src_a, src_b in (("api", "rp"), ("api", "official"), ("official", "rp")):
        # build, for this pair, the compatible-counterpart sets both ways
        cand_ab: dict = {}   # a_idx -> [b_idx,...]
        cand_ba: dict = {}   # b_idx -> [a_idx,...]
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
        # mutually-unique pairs only
        for a_idx, bs in cand_ab.items():
            if a_idx in drop_idx or len(bs) != 1:
                continue
            b_idx = bs[0]
            if b_idx in drop_idx or len(cand_ba.get(b_idx, [])) != 1:
                continue
            _fold(cw, a_idx, b_idx)
            drop_idx.add(b_idx)
            merged += 1
            # the folded row now covers two sources -> remove from lonely pools
            for pool in by_src.values():
                if b_idx in pool:
                    pool.remove(b_idx)

    cw = cw.drop(index=drop_idx)

    # --- Phase B: official name-variant consolidation ---------------------- #
    # The official source can list one player under two initials across years
    # (e.g. Juan "Ignacio/Nacho" Brex -> 'J. Brex' in 2023/25, 'N. Brex' in
    # 2026 -> keys j|brex and n|brex).  After Phase A one variant carries the
    # API id and the other carries the RP slug, leaving the player split.  Fold
    # such pairs when they (a) share a unique last-name token, (b) are
    # nation/team compatible, and (c) between them cover API and RP without
    # both already holding the same id/slug.
    merged += _consolidate_name_variants(cw, drop_idx)
    cw = cw.drop(index=[i for i in drop_idx if i in cw.index])
    cw.attrs["surname_links"] = merged
    return cw


def _consolidate_name_variants(cw: pd.DataFrame, drop_idx: set) -> int:
    """Merge two partially-filled rows that are the same player under official
    name variants.  Operates in place on `cw`; records dropped indices in
    `drop_idx`.  Returns the number of merges performed."""
    # rows still missing at least one source are merge candidates
    incomplete = cw[(cw[["in_api", "in_rp", "in_official"]].sum(axis=1) < 3)
                    & cw["in_official"]]
    by_ln: dict = {}
    for idx, row in incomplete.iterrows():
        ln = last_token(row.get("official_name"))
        if ln:
            by_ln.setdefault(ln, []).append(idx)

    merged = 0
    for ln, idxs in by_ln.items():
        if len(idxs) != 2:               # unique pair only -> no ambiguity
            continue
        i, jx = idxs
        if i in drop_idx or jx in drop_idx:
            continue
        ri, rj = cw.loc[i], cw.loc[jx]
        # the two rows must be complementary: together they add API+RP, and
        # neither pair of the same source carries conflicting ids.
        if bool(ri["in_api"]) and bool(rj["in_api"]):
            continue                     # two different API ids -> distinct
        if bool(ri["in_rp"]) and bool(rj["in_rp"]):
            continue                     # two different RP slugs -> distinct
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
    """Merge row `drop` into row `keep` in place; mark match_method='surname'."""
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


# --------------------------------------------------------------------------- #
#  Verification report (the plan's acceptance gate)
# --------------------------------------------------------------------------- #
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

    # % of API players matched to RugbyPass
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

    # % of official players matched to API
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

    # collisions + surname links
    n_coll = int(cw["key_collision"].sum())
    n_surn = int((cw["match_method"] == "surname").sum())
    print("-" * 74)
    print(f"key_collision rows:    {n_coll}")
    print(f"surname-method links:  {n_surn}")
    if n_coll:
        print("  collision keys:",
              sorted(cw[cw["key_collision"]]["key"].unique().tolist()))

    # spot-check: 10 linked rows, prioritising surname-method ones
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
