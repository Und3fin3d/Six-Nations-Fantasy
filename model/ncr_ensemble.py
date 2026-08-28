#!/usr/bin/env python3
"""
model/ncr_ensemble.py — blend the empirical NCR model with the transplanted 6n champion.
=======================================================================================
The two models fail differently, which is exactly when an ensemble earns its keep:

  * the champion ranks player QUALITY better (GW1: rho 0.600 vs 0.578, MAE 8.96 vs 9.72)
    but is MATCHUP-BLIND on NCR — its FIXTURE (28 cols) and OWNTEAM (11 cols) families are
    entirely NaN and get imputed to the 6n position-means, so it cannot tell who anyone is
    playing;
  * the empirical model is a weaker player-quality estimator but does model the matchup
    (expected margin from World Rugby ratings).

GW1 evidence: a 50/50 blend produced the best ranked list of anything measured (rho 0.606,
vs 0.600 champion / 0.578 empirical), at the cost of diluting the empirical model's
top-end conviction (which is where the captain 2× lives).

Points are NOT rescaled. Each model's numbers are its own: the empirical model predicts NCR
fantasy points, the champion predicts on the 6n scale it was trained against (~1.24x higher).
A global scale factor cannot change either model's own squad — the MILP objective is linear in
points and every constraint is on price — but it DOES shift the weighting of an average, so the
effective split is computed from the two spreads and reported rather than normalised away.

Usage:
    /tmp/6n-model-pinned/bin/python -m model.ncr_ensemble              # 50/50
    /tmp/6n-model-pinned/bin/python -m model.ncr_ensemble --w 0.7      # 0.7·empirical
    /tmp/6n-model-pinned/bin/python -m model.ncr_ensemble --captain empirical
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import model.ncr_champion as CH
import model.ncr_project as NP


def build(w_emp: float = 0.5, exclude_teams=()) -> pd.DataFrame:
    """Both models' RAW points, plus their weighted average.

    The champion's points are NOT rescaled — they are its own output, on the 6n scale.
    Because that scale runs higher, a nominal 50/50 average is not an even split: the
    effective weight is proportional to each model's spread. We report it rather than
    silently normalising it away. Use --w to compensate if you want a true even split.
    """
    emp = NP.build_projection(exclude_teams=exclude_teams).set_index("id")
    champ = CH.build_projection(exclude_teams=exclude_teams).set_index("id")

    common = emp.index.intersection(champ.index)
    df = emp.loc[common].copy()
    df["emp_exp"] = emp.loc[common, "starter_exp"]
    df["champ_exp"] = champ.loc[common, "starter_exp"]     # raw, unrescaled
    df["starter_exp"] = w_emp * df["emp_exp"] + (1 - w_emp) * df["champ_exp"]
    df["supersub_exp"] = np.where(df.status.eq("B"), 3 * df.starter_exp, 0.5 * df.starter_exp)
    if "captain_score" in champ.columns:
        df["champ_captain_score"] = champ.loc[common, "captain_score"]

    # effective weight = share of the blend's variance each model contributes
    se, sc = df["emp_exp"].std(), df["champ_exp"].std()
    eff = (w_emp * se) / (w_emp * se + (1 - w_emp) * sc)
    disagree = (df["emp_exp"].rank(pct=True) - df["champ_exp"].rank(pct=True)).abs()
    print(f"blend on {len(df)} players: nominal w_emp={w_emp:.2f}  →  EFFECTIVE "
          f"{eff:.0%} empirical / {1-eff:.0%} champion (spreads {se:.1f} vs {sc:.1f}, unrescaled)")
    print(f"  mean |rank disagreement| = {disagree.mean():.3f}")
    out = df.reset_index().rename(columns={"index": "id"})
    out.attrs["eff_w"] = eff
    return out


POS_ORDER = {"Prop": 0, "Hooker": 1, "Lock": 2, "Loose Forward": 3, "Scrum Half": 4,
             "Fly Half": 5, "Centre": 6, "Back Three": 7}


def _optimise_on(df: pd.DataFrame, points_col: str, captain_col: str | None = None):
    """Solve the squad using `points_col` as each player's expected points."""
    d = df.copy()
    d["starter_exp"] = d[points_col]
    d["supersub_exp"] = np.where(d.status.eq("B"), 3 * d.starter_exp, 0.5 * d.starter_exp)
    squad, _, _ = NP.optimise(d)
    if captain_col and captain_col in squad.columns and squad[captain_col].notna().any():
        xv = squad[~squad.is_sub & squad.status.eq("P")]
        if len(xv):
            squad["is_capt"] = squad["name"].eq(xv.loc[xv[captain_col].idxmax(), "name"])
    return squad


def _value_under(squad: pd.DataFrame, points: pd.Series) -> float:
    """Total fantasy points of a squad if `points` were the truth (captain 2x, sub 3x/0.5x)."""
    total = 0.0
    for r in squad.itertuples():
        p = float(points.loc[r.id])
        if r.is_sub:
            total += (3.0 if r.status == "B" else 0.5) * p
        else:
            total += p * (2.0 if r.is_capt else 1.0)
    return total


def _team_table(squad: pd.DataFrame) -> list[str]:
    squad = squad.copy()
    squad["po"] = squad.pos.map(POS_ORDER)
    squad = squad.sort_values(["is_sub", "po", "avg_exp"], ascending=[True, True, False])
    L = ["| Pos | Player | Nation | £ | xMin | xPts1 (NCR) | xPts2 (6N champ) | avg | Role |",
         "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in squad.itertuples():
        if r.is_capt:
            role, nm = "**CAPTAIN** 2x", f"**{r.name}**"
        elif r.is_sub:
            role, nm = "**SUPER SUB** 3x off bench", f"**{r.name}**"
        else:
            role, nm = "", r.name
        L.append(f"| {r.pos} | {nm} | {r.team} | {r.value:.1f} | {r.exp_min:.0f} | "
                 f"{r.emp_exp:.1f} | {r.champ_exp:.1f} | {r.avg_exp:.1f} | {role} |")
    return L


def write_comparison(df: pd.DataFrame, squads: dict, path, w_emp: float, exclude_teams) -> None:
    pts = {"NCR empirical": df.set_index("id")["emp_exp"],
           "6N champion": df.set_index("id")["champ_exp"],
           "blend": df.set_index("id")["avg_exp"]}
    L = [f"# Nations Championship Fantasy — GW{NP.CUR_GD}: three teams compared", ""]
    if exclude_teams:
        L += [f"> Excludes already-started matches: **{', '.join(exclude_teams)}**.", ""]
    eff = df.attrs.get("eff_w")
    L += [
        "**xPts1** = NCR empirical model (recency-weighted per-80 rates + calibrated club form, "
        "matchup from World Rugby ratings). Predicts NCR fantasy points.",
        "**xPts2** = 6N champion, frozen and transplanted (LGBM rate heads + XGB point specialist + "
        "latent/POTM block). **Raw, unrescaled** — these are its own numbers, on the 6n points scale "
        "it was trained against, which runs higher than the NCR scale. It is also **matchup-blind** on "
        "NCR: its FIXTURE and OWNTEAM feature families are entirely missing, so it cannot see who a "
        "player is up against.",
        f"**avg** = {w_emp:.2f}·xPts1 + {1-w_emp:.2f}·xPts2.",
        "", "All points are per-player at **1x**; the captain 2x and super-sub 3x multipliers are "
        "applied in the totals, not in the columns.", ""]
    if eff is not None:
        L += [f"> **The two columns are not on the same scale, and are not averaged evenly.** Because "
              f"xPts2 has the wider spread, a nominal {w_emp:.2f}/{1-w_emp:.2f} split gives an effective "
              f"**{eff:.0%} empirical / {1-eff:.0%} champion**. Pass `--w` to compensate. Comparing a "
              f"player's xPts1 against their xPts2 is meaningful only as a *ranking*, not as a difference "
              f"in points.", ""]
    L += [
        "## Cross-valuation — each team scored by each model", "",
        "| Team | by xPts1 (NCR) | by xPts2 (6N) | by avg |", "|---|---:|---:|---:|"]
    for name, sq in squads.items():
        L.append(f"| {name} | {_value_under(sq, pts['NCR empirical']):.1f} | "
                 f"{_value_under(sq, pts['6N champion']):.1f} | "
                 f"{_value_under(sq, pts['blend']):.1f} |")
    L += ["", "> Each team is best under the model that picked it — that is the optimiser working, "
          "not evidence. The informative number is how badly a team degrades under the *other* model.", ""]

    for i, (name, sq) in enumerate(squads.items(), 1):
        capt = sq.loc[sq.is_capt, "name"]
        L += [f"## {i}. {name}", "",
              f"**£{sq.value.sum():.1f}M / {NP.BUDGET:.0f}M**  ·  "
              f"**North {int((sq.hemi==1).sum())} / South {int((sq.hemi==2).sum())}**  ·  "
              f"**{sq.team.nunique()} nations**  ·  "
              f"**Captain:** {capt.iloc[0] if len(capt) else '—'}", ""]
        L += _team_table(sq)
        L.append("")

    ov = set(squads["NCR empirical"]["name"]) & set(squads["6N champion"]["name"])
    L += ["## Where they agree", "",
          f"Only **{len(ov)}/16** players appear in both single-model teams: "
          + (", ".join(sorted(ov)) if ov else "none") + ".", "",
          "Low overlap means the models are making genuinely different bets — which is when a blend "
          "is worth more than either, and also when the captain choice carries the most variance.", ""]
    Path(path).write_text("\n".join(L) + "\n")


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=0.5, help="weight on the EMPIRICAL model")
    ap.add_argument("--captain", choices=["blend", "empirical", "champion"], default="blend",
                    help="whose head picks the armband (the 2x is where the variance lives)")
    ap.add_argument("--exclude", nargs="*", default=[])
    args = ap.parse_args()

    df = build(args.w, args.exclude)
    eff_w = df.attrs.get("eff_w")
    df["avg_exp"] = df["starter_exp"]                  # build() already wrote the blend here
    df = df.set_index("id", drop=False)
    df.attrs["eff_w"] = eff_w
    tag = "_blend" + ("_ex_" + "_".join(t.replace(" ", "") for t in args.exclude)
                      if args.exclude else "")
    df.to_csv(NP.NCR / f"ncr_gw{NP.CUR_GD}_projections{tag}.csv", index=False)

    cap_col = "champ_captain_score" if "champ_captain_score" in df.columns else None
    squads = {
        "NCR empirical": _optimise_on(df, "emp_exp"),
        "6N champion": _optimise_on(df, "champ_exp", captain_col=cap_col),
        f"Blend ({args.w:.2f}/{1-args.w:.2f})": _optimise_on(df, "avg_exp"),
    }
    blend_key = f"Blend ({args.w:.2f}/{1-args.w:.2f})"
    squads = {**squads}                                 # preserve insertion order

    md = NP.NCR / f"ncr_gw{NP.CUR_GD}_squad{tag}.md"
    write_comparison(df, squads, md, args.w, args.exclude)
    squads[blend_key].to_csv(NP.NCR / f"ncr_gw{NP.CUR_GD}_squad{tag}.csv", index=False)

    for name, sq in squads.items():
        capt = sq.loc[sq.is_capt, "name"]
        print(f"  {name:<22} £{sq.value.sum():5.1f}M  "
              f"N{int((sq.hemi==1).sum())}/S{int((sq.hemi==2).sum())}  "
              f"capt {capt.iloc[0] if len(capt) else '—'}")
    print(f"saved: {md.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
