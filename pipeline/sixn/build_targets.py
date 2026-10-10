from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.sixn.official_labels import match_official

SIX_NATIONS = 1266
TARGET_SEASONS = [2023, 2024, 2025, 2026]
MODERN_SEASONS = [2025, 2026]
FORWARD_GROUPS = {"Prop", "Hooker", "Second-row", "Back-row"}
JUMPER_GROUPS = {"Second-row", "Back-row"}

COMPONENTS = [
    "tries", "try_assists", "conversion_goals", "penalty_goals", "drop_goals_converted", "metres", "defenders_beaten", "clean_breaks", "offload", "runs",
    "tackles", "missed_tackles", "tackle_turnover", "passes", "penalties_conceded", "yellow_cards", "red_cards",
]


def score_components(c: dict, is_forward: bool) -> float:
    try_val = 15 if is_forward else 10
    return float(
        try_val * c["tries"]
        + 1.0 * c["tackles"]
        + c["metres"] // 10
        + 4 * c["try_assists"]
        + 2 * c["conversion_goals"]
        + 3 * c["penalty_goals"]
        + 4 * c["drop_goals_converted"]
        + 2 * c["defenders_beaten"]
        + 2 * c["offload"]
        + 5 * c["tackle_turnover"]
        - 1 * c["penalties_conceded"]
        - 5 * c["yellow_cards"]
        - 8 * c["red_cards"]
    )


def build() -> tuple[pd.DataFrame, dict]:
    ap = pd.read_csv("data/api_player_match.csv")
    six = ap[(ap.comp_id == SIX_NATIONS) & (ap.season.isin(TARGET_SEASONS))].copy()

    feat = pd.read_csv("data/model_player_match.csv")[["fixture_id", "player_id", "canonical_pos", "is_forward"]]
    six = six.merge(feat, on=["fixture_id", "player_id"], how="left", suffixes=("", "_feat"))
    six["canonical_pos"] = six["canonical_pos_feat"].fillna(six["canonical_pos"])
    six["is_forward"] = six["is_forward_feat"].fillna(six["is_forward"]).astype(bool)
    six["is_jumper"] = six["canonical_pos"].isin(JUMPER_GROUPS)

    for col in COMPONENTS:
        six[col] = pd.to_numeric(six[col], errors="coerce").fillna(0)
    six["minutes"] = pd.to_numeric(six["minutes"], errors="coerce").fillna(0)
    six["min_share"] = (six["minutes"] / 80.0).clip(0, 1)

    six["recon_pts"] = [score_components({c: getattr(r, c) for c in COMPONENTS}, bool(r.is_forward)) for r in six.itertuples(index=False)]

    tm = pd.read_csv("data/api_team_match.csv")[["fixture_id", "team", "scrums_won", "lineout_won_steal"]]
    for c in ["scrums_won", "lineout_won_steal"]:
        tm[c] = pd.to_numeric(tm[c], errors="coerce").fillna(0)
    six = six.merge(tm.rename(columns={"scrums_won": "team_scrums_won", "lineout_won_steal": "team_lineout_steal"}), on=["fixture_id", "team"], how="left")
    six[["team_scrums_won", "team_lineout_steal"]] = six[["team_scrums_won", "team_lineout_steal"]].fillna(0)

    off = pd.read_csv("data/official_player_match.csv")
    labels = match_official(six, off.dropna(subset=['Pts']))
    six['official_pts'] = labels['Pts']
    six['potm_winner'] = labels['POTM']
    six['official_label_source'] = labels['label_source'].fillna('missing')
    unused = six.season.isin(MODERN_SEASONS) & six.official_pts.isna() & ~six.started.astype(bool) & six.minutes.eq(0) & six[COMPONENTS].eq(0).all(axis=1)
    six.loc[unused, ['official_pts', 'potm_winner']] = 0.0
    six.loc[unused, 'official_label_source'] = 'unused_bench_no_recorded_events'
    six["has_label"] = six["official_pts"].notna()
    six["is_modern"] = six["season"].isin(MODERN_SEASONS)

    six["sw_drive"] = six["is_forward"].astype(float) * six["team_scrums_won"] * six["min_share"]
    six["ls_drive"] = six["is_jumper"].astype(float) * six["team_lineout_steal"] * six["min_share"]

    fit = _fit_latent(six)

    six["lat_sw"] = fit["a"] * six["sw_drive"]
    six["lat_ls"] = fit["b"] * six["ls_drive"]
    pos_potm = six["canonical_pos"].map(fit["potm_base"]).fillna(fit["potm_base_global"])
    six["lat_potm"] = np.where(six["has_label"], 15.0 * six["potm_winner"].fillna(0), pos_potm)
    six["lat_other"] = six["canonical_pos"].map(fit["other_const"]).fillna(fit["other_global"])
    six["latent_total"] = six[["lat_sw", "lat_ls", "lat_potm", "lat_other"]].sum(axis=1)
    six["target_pts"] = six["recon_pts"] + six["latent_total"]

    out = six.rename(columns={c: f"y_{c}" for c in COMPONENTS})
    keep = (
        [
            "season", "comp_id", "round", "fixture_id", "date", "team", "player_id", "player_name", "canonical_pos", "is_forward", "is_jumper", "is_modern",
            "minutes", "started", "min_share", "team_scrums_won", "team_lineout_steal",
        ]
        + [f"y_{c}" for c in COMPONENTS]
        + [
            "recon_pts", "official_pts", "potm_winner", "has_label", "official_label_source", "lat_sw", "lat_ls", "lat_potm", "lat_other", "latent_total",
            "target_pts",
        ]
    )
    return out[keep], fit


def _fit_latent(six: pd.DataFrame) -> dict:
    lab = six[six.has_label & six.is_modern].copy()
    y = (lab["official_pts"] - lab["recon_pts"] - 15.0 * lab["potm_winner"].fillna(0)).to_numpy(float)

    positions = sorted(lab["canonical_pos"].dropna().unique())
    pos_idx = {p: i for i, p in enumerate(positions)}
    n, k = len(lab), len(positions)
    X = np.zeros((n, k + 2))
    for i, p in enumerate(lab["canonical_pos"]):
        if p in pos_idx:
            X[i, pos_idx[p]] = 1.0
    X[:, k] = lab["sw_drive"].to_numpy(float)
    X[:, k + 1] = lab["ls_drive"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    other_const = {p: float(coef[pos_idx[p]]) for p in positions}
    a, b = float(coef[k]), float(coef[k + 1])

    potm_base = lab.groupby("canonical_pos").apply(lambda g: 15.0 * g["potm_winner"].fillna(0).mean())
    return {
        "a": a, "b": b, "other_const": other_const, "other_global": float(np.mean(list(other_const.values()))), "potm_base": potm_base.to_dict(),
        "potm_base_global": float(15.0 * lab["potm_winner"].fillna(0).mean()), "positions": positions,
    }


def main():
    tgt, fit = build()
    out = "data/model_targets.csv"
    tgt.to_csv(out, index=False)

    lab = tgt[tgt.has_label & tgt.is_modern]
    mae_recon = float((lab.official_pts - lab.recon_pts).abs().mean())
    mae_flat = float(
        (lab.official_pts - (lab.recon_pts + lab.canonical_pos.map(lab.assign(r=lab.official_pts - lab.recon_pts).groupby("canonical_pos")["r"].mean())))
        .abs()
        .mean()
    )
    mae_decomp = float((lab.official_pts - lab.target_pts).abs().mean())
    print(f"✅  {len(tgt)} rows, {tgt.fixture_id.nunique()} matches, seasons {sorted(tgt.season.unique())}")
    print(f"    modern (2025+26) labelled rows used for fit/MAE: {len(lab)}")
    print(f"    MAE vs official:  recon={mae_recon:.2f}  +flat-residual={mae_flat:.2f}  +decomposed-latent={mae_decomp:.2f}")
    print(f"    fit: SW a={fit['a']:.2f} pt/(scrum·share)   LS b={fit['b']:.2f} pt/(steal·share)   POTM coeff=15 (realised)")
    print("    mean latent piece by position (fwd/bck):")
    g = lab.groupby("canonical_pos")[["lat_sw", "lat_ls", "lat_potm", "lat_other", "latent_total"]].mean().sort_values("latent_total", ascending=False)
    for pos, row in g.iterrows():
        f = "fwd" if pos in FORWARD_GROUPS else "bck"
        print(
            f"        {str(pos):12s} ({f})  sw={row.lat_sw:4.1f} ls={row.lat_ls:4.1f} "
            f"potm={row.lat_potm:4.1f} other={row.lat_other:5.1f}  total={row.latent_total:5.1f}"
        )
    fwd = lab.is_forward
    print(f"    forward mean latent {lab[fwd].latent_total.mean():+.1f} vs back {lab[~fwd].latent_total.mean():+.1f}")
    print(f"💾  {out}")


if __name__ == "__main__":
    main()
