from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from model.history import past_matches, utc_cutoff
from model.ncr.rp_rates import calibrate, norm, rp_rates

NCR = "data/ncr"


def _current() -> tuple[pd.Timestamp, int]:
    fx = pd.read_csv(f"{NCR}/ncr_fixtures.csv")
    cur = fx[fx.iscurrent == 1]
    return (pd.to_datetime(cur["game_date"]).min().normalize(), int(cur["gameday"].min())) if len(cur) else (pd.Timestamp("2026-07-04"), 1)


ASOF, CUR_GD = _current()
HALFLIFE_DAYS = 420.0

ATT = {
    "tries": 12, "try_assists": 5, "conversion_goals": 2, "missed_conversion_goals": -1, "penalty_goals": 3, "missed_penalty_goals": -1,
    "drop_goals_converted": 5, "defenders_beaten": 2, "offload": 2, "clean_breaks": 3,
}
DEF = {"tackles": 1, "missed_tackles": -1, "tackle_turnover": 4}
DISC = {"turnovers_conceded": -1, "penalties_conceded": -1, "yellow_cards": -5, "red_cards": -10}

REQUIRED = {"Prop": 2, "Hooker": 1, "Lock": 2, "Loose Forward": 3, "Scrum Half": 1, "Fly Half": 1, "Centre": 2, "Back Three": 3}
SKILL2POS = {
    "PROP": "Prop", "HOOKER": "Hooker", "LOCK": "Lock", "LOOSE FORWARD": "Loose Forward", "SCRUM-HALF": "Scrum Half", "FLY-HALF": "Fly Half",
    "CENTRE": "Centre", "BACK THREE": "Back Three",
}
FRONT_ROW = {"Prop", "Hooker"}
BUDGET, SQUAD, MAX_NATION, MAX_HEMI = 100.0, 16, 3, 10
BENCH_J2POS = {16: "Hooker", 17: "Prop", 18: "Prop", 19: "Lock", 20: "Loose Forward", 21: "Scrum Half", 22: "Fly Half", 23: "Back Three"}
POS_ORDER = {"Prop": 0, "Hooker": 1, "Lock": 2, "Loose Forward": 3, "Scrum Half": 4, "Fly Half": 5, "Centre": 6, "Back Three": 7}


def surkey(s):
    t = norm(s).split()
    return t[-1] if t else ""


PLAYER_NAME_ALIASES = {"nacho brex": "juan ignacio brex", "peni ravai kovekalou": "peni ravai"}


def match_players(pool, hist):
    hp = hist.groupby(["player_id", "player_name", "team"]).agg(matches=("fixture_id", "nunique")).reset_index()
    out = {}
    for _, pl in pool.iterrows():
        cand = hp[hp.team == pl.team_name]
        fn = PLAYER_NAME_ALIASES.get(norm(pl.full_name), norm(pl.full_name))
        hit = cand[cand.player_name.map(norm) == fn]
        if not len(hit):
            compact = fn.replace(" ", "")
            hit = cand[cand.player_name.map(lambda x: norm(x).replace(" ", "")) == compact]
        if not len(hit):
            sur = surkey(pl.full_name)
            c2 = cand[cand.player_name.map(surkey) == sur]
            if len(c2) == 1:
                if not fn or norm(c2.iloc[0].player_name)[:1] == fn[:1]:
                    hit = c2
            elif len(c2) > 1 and fn:
                ini = fn.split()[0][0]
                hit = c2[c2.player_name.map(lambda x: norm(x)[:1]) == ini]
        if len(hit):
            out[pl.id] = hit.sort_values("matches").iloc[-1].player_id
    return out


CLUB_ATT_CAL, CLUB_DEF_CAL, CLUB_DSC_CAL = 0.84, 0.97, 1.0
CLUB_CONF = 0.55


def _weighted_rates(g):
    wm = (g.w * g.minutes).sum()
    if wm <= 0:
        return None, 0.0
    rates = {e: (g.w * g[e]).sum() / wm * 80 for e in {**ATT, **DEF, **DISC}}
    return rates, wm


def player_profiles(hist, club=None, *, asof=None):
    prediction_asof = utc_cutoff(asof if asof is not None else ASOF).tz_convert(None)
    hist = past_matches(hist, prediction_asof)
    hist["date"] = pd.to_datetime(hist["date"])
    hist["w"] = 0.5 ** ((prediction_asof - hist["date"]).dt.days / HALFLIFE_DAYS)
    club_by_pid = {}
    if club is not None and len(club):
        club = past_matches(club, prediction_asof)
        club["date"] = pd.to_datetime(club["date"])
        club["w"] = 0.5 ** ((prediction_asof - club["date"]).dt.days / HALFLIFE_DAYS)
        club = club[club["w"] > 0.05]
        club_by_pid = {pid: g for pid, g in club.groupby("player_id")}
    prof = {}
    for pid, g in hist.groupby("player_id"):
        rates, wm = _weighted_rates(g)
        if rates is None:
            continue
        att = sum(rates[e] * v for e, v in ATT.items())
        dfn = sum(rates[e] * v for e, v in DEF.items())
        dsc = sum(rates[e] * v for e, v in DISC.items())
        cg = club_by_pid.get(pid)
        if cg is not None:
            crates, cwm = _weighted_rates(cg)
            if crates is not None:
                catt = sum(crates[e] * v for e, v in ATT.items()) * CLUB_ATT_CAL
                cdfn = sum(crates[e] * v for e, v in DEF.items()) * CLUB_DEF_CAL
                cdsc = sum(crates[e] * v for e, v in DISC.items()) * CLUB_DSC_CAL
                cw = CLUB_CONF * cwm
                att = (wm * att + cw * catt) / (wm + cw)
                dfn = (wm * dfn + cw * cdfn) / (wm + cw)
                dsc = (wm * dsc + cw * cdsc) / (wm + cw)
                wm = wm + cw
        starts, bench = g[g.started], g[~g.started]
        start_min = (starts.w * starts.minutes).sum() / starts.w.sum() if len(starts) and starts.w.sum() > 0 else 72
        bench_min = (bench.w * bench.minutes).sum() / bench.w.sum() if len(bench) and bench.w.sum() > 0 else np.nan
        prof[pid] = dict(
            att=att, dfn=dfn, dsc=dsc, wmin=wm, start_min=min(80, start_min), bench_min=min(35, max(8, bench_min)) if bench_min == bench_min else np.nan,
            n=g.fixture_id.nunique(),
        )
    return prof


def build_projection(exclude_teams=(), *, pool_override=None, asof=None, gameday=None):
    prediction_asof = utc_cutoff(asof if asof is not None else ASOF).tz_convert(None)
    pool = pool_override.copy() if pool_override is not None else pd.read_csv(f"{NCR}/ncr_players.csv")
    if exclude_teams:
        pool = pool[~pool.team_name.isin(exclude_teams)]
    hist = past_matches(pd.read_csv(f"{NCR}/ncr_player_match.csv"), prediction_asof)
    fx = pd.read_csv(f"{NCR}/ncr_fixtures.csv")
    wr = pd.read_csv("data/wr_rankings.csv")
    wr["snapshot_date"] = pd.to_datetime(wr["snapshot_date"])
    wr = wr[wr["snapshot_date"].lt(prediction_asof)]
    wr = wr[wr.snapshot_date == wr.snapshot_date.max()].set_index("team")["wr_pts"].to_dict()
    cur_gd = int(gameday) if gameday is not None else CUR_GD
    opp, home = {}, {}
    for _, r in fx[fx.gameday == cur_gd].iterrows():
        opp[r.home], opp[r.away], home[r.home], home[r.away] = r.away, r.home, True, False
    wx = {}
    for _, r in pd.read_csv(f"{NCR}/ncr_fixture_weather.csv").query("gameday == @cur_gd").iterrows():
        rain_pen = min(0.13, float(r.rain_mm or 0) * 0.02 + max(0.0, float(r.precip_prob or 0) - 50) * 0.0015)
        wx[r.home] = wx[r.away] = float(np.clip(1 - max(0.0, float(r.wind_kph or 0) - 20) * 0.004 - rain_pen, 0.82, 1.0))
    pid_map = match_players(pool, hist)
    prof = player_profiles(hist, pd.read_csv(f"{NCR}/club_player_match.csv"), asof=prediction_asof)
    bench = hist.assign(jersey=pd.to_numeric(hist["jersey"], errors="coerce"))
    bench = bench[(~bench["started"].astype(bool)) & bench["jersey"].between(16, 23)]
    bench_pos_prior = bench.assign(bpos=bench["jersey"].map(BENCH_J2POS)).groupby("bpos")["minutes"].mean().to_dict()
    bench_pos_prior.setdefault("Centre", bench_pos_prior.get("Back Three", 24.0))
    pid2name = hist.groupby("player_id")["player_name"].first().to_dict()
    api_players = pd.DataFrame([{"name_key": norm(pid2name.get(pid, "")), "att": p["att"], "dfn": p["dfn"]} for pid, p in prof.items() if p["wmin"] > 300])
    rp, fac = calibrate(rp_rates(asof=prediction_asof), api_players)
    rp_by_key = rp.set_index("name_key")[["att", "dfn", "dsc", "rp_min"]].to_dict("index")
    print(f"RugbyPass prior: {len(rp_by_key)} players, calibration {fac}")
    known = [(SKILL2POS[pl.skill_desc], prof.get(pid_map.get(pl.id))) for _, pl in pool.iterrows()]
    base = {}
    for pos in REQUIRED:
        ps = [p for q, p in known if q == pos and p]
        base[pos] = (
            {k: float(np.average([p[k] for p in ps], weights=np.array([p["wmin"] for p in ps]))) for k in ("att", "dfn", "dsc")}
            if ps
            else {"att": 6, "dfn": 8, "dsc": -1}
        )
    K = 220.0
    recs = []
    for _, pl in pool.iterrows():
        if pl.player_status not in ("P", "B"):
            continue
        status, pos, team = pl.player_status, SKILL2POS[pl.skill_desc], pl.team_name
        p, b, rpr = prof.get(pid_map.get(pl.id)), base[SKILL2POS[pl.skill_desc]], rp_by_key.get(norm(pl.full_name))
        if rpr:
            rc = min(rpr["rp_min"], 900) / 900 * 200
            prior, rp_min = {k: (rc * rpr[k] + 80 * b[k]) / (rc + 80) for k in ("att", "dfn", "dsc")}, rpr["rp_min"]
        else:
            prior, rp_min = b, 0
        if p:
            W = p["wmin"]
            att, dfn, dsc = ((W * p[k] + K * prior[k]) / (W + K) for k in ("att", "dfn", "dsc"))
            start_min = p["start_min"] if p["start_min"] > 20 else 72
            bench_min = p["bench_min"] if p["bench_min"] == p["bench_min"] else bench_pos_prior.get(pos, 24.0)
            n = p["n"]
        else:
            att, dfn, dsc = prior["att"], prior["dfn"], prior["dsc"]
            start_min, bench_min, n = 70, bench_pos_prior.get(pos, 24.0), 0
        if pos in FRONT_ROW:
            dfn += 2.0
        margin = 2.0 * (wr.get(team, 80) - wr.get(opp.get(team), 80) + (3 if home.get(team) else -3))
        per80 = att * float(np.clip(1 + margin / 60.0, 0.72, 1.35)) * wx.get(team, 1.0) + dfn * float(np.clip(1 - margin / 90.0, 0.80, 1.28)) + dsc
        exp_min, potm_cap, sub_mult = (start_min, 0.13, 0.5) if status == "P" else (bench_min, 0.05, 3.0)
        base_pts = per80 * (exp_min / 80)
        expected_pts = base_pts + float(np.clip(base_pts / 145, 0, potm_cap) * 15)
        recs.append(
            dict(
                id=pl.id, name=pl.full_name, team=team, hemi=pl.hemisphere, pos=pos, value=pl.value, sel=pl.sel_percentage, status=status, opp=opp.get(team),
                home=home.get(team), n_hist=n, rp_min=int(rp_min), wr_margin=round(margin, 1), weather_att=round(wx.get(team, 1.0), 3), per80=round(per80, 2),
                exp_min=round(exp_min, 1), starter_exp=round(expected_pts, 2),
                supersub_exp=round(expected_pts * sub_mult, 2),
            )
        )
    return pd.DataFrame(recs)


def optimise(df):
    n = len(df)
    starter, sub = df.starter_exp.values, df.supersub_exp.values
    Z, I = np.zeros(n), np.eye(n)
    row = lambda squad, supersub, captain: np.concatenate([squad, supersub, captain])
    cons = [
        LinearConstraint(row(np.ones(n), Z, Z), SQUAD, SQUAD), LinearConstraint(row(Z, np.ones(n), Z), 1, 1), LinearConstraint(row(Z, Z, np.ones(n)), 1, 1),
        LinearConstraint(np.hstack([-I, I, np.zeros((n, n))]), -np.inf, 0), LinearConstraint(np.hstack([-I, I, I]), -np.inf, 0),
        LinearConstraint(row(df.value.values, Z, Z), -np.inf, BUDGET),
    ]
    cons += [LinearConstraint(row((df.team == t).values.astype(float), Z, Z), -np.inf, MAX_NATION) for t in df.team.unique()]
    cons += [LinearConstraint(row((df.hemi == h).values.astype(float), Z, Z), -np.inf, MAX_HEMI) for h in (1, 2)]
    cons += [LinearConstraint(row((df.pos == pos).values.astype(float), -(df.pos == pos).values.astype(float), Z), req, req) for pos, req in REQUIRED.items()]
    cons += [
        LinearConstraint(row(Z, Z, (df.status != "P").values.astype(float)), 0, 0), LinearConstraint(row(Z, (df.status != "B").values.astype(float), Z), 0, 0),
    ]
    res = milp(c=-np.concatenate([starter, sub - starter, starter]), constraints=cons, integrality=np.ones(3 * n), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError("MILP failed: " + res.message)
    df = df.copy()
    df["in_squad"], df["is_sub"], df["is_capt"] = res.x[:n] > 0.5, res.x[n : 2 * n] > 0.5, res.x[2 * n :] > 0.5
    return df[df.in_squad].copy()


def render(df, squad, tag, exclude):
    squad = squad.copy()
    squad["role"] = np.where(squad.is_sub, "SUPER SUB", np.where(squad.is_capt, "CAPTAIN", "Starter"))
    squad["proj"] = np.where(squad.is_sub, squad.supersub_exp, np.where(squad.is_capt, 2 * squad.starter_exp, squad.starter_exp))
    squad["po"] = squad.pos.map(POS_ORDER)
    squad = squad.sort_values(["is_sub", "po", "starter_exp"], ascending=[True, True, False])
    print(f"\n{'=' * 78}\nNCR OPTIMAL GW{CUR_GD} SQUAD  (empirical projection)\n{'=' * 78}")
    print(f"{'Pos':<13}{'Player':<24}{'Nation':<13}{'£':>5}{'xMin':>6}{'Proj':>7}  Role")
    print("-" * 78)
    for _, r in squad.iterrows():
        print(f"{r.pos:<13}{r['name'][:23]:<24}{r.team:<13}{r.value:>5.1f}{r.exp_min:>6.0f}{r.proj:>7.1f}{'  ← ' + r.role if r.role != 'Starter' else ''}")
    tot_cost, tot_proj = squad.value.sum(), squad.proj.sum()
    north, south = int((squad.hemi == 1).sum()), int((squad.hemi == 2).sum())
    print("-" * 78)
    print(f"squad cost: £{tot_cost:.1f}M / {BUDGET:.0f}M   projected: {tot_proj:.1f} pts   N{north}/S{south}  nations {squad.team.nunique()}")
    topcaps = df[df.status == "P"].sort_values("starter_exp", ascending=False).head(6)
    print("Top captaincy options:", ", ".join(f"{r['name']} {r.starter_exp:.1f}" for _, r in topcaps.iterrows()))
    squad.to_csv(f"{NCR}/ncr_gw{CUR_GD}_squad{tag}.csv", index=False)
    L = [f"# Nations Championship Fantasy — GW{CUR_GD} optimal squad", ""]
    if exclude:
        L += [f"> Excludes players from already-started matches: **{', '.join(exclude)}**.", ""]
    L += [
        f"**Budget:** £{tot_cost:.1f}M / {BUDGET:.0f}M  ·  **North {north} / South {south}**  ·  **{squad.team.nunique()} nations**  ·  "
        f"**Projected: {tot_proj:.1f} pts** (captain 2x, super sub 3x off bench)",
        "",
        "| Pos | Player | Nation | £ | xMin | xPts | Role |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    roles = {"CAPTAIN": "**CAPTAIN** 2x → {:.1f}", "SUPER SUB": "**SUPER SUB** 3x off bench → {:.1f}", "Starter": ""}
    L += [
        f"| {r.pos} | {'**' + r['name'] + '**' if r.role != 'Starter' else r['name']} | {r.team} | {r.value:.1f} | {r.exp_min:.0f} | "
        f"{r.starter_exp:.1f} | {roles[r.role].format(r.proj)} |"
        for _, r in squad.iterrows()
    ]
    capt = squad.loc[squad.is_capt, "name"]
    if len(capt):
        L += ["", f"**Captain:** {capt.iloc[0]}"]
        top_name = topcaps.iloc[0]["name"] if len(topcaps) else None
        if top_name and top_name != capt.iloc[0]:
            L.append(
                f"> Note: not the highest xPts starter ({top_name}) — the armband was set by a dedicated captain head, "
                "which ranks captaincy rather than raw points."
            )
    L += [
        "",
        "**Captaincy options (highest xPts among confirmed starters, 1x):**",
        *(f"- {r['name']} ({r.team}, {r.pos}) — {r.starter_exp:.1f} xPts  ·  £{r.value}" for _, r in topcaps.iterrows()),
    ]
    L += ["", "*xPts = expected match points at 1x. xMin = expected minutes.*"]
    open(f"{NCR}/ncr_gw{CUR_GD}_squad{tag}.md", "w").write("\n".join(L) + "\n")
    print(f"saved: ncr_gw{CUR_GD}_squad{tag}.csv, ncr_gw{CUR_GD}_squad{tag}.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[])
    exclude = ap.parse_args().exclude
    tag = "_ex_" + "_".join(t.replace(" ", "") for t in exclude) if exclude else ""
    df = build_projection(exclude_teams=exclude)
    df.to_csv(f"{NCR}/ncr_gw{CUR_GD}_projections{tag}.csv", index=False)
    render(df, optimise(df), tag, exclude)
