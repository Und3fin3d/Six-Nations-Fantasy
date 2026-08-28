#!/usr/bin/env python3
"""
Nations Championship Rugby (NCR) fantasy — GW1 projection & squad optimiser
===========================================================================
Projects expected GW1 fantasy points for every player in the NCR fantasy pool
(all 12 nations, both hemispheres) using the game's own scoring rubric applied
to recency-weighted per-80 event rates from international match history, then
solves the 16-man squad as a MILP under the game's constraints.

Scoring rubric (from the game's Rules page):
  Try +12, Try Assist +5, Conversion +2 / miss -1, Penalty +3 / miss -1,
  Drop Goal +5, Defender Beaten +2, Offload +2, Line Break +3,
  Tackle +1, Missed Tackle -1, Turnover Won +4, Own Lineout Won +1,
  Error (knock-on/fwd pass) -1, Penalty Conceded -1, Scrum Won +2 (front row),
  Player of the Match +15, Yellow -5, Red -10.
  (Interception +5 and Lineout Steal +5 are not in the per-player feed → omitted.)

Squad rules: 16 players (XV + 1 Super Sub any position), budget 100M,
  max 3 per nation, max 10 per hemisphere. Captain 2x. Super Sub: 3x if it
  comes off the bench, 0.5x if it starts, 0 if it does not play.
"""
from __future__ import annotations
import re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NCR = DATA / "ncr"


def _current_asof(default="2026-07-04") -> pd.Timestamp:
    """Prediction date = kickoff of the current gameday. Derived, never hardcoded:
    a stale ASOF silently mis-weights recency and lets a played gameweek leak in."""
    try:
        fx = pd.read_csv(NCR / "ncr_fixtures.csv")
        cur = fx[fx.get("iscurrent", 0) == 1]
        if len(cur):
            return pd.to_datetime(cur["game_date"]).min().normalize()
    except Exception:
        pass
    return pd.Timestamp(default)


def _current_gameday(default=1) -> int:
    try:
        fx = pd.read_csv(NCR / "ncr_fixtures.csv")
        cur = fx[fx.get("iscurrent", 0) == 1]
        if len(cur):
            return int(cur["gameday"].min())
    except Exception:
        pass
    return default


ASOF = _current_asof()
CUR_GD = _current_gameday()      # output files are named per gameweek, never overwritten
HALFLIFE_DAYS = 420.0          # recency half-life for rate estimation

# ── NCR scoring rubric, split into attack / defence / discipline components ──
ATT = {"tries": 12, "try_assists": 5, "conversion_goals": 2, "missed_conversion_goals": -1,
       "penalty_goals": 3, "missed_penalty_goals": -1, "drop_goals_converted": 5,
       "defenders_beaten": 2, "offload": 2, "clean_breaks": 3}
# NB: the API credits the hooker (thrower) with EVERY won lineout (~12/80) while
# jumpers get 0, so "Own Lineout Won +1" is dropped — it is a pure hooker-inflation
# artifact here, not a real per-player differentiator.
DEF = {"tackles": 1, "missed_tackles": -1, "tackle_turnover": 4}
DISC = {"turnovers_conceded": -1, "penalties_conceded": -1, "yellow_cards": -5, "red_cards": -10}

REQUIRED = {"Prop": 2, "Hooker": 1, "Lock": 2, "Loose Forward": 3,
            "Scrum Half": 1, "Fly Half": 1, "Centre": 2, "Back Three": 3}  # 15 starters
SKILL2POS = {"PROP": "Prop", "HOOKER": "Hooker", "LOCK": "Lock", "LOOSE FORWARD": "Loose Forward",
             "SCRUM-HALF": "Scrum Half", "FLY-HALF": "Fly Half", "CENTRE": "Centre",
             "BACK THREE": "Back Three"}
FRONT_ROW = {"Prop", "Hooker"}
BUDGET, SQUAD, MAX_NATION, MAX_HEMI = 100.0, 16, 3, 10


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s.lower())).strip()


def surkey(s):
    t = norm(s).split()
    return t[-1] if t else ""


# Fantasy and match feeds occasionally disagree on a nickname or append a
# surname that the API omits. Keep the small semantic exceptions explicit;
# punctuation-only differences are handled generically by compact matching.
PLAYER_NAME_ALIASES = {
    "nacho brex": "juan ignacio brex",
    "peni ravai kovekalou": "peni ravai",
}


# ── 1. match fantasy pool → historical player_ids (team-constrained) ─────────
def match_players(pool, hist):
    hp = (hist.groupby(["player_id", "player_name", "team"])
              .agg(matches=("fixture_id", "nunique")).reset_index())
    out = {}
    for _, pl in pool.iterrows():
        cand = hp[hp.team == pl.team_name]
        fn = PLAYER_NAME_ALIASES.get(norm(pl.full_name), norm(pl.full_name))
        hit = cand[cand.player_name.map(norm) == fn]
        if not len(hit):
            # Apostrophes are inconsistent across the two feeds: OToole vs
            # O'Toole, Taukeiaho vs Taukei'aho, Vaa'i vs Vaai, etc.
            compact = fn.replace(" ", "")
            hit = cand[cand.player_name.map(lambda x: norm(x).replace(" ", "")) == compact]
        if not len(hit):
            sur = surkey(pl.full_name)
            c2 = cand[cand.player_name.map(surkey) == sur]
            if len(c2) == 1:
                # A unique surname is not enough when both names expose different
                # first initials (e.g. Kane James must not inherit Eddie James's
                # history merely because Kane has no historical row yet).
                if not fn or norm(c2.iloc[0].player_name)[:1] == fn[:1]:
                    hit = c2
            elif len(c2) > 1 and fn:
                ini = fn.split()[0][0]
                hit = c2[c2.player_name.map(lambda x: norm(x)[:1]) == ini]
        if len(hit):
            out[pl.id] = hit.sort_values("matches").iloc[-1].player_id
    return out


# club rate contribution: club rugby inflates attack vs test level, so club event
# contributions are calibrated down (matches the RugbyPass club→test factor) and
# carry a lower confidence weight. Club data informs the RATE only — never minutes
# or availability (a Super Rugby starter can be a test-bench player).
CLUB_ATT_CAL, CLUB_DEF_CAL, CLUB_DSC_CAL = 0.84, 0.97, 1.0
CLUB_CONF = 0.55            # club weighted-minutes count 0.55× a test's toward `wmin`


def _weighted_rates(g):
    wm = (g.w * g.minutes).sum()
    if wm <= 0:
        return None, 0.0
    rates = {e: (g.w * g[e]).sum() / wm * 80 for e in {**ATT, **DEF, **DISC}}
    return rates, wm


# ── 2. recency-weighted per-80 rates + expected-minutes profile per player ───
def player_profiles(hist, club=None, *, asof=None):
    prediction_asof = pd.Timestamp(asof) if asof is not None else ASOF
    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist["w"] = 0.5 ** ((prediction_asof - hist["date"]).dt.days / HALFLIFE_DAYS)
    club_by_pid = {}
    if club is not None and len(club):
        club = club.copy()
        club["date"] = pd.to_datetime(club["date"])
        club["w"] = 0.5 ** ((prediction_asof - club["date"]).dt.days / HALFLIFE_DAYS)
        club = club[club["w"] > 0.05]                    # ignore near-dead-weight club history
        club_by_pid = {pid: g for pid, g in club.groupby("player_id")}
    prof = {}
    for pid, g in hist.groupby("player_id"):
        rates, wm = _weighted_rates(g)
        if rates is None:
            continue
        att = sum(rates[e] * v for e, v in ATT.items())
        dfn = sum(rates[e] * v for e, v in DEF.items())
        dsc = sum(rates[e] * v for e, v in DISC.items())
        # blend in calibrated club form (rate only), weighted by CLUB_CONF·club-minutes
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
                wm = wm + cw                             # richer rate → more shrinkage confidence
        starts = g[g.started]
        bench = g[~g.started]
        p_start = (starts.w.sum()) / g.w.sum()
        start_min = (starts.w * starts.minutes).sum() / starts.w.sum() if len(starts) and starts.w.sum() > 0 else 72
        bench_min = ((bench.w * bench.minutes).sum() / bench.w.sum()
                     if len(bench) and bench.w.sum() > 0 else np.nan)   # NaN → position prior
        prof[pid] = dict(att=att, dfn=dfn, dsc=dsc, wmin=wm, p_start=p_start,
                         start_min=min(80, start_min),
                         bench_min=min(35, max(8, bench_min)) if bench_min == bench_min else np.nan,
                         n=g.fixture_id.nunique())
    return prof


def price_minutes(v):                                    # fallback expected minutes from price
    return float(np.interp(v, [2, 3, 3.5, 4, 4.5, 5, 6, 7, 8.5],
                              [10, 16, 24, 32, 42, 54, 66, 72, 76]))


def recent_appearance_share(hist, last_n=6):
    """Hemisphere-neutral 'nailed-on' signal: recency-weighted share of each
    team's last N tests in which the player featured (≥20 min). Avoids penalising
    Southern players just because their season ended in Nov vs the North's March."""
    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    share = {}
    for team, g in hist.groupby("team"):
        fixtures = (g.drop_duplicates("fixture_id").sort_values("date").tail(last_n))
        fids = fixtures.fixture_id.tolist()
        fw = {r.fixture_id: 0.85 ** i for i, (_, r) in
              enumerate(fixtures.sort_values("date", ascending=False).iterrows())}
        tot = sum(fw.values())
        gg = g[g.fixture_id.isin(fids) & (g.minutes >= 20)]
        for pid, pg in gg.groupby("player_id"):
            share[pid] = sum(fw[f] for f in pg.fixture_id.unique()) / tot
    return share


def build_projection(
    exclude_teams=(), override_rates=None, *, pool_override=None,
    asof=None, gameday=None,
):
    """override_rates: optional DataFrame [fantasy_id, att, dfn, dsc] of per-80
    NCR-point components from an external model (e.g. the research.py rate heads);
    when given it replaces the empirical/RugbyPass per-80 estimate, keeping the
    same matchup / minutes / scoring / optimiser downstream."""
    ov = ({int(r.fantasy_id): (r.att, r.dfn, r.dsc) for r in override_rates.itertuples()}
          if override_rates is not None else {})
    prediction_asof = pd.Timestamp(asof) if asof is not None else ASOF
    pool = (
        pool_override.copy() if pool_override is not None
        else pd.read_csv(NCR / "ncr_players.csv")
    )
    if exclude_teams:
        pool = pool[~pool.team_name.isin(exclude_teams)]   # e.g. matches already kicked off
    hist = pd.read_csv(NCR / "ncr_player_match.csv")
    teams = pd.read_csv(NCR / "ncr_teams.csv")
    fx = pd.read_csv(NCR / "ncr_fixtures.csv")
    wr = pd.read_csv(DATA / "wr_rankings.csv")
    wr["snapshot_date"] = pd.to_datetime(wr["snapshot_date"])
    point_in_time_wr = wr[wr["snapshot_date"].lt(prediction_asof)]
    if point_in_time_wr.empty:
        point_in_time_wr = wr
    wr = point_in_time_wr[
        point_in_time_wr.snapshot_date == point_in_time_wr.snapshot_date.max()
    ].set_index("team")["wr_pts"].to_dict()

    # matchup context for the CURRENT gameday (iscurrent flag; falls back to GW1)
    cur_gd = (
        int(gameday) if gameday is not None else
        int(fx.loc[fx.get("iscurrent", 0) == 1, "gameday"].min())
        if (fx.get("iscurrent", 0) == 1).any() else 1
    )
    gwc = fx[fx.gameday == cur_gd]
    opp, home = {}, {}
    for _, r in gwc.iterrows():
        opp[r.home] = r.away; opp[r.away] = r.home
        home[r.home] = True; home[r.away] = False

    # venue weather (physical conditions, not a third-party forecast of the result)
    wx = {}
    wx_path = NCR / "ncr_fixture_weather.csv"
    if wx_path.exists():
        wd = pd.read_csv(wx_path)
        for _, r in wd[wd.gameday == cur_gd].iterrows():
            # attacking suppression from wind (>20kph) and rain / high precip probability
            wind_pen = max(0.0, float(r.wind_kph or 0) - 20) * 0.004
            rain_pen = min(0.13, float(r.rain_mm or 0) * 0.02 + max(0.0, float(r.precip_prob or 0) - 50) * 0.0015)
            fac = float(np.clip(1 - wind_pen - rain_pen, 0.82, 1.0))
            wx[r.home] = wx[r.away] = fac

    pid_map = match_players(pool, hist)
    # per-match club form (URC/Prem/Top14/Cups/Super Rugby/Japan LO) — same API
    # player_id space; folded into the FORM rate (calibrated, recency-weighted).
    club = None
    club_path = NCR / "club_player_match.csv"
    if club_path.exists() and not ov:                    # skip when an external model overrides rates
        club = pd.read_csv(club_path)
    prof = player_profiles(hist, club, asof=prediction_asof)

    # bench-minute prior by SLOT (GW1 audit: j18 props ~32 min vs j21 scrum-halves ~16;
    # a flat fallback misprices the super sub). Bench jersey → fantasy position group.
    BENCH_J2POS = {16: "Hooker", 17: "Prop", 18: "Prop", 19: "Lock", 20: "Loose Forward",
                   21: "Scrum Half", 22: "Fly Half", 23: "Back Three"}
    _h = hist.copy()
    _h["jersey"] = pd.to_numeric(_h["jersey"], errors="coerce")
    _b = _h[(~_h["started"].astype(bool)) & _h["jersey"].between(16, 23)]
    _b = _b.assign(bpos=_b["jersey"].map(BENCH_J2POS))
    bench_pos_prior = _b.groupby("bpos")["minutes"].mean().to_dict()
    bench_pos_prior.setdefault("Centre", bench_pos_prior.get("Back Three", 24.0))

    # RugbyPass per-80 rates (club+international form), calibrated to test level on
    # the players who appear in both sources → a player-specific shrinkage prior.
    # Skipped entirely when an external model supplies the rates (override_rates).
    rp_by_key = {}
    if not ov:                                            # external model supplies rates → skip
        try:
            try:
                import rp_rates as RP
            except ModuleNotFoundError:
                from model import rp_rates as RP
            pid2name = hist.groupby("player_id")["player_name"].first().to_dict()
            api_players = pd.DataFrame([
                {"name_key": RP._norm(pid2name.get(pid, "")), "att": p["att"], "dfn": p["dfn"]}
                for pid, p in prof.items() if p["wmin"] > 300])
            rp = RP.rp_rates()
            rp, fac = RP.calibrate(rp, api_players)
            rp_by_key = rp.set_index("name_key")[["att", "dfn", "dsc", "rp_min"]].to_dict("index")
            print(f"RugbyPass prior: {len(rp_by_key)} players, calibration {fac}")
        except Exception as e:
            print(f"RugbyPass prior unavailable ({e}); using position baselines only")

    def rp_lookup(full_name):
        return rp_by_key.get(re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ",
               unicodedata.normalize("NFKD", str(full_name)).encode("ascii", "ignore")
               .decode().lower())).strip())

    # position baselines (weighted-mean per-80 components) for shrinkage / fallbacks
    rows = []
    for _, pl in pool.iterrows():
        pos = SKILL2POS[pl.skill_desc]
        p = prof.get(pid_map.get(pl.id))
        rows.append((pl.id, pos, p))
    base = {}
    for pos in REQUIRED:
        ps = [p for (_, q, p) in rows if q == pos and p]
        if ps:
            W = np.array([p["wmin"] for p in ps])
            base[pos] = {k: float(np.average([p[k] for p in ps], weights=W)) for k in ("att", "dfn", "dsc")}
        else:
            base[pos] = {"att": 6, "dfn": 8, "dsc": -1}

    K = 220.0     # shrinkage strength (≈ 2.7 full matches of minutes)
    recs = []
    for _, pl in pool.iterrows():
        status = pl.player_status if pl.player_status in ("P", "B") else "OUT"
        if status == "OUT":
            continue                                     # not in the confirmed matchday 23
        pos = SKILL2POS[pl.skill_desc]
        team = pl.team_name
        p = prof.get(pid_map.get(pl.id))
        b = base[pos]
        # shrinkage prior = RugbyPass rate (if any) blended with the position baseline
        rpr = rp_lookup(pl.full_name)
        if rpr:
            rc = min(rpr["rp_min"], 900) / 900 * 200      # RP confidence from weighted minutes
            prior = {k: (rc * rpr[k] + 80 * b[k]) / (rc + 80) for k in ("att", "dfn", "dsc")}
            rp_min = rpr["rp_min"]
        else:
            prior = b
            rp_min = 0
        if p:                                            # blend API international history → prior
            W = p["wmin"]
            att = (W * p["att"] + K * prior["att"]) / (W + K)
            dfn = (W * p["dfn"] + K * prior["dfn"]) / (W + K)
            dsc = (W * p["dsc"] + K * prior["dsc"]) / (W + K)
            start_min = p["start_min"] if p["start_min"] > 20 else 72
            bench_min = (p["bench_min"] if p["bench_min"] == p["bench_min"]
                         else bench_pos_prior.get(pos, 24.0))
            n = p["n"]
        else:                                            # no caps → RugbyPass prior (or baseline)
            att, dfn, dsc = prior["att"], prior["dfn"], prior["dsc"]
            start_min, bench_min = 70, bench_pos_prior.get(pos, 24.0)
            n = 0
        if pl.id in ov:                                  # external model rates override
            att, dfn, dsc = ov[pl.id]
        if pos in FRONT_ROW:
            dfn += 2.0                                    # scrum-won bonus (≈1 scrum/80 while on)

        # Matchup predicted from World Rugby ratings — the model forecasts the game
        # itself rather than reading a bookmaker's line back in. A rating point is
        # worth ~2 points of margin; home advantage ~3 rating points.
        #   attacking output scales with expected dominance;
        #   tackle volume rises for the defending underdog; weather suppresses attack.
        margin = 2.0 * (wr.get(team, 80) - wr.get(opp.get(team), 80)
                        + (3 if home.get(team) else -3))
        att_mult = float(np.clip(1 + margin / 60.0, 0.72, 1.35))
        def_mult = float(np.clip(1 - margin / 90.0, 0.80, 1.28))
        weather_att = wx.get(team, 1.0)
        per80 = att * att_mult * weather_att + dfn * def_mult + dsc

        # confirmed lineup → expected minutes and multiplier are known
        if status == "P":                                # named starter
            exp_min = start_min
            base_pts = per80 * (exp_min / 80)
            potm = float(np.clip(base_pts / 145, 0, 0.13) * 15)
            starter_exp = base_pts + potm
            supersub_exp = per80 * (start_min / 80) * 0.5    # a starter used as sub → 0.5x
        else:                                            # named on the bench (impact sub)
            exp_min = bench_min
            base_pts = per80 * (exp_min / 80)
            potm = float(np.clip(base_pts / 145, 0, 0.05) * 15)
            starter_exp = base_pts + potm                    # scores 1x its bench minutes
            supersub_exp = per80 * (bench_min / 80) * 3.0    # bench player as super sub → 3x

        recs.append(dict(id=pl.id, name=pl.full_name, team=team, hemi=pl.hemisphere,
                         pos=pos, value=pl.value, sel=pl.sel_percentage, status=status,
                         opp=opp.get(team), home=home.get(team), n_hist=n, rp_min=int(rp_min),
                         wr_margin=round(margin, 1),      # model's own expected margin
                         weather_att=round(wx.get(team, 1.0), 3),
                         per80=round(per80, 2), exp_min=round(exp_min, 1),
                         starter_exp=round(starter_exp, 2), supersub_exp=round(supersub_exp, 2)))
    return pd.DataFrame(recs)


# ── 3. MILP squad optimiser ─────────────────────────────────────────────────
def optimise(df):
    n = len(df)
    # decision vars: x (in squad), s (is super sub), c (is captain)  → 3n binaries
    starter = df.starter_exp.values
    sub = df.supersub_exp.values
    # objective (maximise): x·starter (as starter) + s·(sub-starter) + c·starter
    #   a squad player scores starter_exp unless super sub (then sub); captain adds +1x
    obj = np.concatenate([starter, sub - starter, starter])
    c = -obj                                             # milp minimises

    cons = []
    Z = lambda: np.zeros(n)
    # Σx = 16
    cons.append(LinearConstraint(np.concatenate([np.ones(n), Z(), Z()]), SQUAD, SQUAD))
    # Σs = 1 ; Σc = 1
    cons.append(LinearConstraint(np.concatenate([Z(), np.ones(n), Z()]), 1, 1))
    cons.append(LinearConstraint(np.concatenate([Z(), Z(), np.ones(n)]), 1, 1))
    # s_i ≤ x_i ; c_i ≤ x_i − s_i  (captain must be a starter)
    I = np.eye(n)
    cons.append(LinearConstraint(np.hstack([-I, I, np.zeros((n, n))]), -np.inf, 0))
    cons.append(LinearConstraint(np.hstack([-I, I, I]), -np.inf, 0))
    # budget
    cons.append(LinearConstraint(np.concatenate([df.value.values, Z(), Z()]), -np.inf, BUDGET))
    # per-nation ≤ 3
    for t in df.team.unique():
        m = (df.team == t).values.astype(float)
        cons.append(LinearConstraint(np.concatenate([m, Z(), Z()]), -np.inf, MAX_NATION))
    # per-hemisphere ≤ 10
    for h in (1, 2):
        m = (df.hemi == h).values.astype(float)
        cons.append(LinearConstraint(np.concatenate([m, Z(), Z()]), -np.inf, MAX_HEMI))
    # starters per position == required  (starter = x − s)
    for pos, req in REQUIRED.items():
        m = (df.pos == pos).values.astype(float)
        cons.append(LinearConstraint(np.concatenate([m, -m, Z()]), req, req))
    # captain must be a confirmed starter (status P); super sub must be a bench player (status B)
    notP = (df.status != "P").values.astype(float)
    cons.append(LinearConstraint(np.concatenate([Z(), Z(), notP]), 0, 0))
    notB = (df.status != "B").values.astype(float)
    cons.append(LinearConstraint(np.concatenate([Z(), notB, Z()]), 0, 0))

    res = milp(c=c, constraints=cons, integrality=np.ones(3 * n),
               bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError("MILP failed: " + res.message)
    x = res.x[:n] > 0.5; s = res.x[n:2 * n] > 0.5; cap = res.x[2 * n:] > 0.5
    df = df.copy()
    df["in_squad"], df["is_sub"], df["is_capt"] = x, s, cap
    return df[df.in_squad].copy(), obj, res


def write_markdown(squad, tot_cost, tot_proj, topcaps, path, exclude_teams):
    """xPts is the player's own expected match points at 1x, so every row is directly
    comparable. The captain 2x and super-sub 3x multipliers are shown in the Role cell
    (base -> effective); the headline `Projected` total already applies them."""
    L = [f"# Nations Championship Fantasy — GW{CUR_GD} optimal squad", ""]
    if exclude_teams:
        L.append(f"> Excludes players from already-started matches: **{', '.join(exclude_teams)}**.")
        L.append("")
    L += [f"**Budget:** £{tot_cost:.1f}M / {BUDGET:.0f}M  ·  "
          f"**North {int((squad.hemi==1).sum())} / South {int((squad.hemi==2).sum())}**  ·  "
          f"**{squad.team.nunique()} nations**  ·  "
          f"**Projected: {tot_proj:.1f} pts** (captain 2x, super sub 3x off bench)", "",
          "| Pos | Player | Nation | £ | xMin | xPts | Role |",
          "|---|---|---|---:|---:|---:|---|"]
    for _, r in squad.iterrows():
        nm = f"**{r['name']}**" if r.role != "Starter" else r['name']
        if r.role == "CAPTAIN":
            role = f"**CAPTAIN** 2x → {r.proj:.1f}"
        elif r.role == "SUPER SUB":
            role = f"**SUPER SUB** 3x off bench → {r.proj:.1f}"
        else:
            role = ""
        L.append(f"| {r.pos} | {nm} | {r.team} | {r.value:.1f} | "
                 f"{r.exp_min:.0f} | {r.starter_exp:.1f} | {role} |")

    capt = squad.loc[squad.is_capt, "name"]
    if len(capt):
        L += ["", f"**Captain:** {capt.iloc[0]}"]
        top_name = topcaps.iloc[0]["name"] if len(topcaps) else None
        if top_name and top_name != capt.iloc[0]:
            L.append(f"> Note: not the highest xPts starter ({top_name}) — the armband was set "
                     f"by a dedicated captain head, which ranks captaincy rather than raw points.")
    L += ["", "**Captaincy options (highest xPts among confirmed starters, 1x):**"]
    for _, r in topcaps.iterrows():
        L.append(f"- {r['name']} ({r.team}, {r.pos}) — {r.starter_exp:.1f} xPts  ·  £{r.value}")
    L += ["", "*xPts = expected match points at 1x. xMin = expected minutes.*"]
    Path(path).write_text("\n".join(L) + "\n")


def render(df, squad, tag, exclude, title="empirical projection"):
    squad = squad.copy()
    squad["role"] = np.where(squad.is_sub, "SUPER SUB",
                     np.where(squad.is_capt, "CAPTAIN", "Starter"))
    squad["proj"] = np.where(squad.is_sub, squad.supersub_exp,
                    np.where(squad.is_capt, 2 * squad.starter_exp, squad.starter_exp))
    order = {"Prop": 0, "Hooker": 1, "Lock": 2, "Loose Forward": 3, "Scrum Half": 4,
             "Fly Half": 5, "Centre": 6, "Back Three": 7}
    squad["po"] = squad.pos.map(order)
    squad = squad.sort_values(["is_sub", "po", "starter_exp"], ascending=[True, True, False])

    print(f"\n{'='*78}\nNCR OPTIMAL GW{CUR_GD} SQUAD  ({title})\n{'='*78}")
    print(f"{'Pos':<13}{'Player':<24}{'Nation':<13}{'£':>5}{'xMin':>6}{'Proj':>7}  Role")
    print("-" * 78)
    for _, r in squad.iterrows():
        mark = "  ← " + r.role if r.role != "Starter" else ""
        print(f"{r.pos:<13}{r['name'][:23]:<24}{r.team:<13}{r.value:>5.1f}{r.exp_min:>6.0f}{r.proj:>7.1f}{mark}")
    tot_cost, tot_proj = squad.value.sum(), squad.proj.sum()
    print("-" * 78)
    print(f"squad cost: £{tot_cost:.1f}M / {BUDGET:.0f}M   projected: {tot_proj:.1f} pts   "
          f"N{int((squad.hemi==1).sum())}/S{int((squad.hemi==2).sum())}  "
          f"nations {squad.team.nunique()}")
    topcaps = df[df.status == "P"].sort_values("starter_exp", ascending=False).head(6)
    print("Top captaincy options:", ", ".join(f"{r['name']} {r.starter_exp:.1f}" for _, r in topcaps.iterrows()))
    squad.to_csv(NCR / f"ncr_gw{CUR_GD}_squad{tag}.csv", index=False)
    md = NCR / f"ncr_gw{CUR_GD}_squad{tag}.md"
    write_markdown(squad, tot_cost, tot_proj, topcaps, md, exclude)
    print(f"saved: ncr_gw{CUR_GD}_squad{tag}.csv, {md.name}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[], help="team names to drop (matches started)")
    ap.add_argument("--tag", default="", help="output filename suffix")
    args = ap.parse_args()
    exclude = args.exclude
    tag = args.tag or ("_ex_" + "_".join(t.replace(" ", "") for t in exclude) if exclude else "")

    df = build_projection(exclude_teams=exclude)
    df.to_csv(NCR / f"ncr_gw{CUR_GD}_projections{tag}.csv", index=False)
    squad, _, _ = optimise(df)
    render(df, squad, tag, exclude)


if __name__ == "__main__":
    main()
