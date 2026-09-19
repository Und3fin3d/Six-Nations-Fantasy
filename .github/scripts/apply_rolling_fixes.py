"""One-time branch bootstrap; refuses to edit an unexpected source revision."""
from pathlib import Path


def replace(path, old, new):
    p = Path(path)
    text = p.read_text()
    if new in text:
        return
    if text.count(old) != 1:
        raise AssertionError(f"{path}: expected one source anchor, got {text.count(old)}")
    p.write_text(text.replace(old, new, 1))

p = "model/ncr_project.py"
replace(p, "from scipy.optimize import milp, LinearConstraint, Bounds", "from scipy.optimize import milp, LinearConstraint, Bounds\nfrom model.pit import history_before, day_cutoff")
replace(p, "    prediction_asof = pd.Timestamp(asof) if asof is not None else ASOF\n    hist = hist.copy()", "    prediction_asof = day_cutoff(asof if asof is not None else ASOF).tz_localize(None)\n    hist = history_before(hist, prediction_asof)")
replace(p, "        club = club.copy()\n        club[\"date\"]", "        club = history_before(club, prediction_asof)\n        club[\"date\"]")
replace(p, "    asof=None, gameday=None,", "    asof=None, gameday=None, use_weather=True,")
replace(p, "    prediction_asof = pd.Timestamp(asof) if asof is not None else ASOF\n    pool =", "    prediction_asof = day_cutoff(asof if asof is not None else ASOF).tz_localize(None)\n    pool =")
replace(p, "    hist = pd.read_csv(NCR / \"ncr_player_match.csv\")", "    hist = history_before(pd.read_csv(NCR / \"ncr_player_match.csv\"), prediction_asof)")
replace(p, "    if point_in_time_wr.empty:\n        point_in_time_wr = wr\n", "    # No past ranking means a neutral fallback, never a future snapshot.\n")
replace(p, "    if wx_path.exists():", "    if use_weather and wx_path.exists():")
replace(p, "        club = pd.read_csv(club_path)", "        club = history_before(pd.read_csv(club_path), prediction_asof)")
replace(p, "            rp = RP.rp_rates()", "            rp = RP.rp_rates(asof=prediction_asof)")

p = "model/rp_rates.py"
replace(p, "import pandas as pd", "import pandas as pd\nfrom model.pit import completed_seasons, day_cutoff")
replace(p, "def rp_rates(asof_year=2026, min_minutes=120):", "def rp_rates(asof_year=None, min_minutes=120, *, asof=None):")
replace(p, '    cs["w"] = 0.5 **', '''    if asof is None:
        asof = f"{int(asof_year)}-01-01" if asof_year is not None else pd.Timestamp.now(tz="UTC")
    cutoff = day_cutoff(asof)
    cs = cs[completed_seasons(cs["season"], cutoff)].copy()
    asof_year = cutoff.year
    cs["w"] = 0.5 **''')
replace(p, "    out = pd.DataFrame(rows)", "    out = pd.DataFrame(rows, columns=[\"slug\", \"att\", \"dfn\", \"dsc\", \"rp_min\"])")

p = "model/empirical_unified.py"
replace(p, "import pandas as pd", "import pandas as pd\nfrom model.pit import completed_seasons")
replace(p, '''    end_year = cs["season"].map(_season_end_year)
    concluded = pd.to_datetime(end_year.astype(str) + "-06-30")
    cs = cs[concluded < asof.tz_localize(None) if asof.tzinfo else concluded < asof]''', '''    cs = cs[completed_seasons(cs["season"], asof)].copy()''')
replace(p, "RugbyPass seasons only if concluded (June 30 of the\nseason's end year) before the round.", "RugbyPass full-season totals use conservative completed-season availability\n(split seasons after June, calendar-year seasons after December).")

p = "model/unified/features.py"
replace(p, "from .schema import EVENTS", "from .schema import EVENTS\nfrom model.pit import team_margin_history")
replace(p, '''    df["team_recent_margin"] = df.groupby("team", sort=False)["team_margin"].transform(
        lambda x: x.shift(1).ewm(span=8, min_periods=1).mean())''', '''    df["team_recent_margin"] = team_margin_history(df, prior_only=True)''')
p = "model/unified/raw_benchmark/features.py"
replace(p, "from ..features import build_pit_features", "from ..features import build_pit_features\nfrom model.pit import team_margin_history")
replace(p, '''    margin_state = team_margin.groupby(ordered["team"], sort=False).transform(
        lambda values: values.ewm(span=8, min_periods=1).mean()
    )''', '''    margin_state = team_margin_history(ordered, prior_only=False)''')

p = "model/unified/raw_benchmark/empirical.py"
replace(p, "from model.rp_rates import _norm, _season_end_year", "from model.rp_rates import _norm, _season_end_year\nfrom model.pit import completed_seasons")
replace(p, '''        end_year = rp["season"].map(_season_end_year)
        concluded = pd.to_datetime(end_year.astype(str) + "-06-30", errors="coerce")
        rp = rp[concluded.lt(asof)].copy()''', '''        rp = rp[completed_seasons(rp["season"], asof)].copy()''')
replace(p, '''                rate = float((group.loc[valid, "_weight"] * values[valid]).sum() / weighted_minutes * 80)
                by_name[(_norm(str(slug).replace("-", " ")), event)] = (rate, weighted_minutes)''', '''                exposure = float((group.loc[valid, "_weight"] * minutes[valid]).sum())
                if exposure <= 0:
                    continue
                rate = float((group.loc[valid, "_weight"] * values[valid]).sum() / exposure * 80)
                by_name[(_norm(str(slug).replace("-", " ")), event)] = (rate, exposure)''')
replace(p, '    active_events: set[str] = field(default_factory=set)', '''    active_events: set[str] = field(default_factory=set)
    minutes_prior_matches: float = 0.0
    exposure_weighted_priors: bool = False''')
replace(p, '''            position = block.groupby("position")["_rate"].mean()
            overall = float(block["_rate"].mean())''', '''            position = block.groupby("position")["_rate"].mean()
            overall = float(block["_rate"].mean())
            if self.exposure_weighted_priors:
                total = block.groupby("position")[[event, "minutes"]].sum()
                position = total[event] / total["minutes"].clip(lower=1) * 80.0
                overall = float(block[event].sum() / max(block["minutes"].sum(), 1) * 80.0)''')
replace(p, '        minute_rows = train[minute_valid].copy()', '''        minute_rows = train[minute_valid].copy()
        if self.minutes_prior_matches > 0:
            minute_rows = minute_rows[minute_rows["competition_level"].eq("international")].copy()''')
replace(p, '        self.dispersion["minutes"] = max(', '''        if self.minutes_prior_matches > 0:
            for (player, started), group in minute_rows.groupby(["player_id", "started"]):
                key = (str(player), bool(started))
                position = str(group.iloc[-1]["position"])
                prior = self.minutes_by_position.get((position, bool(started)), 70.0 if started else 20.0)
                weight = float(group["_weight"].sum())
                self.minutes_by_player[key] = (weight * self.minutes_by_player[key] + self.minutes_prior_matches * prior) / (weight + self.minutes_prior_matches)
        self.dispersion["minutes"] = max(''')

p = "model/unified/ncr_gw_eval.py"
replace(p, '''    args = parser.parse_args()
    if args.teams_only:''', '''    parser.add_argument("--history", choices=("rolling", "frozen"), default="rolling",
                        help="rolling updates all models before each round; frozen reproduces the old tournament holdout")
    args = parser.parse_args()
    if args.history == "rolling":
        if args.teams_only:
            parser.error("--teams-only requires --history frozen; rolling comparisons rebuild all models")
        from .rolling_evaluation import run
        for gw in (1, 2, 3):
            run(f"ncr-2026-gw{gw}")
        return
    if args.teams_only:''')
print("Applied rolling-history fixes")
