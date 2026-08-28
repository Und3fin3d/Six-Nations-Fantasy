#!/usr/bin/env bash
#
# gw_update.sh — one command to take either fantasy game from "a gameweek is coming"
#                to "a team is picked".
#
#   ./gw_update.sh ncr                    # Nations Championship: snapshot → ingest → score → WR → team
#   ./gw_update.sh 6n                     # Six Nations: snapshot → ingest → WR → predictions
#   ./gw_update.sh ncr --exclude "New Zealand" "Italy"
#   ./gw_update.sh ncr --dry-run          # print the plan, touch nothing
#   ./gw_update.sh 6n  --rebuild-features # ALSO regenerate the model stores (see warning)
#
# Handles the two footguns that keep biting us:
#   * the data interpreter must contain pandas
#   * the pinned model venv lives in /tmp and gets wiped → rebuilt automatically
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY_DATA=${PY_DATA:-"$HOME/.venvs/main/bin/python"}
PY_MODEL=/tmp/6n-model-pinned/bin/python
VENV_BASE=~/.local/bin/python3.11

GAME=""; DRY=0; NO_FETCH=0; PREDICT_ONLY=0; REBUILD_FEATURES=0; EXCLUDE=()

die()  { printf '\n\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1m▸ %s\033[0m\n' "$*"; }
note() { printf '  \033[2m%s\033[0m\n' "$*"; }
run()  { if (( DRY )); then printf '  \033[2m$ %s\033[0m\n' "$*"; else "$@"; fi; }

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# ── args ────────────────────────────────────────────────────────────────────
[[ $# -gt 0 ]] || usage 1
case "$1" in ncr|6n) GAME="$1"; shift ;; -h|--help) usage ;; *) die "unknown game '$1' (want ncr|6n)" ;; esac
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)          DRY=1; shift ;;
    --no-fetch)         NO_FETCH=1; shift ;;          # cache-only; spends no API quota
    --predict-only)     PREDICT_ONLY=1; shift ;;      # skip all data refresh
    --rebuild-features) REBUILD_FEATURES=1; shift ;;
    --exclude)          shift; while [[ $# -gt 0 && "$1" != --* ]]; do EXCLUDE+=("$1"); shift; done ;;
    -h|--help)          usage ;;
    *)                  die "unknown flag '$1'" ;;
  esac
done

# ── preflight ───────────────────────────────────────────────────────────────
step "preflight"
[[ -x "$PY_DATA" ]] || die "$PY_DATA missing"
"$PY_DATA" -c 'import pandas' 2>/dev/null || die "$PY_DATA has no pandas"
note "data interpreter: $PY_DATA"

if [[ ! -x "$PY_MODEL" ]]; then
  note "pinned venv missing (/tmp gets wiped) — rebuilding"
  if (( ! DRY )); then
    [[ -x "$VENV_BASE" ]] || die "need python3.11 at $VENV_BASE to rebuild the venv"
    "$VENV_BASE" -m venv /tmp/6n-model-pinned
    /tmp/6n-model-pinned/bin/pip install -q -r requirements-model.txt
  fi
fi
note "model interpreter: $PY_MODEL"

quota() {
  "$PY_DATA" - <<'PY' 2>/dev/null || echo "unknown"
import sys
sys.path.insert(0, ".")
from rugby_api import RugbyAPI
api = RugbyAPI(verbose=False)
api.get("/competitions", refresh=True)
print(api.remaining if api.remaining is not None else "?")
PY
}

if (( ! PREDICT_ONLY && ! NO_FETCH && ! DRY )); then
  Q=$(quota); note "API quota remaining: $Q  (costs 1 call to check)"
  if [[ "$Q" =~ ^[0-9]+$ ]] && (( Q < 50 )); then
    note "quota below 50 — forcing --no-fetch (cache only)"; NO_FETCH=1
  fi
fi

# ── the two pipelines ───────────────────────────────────────────────────────
ncr_pipeline() {
  if (( ! PREDICT_ONLY )); then
    step "1/6  snapshot fantasy feed  (free — prices, injuries, TEAM SHEETS)"
    run "$PY_DATA" ncr_snapshot.py

    step "2/6  ingest results → history tables"
    if (( NO_FETCH )); then run "$PY_DATA" ncr_ingest.py --rebuild
    else                    run "$PY_DATA" ncr_ingest.py --floor 50; fi

    step "3/6  score the completed gameweek's saved model teams"
    local score_gw
    score_gw=$( (( DRY )) && echo "<previous-gameday>" || "$PY_DATA" - <<'PY'
from pathlib import Path
import pandas as pd

fx = pd.read_csv("data/ncr/ncr_fixtures.csv")
cur = fx[pd.to_numeric(fx["iscurrent"], errors="coerce").eq(1)]
if len(cur):
    gw = int(cur["gameday"].iloc[0]) - 1
    date = pd.to_datetime(fx[fx["gameday"].eq(gw)]["game_date"]).dt.strftime("%Y-%m-%d")
    hist = pd.read_csv("data/ncr/ncr_player_match.csv")
    official = Path(f"data/ncr/feeds/players_gw{gw}.json")
    potm = pd.read_csv("data/ncr/ncr_potm.csv") if Path("data/ncr/ncr_potm.csv").exists() else pd.DataFrame()
    squads = [
        Path(f"data/ncr/ncr_gw{gw}_squad.csv"),
        Path(f"data/ncr/ncr_gw{gw}_squad_champion.csv"),
        Path(f"data/ncr/ncr_gw{gw}_squad_blend.csv"),
    ]
    reconstruction_ready = (
        len(date)
        and hist[hist["date"].astype(str).eq(date.iloc[0])]["fixture_id"].nunique() == 6
        and len(potm)
        and potm[potm["date"].astype(str).eq(date.iloc[0])]["fixture_id"].nunique() == 6
    )
    if gw > 0 and all(p.exists() for p in squads) and (official.exists() or reconstruction_ready):
        print(gw)
PY
)
    if [[ -n "$score_gw" ]]; then
      run "$PY_MODEL" -m model.ncr_score_gw --gw "$score_gw" --update-markdown
    else
      note "results or verified POTM data are not complete yet — skipping"
    fi

    step "4/6  refresh World Rugby ratings  (free, merges)"
    local d
    d=$( (( DRY )) && echo "<current-gameday>" || "$PY_DATA" - <<'PY'
import pandas as pd
fx = pd.read_csv("data/ncr/ncr_fixtures.csv")
cur = fx[fx.get("iscurrent", 0) == 1]
print(pd.to_datetime(cur["game_date"]).min().date() if len(cur) else "")
PY
)
    [[ -n "$d" ]] && run "$PY_DATA" build_wr.py --dates "$d" || note "no current gameday — skipping"
  fi

  step "5/6  pick the team"
  if (( ${#EXCLUDE[@]} )); then run "$PY_MODEL" -m model.ncr_project --exclude "${EXCLUDE[@]}"
  else                          run "$PY_MODEL" -m model.ncr_project; fi

  step "6/6  freeze configured unified-v3 shadow prediction"
  local shadow_config="data/unified/v3/shadow_active.json"
  if (( DRY )); then
    note "would inspect $shadow_config and freeze the current GW once"
  elif [[ ! -f "$shadow_config" ]]; then
    note "no active unified-v3 shadow model — skipping"
  else
    local shadow_spec shadow_engine shadow_model shadow_gw
    # shadow_active.json holds one spec (legacy object) or a list of specs so
    # several engines (e.g. v3 baseline + gbdt_v4) can shadow the same GW.
    shadow_spec=$("$PY_MODEL" - <<'PY'
import json
import pandas as pd

config = json.load(open("data/unified/v3/shadow_active.json"))
specs = config if isinstance(config, list) else [config]
fixtures = pd.read_csv("data/ncr/ncr_fixtures.csv")
current = fixtures[pd.to_numeric(fixtures["iscurrent"], errors="coerce").eq(1)]
if len(current):
    gw = int(current["gameday"].iloc[0])
    for spec in specs:
        if spec.get("target_gw") == gw:
            print(f"{spec['engine']}\t{spec['model']}\t{gw}")
PY
)
    if [[ -z "$shadow_spec" ]]; then
      note "active shadow is not configured for the current GW — skipping"
    else
      while IFS=$'\t' read -r shadow_engine shadow_model shadow_gw; do
        [[ -z "$shadow_engine" ]] && continue
        if [[ -f "data/unified/v3/shadow/ncr_gw${shadow_gw}_${shadow_engine}.csv" ]]; then
          note "immutable GW${shadow_gw} ${shadow_engine} shadow already exists — keeping it"
        else
          run "$PY_MODEL" -m model.unified.v3.cli shadow \
            --gw "$shadow_gw" --engine "$shadow_engine" --model "$shadow_model"
        fi
      done <<< "$shadow_spec"
    fi
  fi
}

# Six Nations + the club form-window comps. VERIFIED against the comp_id distribution
# of data/api_player_match.csv — 1218=Premiership, 1230=TOP 14 (easy to transpose).
SIXN_COMPS=(1266 1218 1230 1236 1464 1470)

sixn_pipeline() {
  if (( ! PREDICT_ONLY )); then
    step "1/4  snapshot fantasy catalogue  (free — prices, ownership)"
    run "$PY_DATA" snapshot_fantasy_market.py --game m6n --allow-unavailable

    step "2/4  ingest matches → api_player_match / api_team_match"
    note "this REWRITES both stores from the cache (club comps are Oct→Feb windowed)"
    if (( NO_FETCH )); then note "--no-fetch: re-parsing cache only, no new /match calls"; fi
    run "$PY_DATA" ingest_6n.py --comps "${SIXN_COMPS[@]}"
    run "$PY_DATA" ingest_6n.py --comps "${SIXN_COMPS[@]}" --team

    step "3/4  refresh World Rugby ratings  (free, merges)"
    run "$PY_DATA" build_wr.py

    if (( REBUILD_FEATURES )); then
      step "3b/4  rebuild model stores  ⚠  model/data.py asserts exactly 2760 rows"
      note "if the row count moves, load() fails — that assertion is the tripwire, not a bug"
      run "$PY_DATA" build_crosswalk.py
      run "$PY_DATA" build_team_form.py
      run "$PY_DATA" build_features.py
      run "$PY_DATA" build_targets.py
    elif (( ! DRY )) && [[ data/model_player_match.csv -ot data/api_player_match.csv ]]; then
      note "⚠  model_player_match.csv is now OLDER than api_player_match.csv —"
      note "   the feature store is stale. Re-run with --rebuild-features."
    else
      note "skipping feature/target rebuild (pass --rebuild-features to include)"
    fi
  fi

  step "4/4  predictions  (backtest + promoted config)"
  run "$PY_MODEL" -m model.run
}

case "$GAME" in
  ncr) ncr_pipeline ;;
  6n)  sixn_pipeline ;;
esac

# ── summary ─────────────────────────────────────────────────────────────────
step "done"
if (( DRY )); then
  note "dry run — nothing was executed"
elif [[ "$GAME" == ncr ]]; then
  ls -t data/ncr/ncr_gw*_squad*.md 2>/dev/null | head -1 | sed 's/^/  team: /'
  note "if ncr_snapshot said team sheets are CARRYOVER, this team is PROVISIONAL —"
  note "re-run once the real sheets post (~48h before lock)."
else
  ls -t data/model_predictions_*.csv 2>/dev/null | head -2 | sed 's/^/  preds: /'
fi
