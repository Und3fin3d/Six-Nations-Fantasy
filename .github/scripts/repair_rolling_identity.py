"""Apply small rolling-evaluation fixes once; fail if source anchors change."""
from pathlib import Path


def replace(path, old, new):
    p = Path(path)
    text = p.read_text()
    if new in text:
        return
    if text.count(old) != 1:
        raise AssertionError(f"{path}: expected exactly one source anchor")
    p.write_text(text.replace(old, new, 1))


replace("model/ncr_project.py",
        "        recs.append(dict(id=pl.id, name=pl.full_name, team=team, hemi=pl.hemisphere,",
        "        recs.append(dict(id=pl.id, name=pl.full_name, team=team, hemi=pl.hemisphere,\n                         history_player_id=pid_map.get(pl.id, np.nan),")
replace("model/unified/v3/shadow.py",
        "        api_id = (\n            str(int(row.api_player_id)) if pd.notna(row.api_player_id)",
        "        # New projections carry the exact identity used by the empirical model.\n        # Missing history is a cold start, not permission to use a fuzzy namesake.\n        history_id = getattr(row, \"history_player_id\", row.api_player_id)\n        api_id = (\n            str(int(history_id)) if pd.notna(history_id)")
replace("model/unified/v3/shadow.py",
        '            "player_name": row.full_name if isinstance(row.full_name, str) else row.name,',
        '            "player_name": row.name,')
replace("model/unified/rolling_evaluation.py",
        'def run(round_id: str, output: Path = OUT) -> pd.DataFrame:\n',
        'def run(round_id: str, output: Path = OUT) -> pd.DataFrame:\n    output = output.resolve()\n')
print("Applied shared historical identity and output-path fixes")
