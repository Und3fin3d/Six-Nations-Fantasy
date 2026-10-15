from __future__ import annotations

import re
import unicodedata

import pandas as pd

OFFICIAL = {
    2023: "data/official/2023/Six_Nations_Data_2023.xlsx", 2025: "data/official/2025/sixnationsmatch.xlsx", 2026: "data/official/2026/sixnationmatch.xlsx",
}

STAT_MAP = {
    "Min": "minutes", "T": "tries", "As": "try_assists", "C": "conversion_goals", "Pen": "penalty_goals", "MC": "metres", "DB": "defenders_beaten",
    "Ta": "tackles", "CPen": "penalties_conceded", "DG": "drop_goals_converted", "OF": "offload", "YC": "yellow_cards", "RC": "red_cards",
    "BS": "tackle_turnover",
}
API_BLIND = ["50-22", "KR", "LS", "POTM", "SW"]


def norm_key(name: str) -> str:
    parts = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().strip().split() if isinstance(name, str) else []
    return f"{parts[0][0].lower()}|{re.sub(r'[^a-z]', '', (''.join(parts[1:]) if len(parts) > 1 else parts[0]).lower())}" if parts else ""


def parse_official(path: str, season: int) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    out = []
    for r_idx, sheet in enumerate(xl.sheet_names, start=1):
        raw = pd.read_excel(xl, sheet_name=sheet, header=None)
        labels = team = None
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
            out.append({"season": season, "round": r_idx, "team": team, "name": cell0.strip(), "key": norm_key(cell0), **dict(zip(labels[1:], vals[1:]))})
    df = pd.DataFrame(out)
    for lab in set(STAT_MAP) | set(API_BLIND) | {"Pts"}:
        if lab in df.columns:
            df[lab] = pd.to_numeric(df[lab], errors="coerce").fillna(0)
    return df
