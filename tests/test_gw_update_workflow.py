from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from model_env_preflight import check


ROOT = Path(__file__).resolve().parents[1]


def test_model_environment_preflight_requires_every_exact_pin(tmp_path: Path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("scikit-learn==1.3.1\npandas==2.1.0\n")

    assert check(
        requirements,
        python_version=(3, 11),
        installed_version={"scikit-learn": "1.3.1", "pandas": "2.1.0"}.__getitem__,
    ) == []
    assert check(
        requirements,
        python_version=(3, 12),
        installed_version={"scikit-learn": "1.8.0", "pandas": "2.1.0"}.__getitem__,
    ) == [
        "Python 3.11 is required; found 3.12",
        "scikit-learn==1.3.1 is required; found 1.8.0",
    ]


def test_ncr_shadow_config_uses_saved_p3_for_all_prospective_rounds():
    config = json.loads((ROOT / "data/unified/v3/shadow_active.json").read_text())

    assert config == {
        "engine": "p3_event_50",
        "model": (
            "data/unified/raw_benchmark/v1/models/p3_event_50/"
            "nations_championship_2026.pkl"
        ),
        "target_gws": [4, 5, 6, 7],
    }
    assert (ROOT / config["model"]).is_file()


def test_ncr_dry_run_keeps_predictions_non_mutating_and_reports_p3_shadow():
    environment = os.environ.copy()
    environment.update({
        "PY_DATA": sys.executable,
        "PY_MODEL": sys.executable,
        "VENV_BASE": sys.executable,
    })
    result = subprocess.run(
        ["bash", "gw_update.sh", "ncr", "--dry-run"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert f"model interpreter: {sys.executable}" in result.stdout
    assert "freeze the current GW once" in result.stdout
    assert "dry run — nothing was executed" in result.stdout
