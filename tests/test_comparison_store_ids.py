import json
import pytest

from model.unified import comparison_report as reporting


def test_report_preserves_text_identifiers_before_label_matching(tmp_path, monkeypatch):
    fixtures = [str(i) for i in range(15)]
    config = tmp_path / "config"
    config.mkdir()
    (config / "fixtures.json").write_text(json.dumps({"fixtures": [{"fixture_id": f} for f in fixtures]}))
    monkeypatch.setattr(reporting, "CONFIG_DIR", config)
    output = tmp_path / "results"
    names = {f"test_{f}" for f in fixtures}
    names |= {f"{c}_{y}_r{r}" for (c,y),n in reporting.SEASONS.items() for r in range(1,n+1)}
    for name in names:
        (output / "jobs" / name).mkdir(parents=True)
    (output / "inputs").mkdir()
    (output / "inputs/player_match.csv").write_text(
        "fixture_id,player_id,team,date,match_at\n001,002,A,2024-01-01,2024-01-01T12:00:00Z\n"
    )
    class Inspected(Exception):
        pass
    def inspect(store, competitions):
        assert store.fixture_id.tolist() == ["001"]
        assert store.player_id.tolist() == ["002"]
        assert store.team.tolist() == ["A"]
        raise Inspected
    monkeypatch.setattr(reporting, "official_slates", inspect)
    with pytest.raises(Inspected):
        reporting.report(output, tmp_path / "previous", tmp_path / "report", "p3_seedbag_native")
