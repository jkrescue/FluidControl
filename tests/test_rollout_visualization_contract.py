from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_tandem_fno.py"


def test_rollout_figure_declares_units_and_percentile_clipping():
    source = SCRIPT.read_text()
    assert '("u/U∞", "v/U∞", "p/(ρU∞²)")' in source
    assert "Shared truth/prediction colors: 1st–99th percentile" in source
    assert "absolute-error colors: 0–99th percentile" in source

