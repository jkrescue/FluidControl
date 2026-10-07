import importlib.util
from pathlib import Path
import pytest

SOURCE=Path(__file__).with_name('audit_future_time_cfd.py')
if not SOURCE.exists():SOURCE=Path(__file__).resolve().parents[1]/'scripts/audit_future_time_cfd.py'
spec=importlib.util.spec_from_file_location('audit',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_exact_baseline_clock_and_health():
    rows=[dict(step=i,start_time=228+.1*(i-1),end_time=228+.1*i,solver_health={'baseline':dict(steps=20,solver_ended_cleanly=True,max_courant=.3,max_abs_global_continuity_per_step=1e-12)}) for i in range(1,201)]
    m.check_baseline_rows(rows)
    rows[9]['end_time']+=.005
    with pytest.raises(AssertionError):m.check_baseline_rows(rows)

def test_zero_tables_short_and_openfoam_written():
    for text in ['rearCylinder { omega table ((228 0)(228.1 0)); } frontBack', 'rearCylinder {omega table; omegaCoeffs { values 2 ((228 0)(228.1 -0)); }} frontBack']:
        m.zero_table(text,228,228.1)
    with pytest.raises(AssertionError):m.zero_table('rearCylinder {omega table ((228 0)(228.1 .1));} frontBack',228,228.1)

def test_terminal_guard_precedes_payload_and_no_historical_zero_requirement():
    text=Path(m.__file__).read_text()
    assert text.index('require successful terminal')<text.index("r=read(P/'result.json')")
    assert 'CFD_REFERENCE' not in text and 'oldzero' not in text
    assert "assert len(containers)==3" in text and "len(logs)==1600" in text
