import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("oneframe",Path(__file__).resolve().parents[1]/"scripts"/"verify_online_current_frame_cpu.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_exact_and_nonexact_report_without_threshold():
    assert m.differences([1,2],[1,2])["exact_equal"]
    r=m.differences([1.,3.],[1.,2.])
    assert not r["exact_equal"] and r["max_abs"]==1 and r["rmse"]==np.sqrt(.5)

@pytest.mark.parametrize("a,b",[([1],[1,2]),([float("nan")],[1]),([],[])])
def test_invalid_comparison(a,b):
    with pytest.raises(ValueError): m.differences(a,b)

def test_hidden_gpu_required_before_any_data_read(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    with pytest.raises(ValueError,match="CUDA must be hidden"): m.execute(tmp_path,tmp_path/"out")
    assert not (tmp_path/"out").exists()
