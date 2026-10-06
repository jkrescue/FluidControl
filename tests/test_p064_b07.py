import ast
import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
SCRIPT = HERE / 'run_p064_candidate_projected_32768_ppo_b07_long_cfd.py'
if not SCRIPT.exists():
    SCRIPT = HERE.parent / 'scripts' / SCRIPT.name
BASE = Path(os.environ.get('P064_B01_BASE', str(SCRIPT.with_name('run_p064_candidate_projected_32768_ppo_b01_long_cfd.py'))))
spec = importlib.util.spec_from_file_location('p064_b07', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_exact_phase_only_delta():
    source = BASE.read_text()
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == '4b8fa43f8ac020521ae8cde1047b6ec2f80d35ec0512bc7606d1050835010619'
    changes = [('B01','B07'), ('b01','b07'),
        ('START, END, PRIMARY_START = 130., 210., 150.', 'START, END, PRIMARY_START = 110., 190., 130.'),
        ("row['phase_bin'] == 1", "row['phase_bin'] == 7"),
        ("phase['split'] == 'validation'", "phase['split'] == 'frozen_test'"),
        ('.6991961542646722', '5.587043363374366'),
        ('ac412e9e3de151253dd3006a70ab195ef8f49f9c81edeeef28086b808fe9d230', '4573d15859d34acaa27dcf16959730d3f6574c95ce93c87199a24a0373c32963'),
        ('1ddc27110bacd814e31b3425e3550927ba7fee3f57112e62a4cb2a18288aa52c', '606b05613f479d9ad1cdd68b133c7f373ee0a51fffc171d6816b372f11569751'),
        ('matched_start_acquisition_validation_b07_zero', 'matched_start_acquisition_frozen_test_b07_zero'),
        ('0b59387acf6365700a4a1b2e5b00f63f3c71abb6efcfcd10850ccd5ba4df1f7c', '7df9372892131839fa9a964d70919d1b1513638a1fc4499a062b3d601454597d')]
    for old,new in changes:
        assert old in source
        source = source.replace(old,new)
    assert source.rstrip() == SCRIPT.read_text().rstrip()  # Ignore only EOF whitespace.


def test_fixed_six_windows_and_original_primary():
    assert (m.START,m.END,m.PRIMARY_START) == (110.,190.,130.)
    assert m.declared_windows() == [
        ('early_12p4',110.,122.4,False), ('early_first_6p2',110.,116.2,False),
        ('early_trailing_6p2',116.2,122.4,False), ('primary_final_60',130.,190.,False),
        ('historical_inclusive_final_60',130.,190.,True), ('full_80',110.,190.,False)]
    data = np.zeros((16000,3))
    data[:,0] = 110. + .005*np.arange(1,16001)
    assert [len(m.fixed_window(data,a,b,c)) for _,a,b,c in m.declared_windows()] == [2480,1240,1240,12000,12001,16000]


def test_restart_copy_is110_only(tmp_path):
    source=tmp_path/'source'; output=tmp_path/'output'
    output.mkdir()
    for part in ('110','130','constant','system'):
        (source/part).mkdir(parents=True)
        (source/part/'fixture').write_text(part)
    cases=m.build_pair_at(source,output)
    for case in cases.values():
        assert (case/'110/fixture').read_text() == '110'
        assert not (case/'130').exists()


def test_projection_and_single_filter_loop_unchanged():
    def functions(path):
        return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    a,b=functions(BASE),functions(SCRIPT)
    for name in ('predict','reflect_physical69','projected_request','summarize','validate_training','training_bindings'):
        assert a[name] == b[name]
    assert SCRIPT.read_text().count("action = transport.apply_action_rate_limit(") == 1


def test_duplicate_raw_grid_rejected():
    data=np.zeros((12000,3));data[:,0]=130+.005*np.arange(1,12001)
    data[2,0]=data[1,0]
    with pytest.raises(ValueError,match='force grid'):
        m.fixed_window(data,130,190)


def test_pending_is_not_execution_authority():
    with pytest.raises(ValueError,match='explicit approval'):
        m.validate_spec({'status':'P064_B07_PREPARATION_ONLY','execution_authorized':False})
