import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).parents[1]
REPO = Path('/workspace/fluid_control')
PENDING = ROOT/'PPO_PENDING.json'
if not PENDING.exists():
    PENDING = ROOT/'docs/P064_G_SYMMETRY_CANONICAL_PPO_APPROVAL_20261007.json'


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m)
    return m


def functions(path):
    tree = ast.parse(path.read_text().replace('P064_G_SYMMETRY_CANONICAL', 'P064_B_SYMMETRY_CANONICAL'))
    return {x.name: ast.dump(x, include_attributes=False) for x in tree.body if isinstance(x, ast.FunctionDef)}


def test_training_execute_and_protocol_exact_e082():
    old = REPO/'artifacts/p064_symmetry_canonical_policy_source_20261007_immutable/scripts/train_p064_symmetry_canonical_b_32768_ppo.py'
    new = ROOT/'scripts/train_p064_g_symmetry_canonical_32768_ppo.py'
    assert functions(old)['execute'] == functions(new)['execute']
    a=load('old_train',old);b=load('g_train',new)
    assert a.PROTOCOL == b.PROTOCOL
    spec=json.loads(PENDING.read_text())
    assert spec['execution_authorized'] is (PENDING.name != 'PPO_PENDING.json')
    assert spec['protocol'] == a.PROTOCOL


def test_g_real_terminal_proofs_and_old_kind_rejection():
    m=load('g_proofs',ROOT/'scripts/train_p064_g_symmetry_canonical_32768_ppo.py')
    spec=json.loads(PENDING.read_text())
    clone=copy.deepcopy(spec);clone.update(status=m.STATUS,execution_authorized=True,reviewed_by_lead=True)
    m.validate_spec(clone)
    wrong=copy.deepcopy(clone);wrong['candidate_arm']='B'
    with pytest.raises(ValueError):m.validate_spec(wrong)
    wrong=copy.deepcopy(clone);wrong['candidate_manifest_kind']='FC_P064_ARM_B_CONTROLLED_AERO_FORCE_FNO'
    with pytest.raises(ValueError):m.validate_spec(wrong)


def test_cfd_execute_numerics_exact_e085():
    old=REPO/'artifacts/p064_symmetry_canonical_b01_cfd_source_20261007_immutable/run_p064_symmetry_canonical_b_ppo_b01_long_cfd.py'
    new=ROOT/'scripts/run_p064_g_symmetry_canonical_b01_cfd.py'
    a=functions(old);b=functions(new)
    for name in a:
        if name!='validate_spec':assert a[name]==b[name],name


def test_supervisor_only_candidate_identity_delta():
    old=REPO/'artifacts/p064_symmetry_canonical_policy_source_20261007_immutable/scripts/supervise_p064_symmetry_canonical_ppo.py'
    new=ROOT/'scripts/supervise_p064_g_symmetry_canonical_ppo.py'
    a=functions(old);b=functions(new)
    for name in a:
        if name!='candidate_binding':assert a[name]==b[name],name
    m=load('g_supervisor',new)
    assert m.candidate_binding(dict(candidate_arm='G',candidate_manifest_kind='FC_P064_AR5_RESET_K1_FRESH_FORCE_FNO',candidate_manifest_sha256='a'*64))[0]=='G'
    with pytest.raises(ValueError):m.candidate_binding(dict(candidate_arm='B'))


def test_runtime_mapping_single_new_profile_only():
    old=REPO/'artifacts/p064_candidate_policy_source_20261006_immutable/runtime_src/fluid_control/dual_control_contract.py'
    new=ROOT/'runtime_src/fluid_control/dual_control_contract.py'
    if not new.exists():
        new=ROOT/'src/fluid_control/dual_control_contract_g_canonical.py'
    text=new.read_text().replace('        "FC_P064_AR5_RESET_K1_FRESH_FORCE_FNO": ("p026_k1", 1),\n','')
    assert text.rstrip()==old.read_text().rstrip()
