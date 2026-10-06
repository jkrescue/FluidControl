"""CPU-only identity and exact retained lifecycle regression; no child launches."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import pytest

HERE=Path(__file__).parent
SOURCE=HERE/'supervise_p064_candidate_ppo.py'
if not SOURCE.exists(): SOURCE=HERE.parent/'scripts/supervise_p064_candidate_ppo.py'
spec=importlib.util.spec_from_file_location('candidate_supervisor',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OLD=Path('/workspace/fluid_control/artifacts/exploratory_diverse_h5_32768_source_20261006_immutable/supervise_exploratory_diverse_h5_32768_ppo.py')

def binding(arm='A'):
    return dict(candidate_arm=arm,candidate_manifest_sha256='a'*64,
                candidate_manifest_kind=f'FC_P064_ARM_{arm}_CONTROLLED_AERO_FORCE_FNO')

def test_both_explicit_arms():
    for arm in ('A','B'): assert m.candidate_binding(binding(arm))[0]==arm

@pytest.mark.parametrize('key,value',[('candidate_arm','K1'),('candidate_manifest_sha256',None),
    ('candidate_manifest_sha256','x'*64),('candidate_manifest_kind','FC_P026_K1_HISTORY_FORCE_FNO'),
    ('candidate_manifest_kind','FC_P064_ARM_B_CONTROLLED_AERO_FORCE_FNO')])
def test_wrong_identity_rejected(key,value):
    row=binding();row[key]=value
    with pytest.raises(ValueError):m.candidate_binding(row)

def test_original_lifecycle_exact_except_identity():
    old=OLD.read_text()
    assert hashlib.sha256(old.encode()).hexdigest()=='8363986dc675cc8efe09285b60c25557f1307c7ffcf49a54e0e4388ce3ff9a47'
    source=SOURCE.read_text().replace('P064_CANDIDATE_DIVERSE_H5_32768','EXPLORATORY_DIVERSE_H5_32768')
    tree=ast.parse(source)
    tree.body=[node for node in tree.body if not (isinstance(node,ast.FunctionDef) and node.name=='candidate_binding')]
    run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run')
    run.body=[node for node in run.body if not (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
        and isinstance(node.value.func,ast.Name) and node.value.func.id=='candidate_binding')]
    block=next(node for node in run.body if isinstance(node,ast.Try))
    block.body=[node for node in block.body if 'worker candidate differs' not in ast.unparse(node)]
    assert ast.dump(tree,include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False)
