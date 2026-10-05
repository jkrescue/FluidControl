import ast
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import supervise_fcp018_formal as m


def test_reuses_reviewed_resource_cleanup_algorithms():
    root=Path(__file__).parents[1]/'scripts'
    def functions(name):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse((root/name).read_text()).body
                if isinstance(n,ast.FunctionDef)}
    old,new=functions('supervise_fcp015_formal.py'),functions('supervise_fcp018_formal.py')
    for key in ('parse_memory','read_memory','owns_container','discover_owned','stop_owned','stop_containers','monitor'):
        assert old[key]==new[key]


def test_exact_p018_environment(tmp_path):
    args=SimpleNamespace(training_approval_sha256='a'*64,observation_sha256='b'*64,
        approval=tmp_path/'approval.json',approval_sha256='c'*64)
    env=m.child_environment(args)
    assert env['FCP_POSTEVAL_PROFILE']=='p018'
    assert env['FCP018_EXECUTION_OBSERVATION_SHA256']=='b'*64
    assert env['FCP008_POSTEVAL_TOKEN']=='EXECUTE_APPROVED_FC_P018_POSTEVAL'
    assert m.CHAIN.name=='p018_posteval_chain_d46622b7cb51_immutable'
    assert m.CANDIDATE.name=='fcp018_reduced_rate_training_20261005'
    assert m.OUTPUT.name=='posteval_fc_p018'


def info(output):
    return {'Image':m.IMAGE,'State':{'Running':True},'Mounts':[
        {'Type':'bind','Destination':d,'Source':str(s),'RW':rw}
        for d,s,rw in [('/workspace/output',output,True),
            ('/workspace/src',m.CHAIN/'numerical_source/src',False),
            ('/workspace/scripts',m.CHAIN/'numerical_source/scripts',False)]]}


def test_only_actual_p018_owned():
    assert m.owns_container(info(m.OUTPUT))
    assert not m.owns_container(info(m.REPO/'artifacts/fcp015_window_accumulation_training_20261005/posteval_fc_p015'))


@pytest.mark.parametrize('field',['MemAvailable','MemFree'])
def test_each_memory_floor_enforced(field):
    memory={'MemAvailable':m.FLOOR_KIB,'MemFree':m.FLOOR_KIB};memory[field]-=1
    with pytest.raises(m.GuardFailure,match='below 20'):
        m.monitor(SimpleNamespace(poll=lambda:None),io.StringIO(),lambda:False,
            clock=lambda:0,memory_reader=lambda:memory,discover=lambda:[])
