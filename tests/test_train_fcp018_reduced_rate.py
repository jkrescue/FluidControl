import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import train_fcp018_reduced_rate as p


ROOT=Path(__file__).resolve().parents[1]


def protocol(): return p.validate_protocol(ROOT/'docs/FC_P018_TRAINING_PROTOCOL_20261005.json',p.PROTOCOL_SHA)


def helpers():
    return p.load_helpers(SimpleNamespace(accumulation_helper=ROOT/'scripts/train_fcp015_window_accumulation.py',
        diagnostic_script=ROOT/'scripts/diagnose_fcp014_train_objective.py',
        source_root=ROOT/'artifacts/fcp013_training_source_1634c05_immutable'))


def test_actual_pinned_protocol():
    value=protocol()
    assert value['sole_optimizer_override']=={'learning_rate':p.LEARNING_RATE}
    assert value['optimizer_steps']==171 and value['training_windows']==1368 and value['accumulation_windows']==8


def test_no_self_declared_new_protocol(tmp_path):
    path=tmp_path/'protocol.json'
    value=protocol();value['sole_optimizer_override']['learning_rate']=1e-5
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):p.validate_protocol(path,p.sha(path))
    with pytest.raises(ValueError):p.validate_protocol(path,p.PROTOCOL_SHA)


def test_helper_sha_failclosed(tmp_path):
    path=tmp_path/'helper.py';path.write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError):p.load_helpers(SimpleNamespace(accumulation_helper=path))


def test_actual_helper_keeps_eight_window_clipping_and_low_rate():
    helper,_,_=helpers()
    model=torch.nn.Linear(1,1,dtype=torch.float64)
    optimizer=torch.optim.AdamW(model.parameters(),lr=p.LEARNING_RATE,weight_decay=1e-4)
    seen=[]
    def run(index):
        seen.append(index)
        loss=(model(torch.tensor([[index+1.]],dtype=torch.float64))-10).square().mean()
        loss.backward()
        return {k:float(loss.detach()) for k in ('h1_balanced','ar_balanced','total')}
    record=helper.accumulation_step(model,optimizer,range(8),run,lambda _:dict(finite=True))
    p.check_optimizer(optimizer,protocol(),expected_steps=1)
    assert seen==list(range(8)) and record['windows']==8 and record['optimizer_steps']==1
    assert record['preclip_mean_gradient_norm']>1
    with pytest.raises(ValueError):p.check_optimizer(optimizer,protocol(),expected_steps=2)


@pytest.mark.parametrize('field,value',[('lr',1e-5),('weight_decay',0),('eps',1e-7),('betas',(.8,.9))])
def test_optimizer_mismatch(field,value):
    optimizer=torch.optim.AdamW(torch.nn.Linear(1,1).parameters(),lr=p.LEARNING_RATE,weight_decay=1e-4)
    optimizer.param_groups[0][field]=value
    with pytest.raises(ValueError):p.check_optimizer(optimizer,protocol())


def test_nonfinite_optimizer():
    model=torch.nn.Linear(1,1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=p.LEARNING_RATE,weight_decay=1e-4)
    optimizer.state[model.weight]['exp_avg']=torch.tensor(float('nan'))
    with pytest.raises(FloatingPointError):p.check_optimizer(optimizer,protocol())


def test_frozen_bias_invariants():
    _,_,objective=helpers()
    model=torch.nn.Linear(1,1);model.bias.requires_grad_(False)
    frozen={'bias':objective.tensor_sha256(model.bias)}
    p.check_frozen(model,frozen,objective)
    model.bias.grad=torch.ones_like(model.bias)
    with pytest.raises(RuntimeError):p.check_frozen(model,frozen,objective)
    model.bias.grad=None
    with torch.no_grad():model.bias.add_(1)
    with pytest.raises(RuntimeError):p.check_frozen(model,frozen,objective)


def test_manifest_distinct_identity_and_protocol(tmp_path):
    helper,_,objective=helpers()
    for name in ('flow/FNO.0.0.mdlus','flow/checkpoint.0.0.pt','aerodynamic/FNO.0.1.mdlus','aerodynamic/checkpoint.0.1.pt'):
        path=tmp_path/name;path.parent.mkdir(exist_ok=True);path.write_bytes(b'fixture')
    manifest=p.manifest_payload(helper,objective,tmp_path,tmp_path/'aerodynamic/FNO.0.1.mdlus',
                              tmp_path/'aerodynamic/checkpoint.0.1.pt',{},p.PROTOCOL_SHA)
    assert manifest['kind']==p.SYSTEM_KIND and manifest['status']==p.MANIFEST_STATUS
    assert manifest['aerodynamic']['metadata_kind']==p.AERO_KIND
    for key,value in p.identity(p.PROTOCOL_SHA).items():
        assert manifest[key]==value and manifest['training_semantics'][key]==value
    assert manifest['training_semantics']['learning_rate']==p.LEARNING_RATE
    assert manifest['config_sha256']==manifest['base_config_sha256']==objective.CONFIG_SHA
    assert manifest['flow']['metadata_kind']==objective.PARENT_KIND
    assert 'FC-P015' not in json.dumps(manifest)


def test_identity_shared_metadata_contract():
    value=p.identity(p.PROTOCOL_SHA)
    assert value['training_protocol_file']=='training_protocol.json'
    assert value['actual_learning_rate']==1e-5/64
    assert value['training_experiment']=='FC-P018'
    assert value['optimizer_steps']==171
