import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import verify_fcp015_dual_reload as v


@pytest.fixture
def files(tmp_path):
    source=tmp_path/'chain/numerical_source'
    source.mkdir(parents=True)
    for name in ('src/fluid_control/dual_fno.py','src/fluid_control/calibrated_checkpoint.py','scripts/train_tandem_fno.py'):
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('source fixture')
    for name in ('scripts/train_fcp013_independent_force_fno.py','scripts/verify_fcp015_dual_reload.py','receipt.json'):
        p=source.parent/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('chain fixture')
    config=tmp_path/'config.yaml'; config.write_text('fixture')
    manifest=tmp_path/'dual/dual_model_manifest.json';manifest.parent.mkdir()
    payload={'kind':'FC_P015_WINDOW_ACCUMULATION_FORCE_FNO',**v.EXPERIMENT,'config_sha256':v.sha(config)}
    for role in ('flow','aerodynamic'):
        directory=manifest.parent/role;directory.mkdir()
        row={'checkpoint_relative_directory':role}
        for field in ('model','state'):
            p=directory/field;p.write_text(role+field)
            row.update({field+'_file':field,field+'_sha256':v.sha(p)})
        payload[role]=row
    manifest.write_text(json.dumps(payload))
    result=tmp_path/'result.json'
    result.write_text(json.dumps({'status':'FC_P015_WINDOW_ACCUMULATION_TRAINING_COMPLETE_NOT_ADMISSION',
        **v.EXPERIMENT,'dual_model_manifest_sha256':v.sha(manifest),'flow_tensor_sha256_before':'a'*64,
        'flow_tensor_sha256_after':'a'*64,'aerodynamic_tensor_sha256_after':'b'*64}))
    return manifest,config,result,source


def test_receipt_binds_files_sources_and_exact_nonexecution_flags(files,tmp_path):
    receipt=tmp_path/'verified.json'
    expected=v.bindings(*files)
    receipt.write_text(json.dumps(expected))
    assert v.validate_receipt(receipt,*files)==expected
    for key in ('status','device','required_official_image_id','gpu_used','forward_performed','tensor_sha256',
                'runtime_source_sha256','dual_manifest_sha256','posteval_chain_receipt_sha256'):
        receipt.write_text(json.dumps({**expected,key:'wrong'}))
        with pytest.raises(ValueError):v.validate_receipt(receipt,*files)


@pytest.mark.parametrize('target',['checkpoint','config','source','result','chain'])
def test_after_verification_mutation_rejected(files,tmp_path,target):
    manifest,config,result,source=files
    receipt=tmp_path/'verified.json';receipt.write_text(json.dumps(v.bindings(*files)))
    path={'checkpoint':manifest.parent/'aerodynamic/model','config':config,
          'source':source/'src/fluid_control/dual_fno.py','result':result,
          'chain':source.parent/'receipt.json'}[target]
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError):v.validate_receipt(receipt,*files)


def test_reject_p013_or_wrong_steps(files):
    manifest=files[0]; original=json.loads(manifest.read_text())
    for key,value in [('kind','FC_P013_INDEPENDENT_FORCE_FNO'),('optimizer_steps',1368),('accumulation_windows',1)]:
        manifest.write_text(json.dumps({**original,key:value}))
        with pytest.raises(ValueError): v.bindings(*files)


def test_loaded_models_require_matching_tensors_cpu_and_frozen():
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__();self.flow_model=torch.nn.Linear(1,1);self.aerodynamic_model=torch.nn.Linear(1,1)
    a=Adapter().requires_grad_(False)
    def digest(m):return 'flow' if m is a.flow_model else 'aero'
    v.verify_loaded_tensors(a,{'flow':'flow','aerodynamic':'aero'},digest)
    with pytest.raises(ValueError):v.verify_loaded_tensors(a,{'flow':'wrong','aerodynamic':'aero'},digest)
    a.aerodynamic_model.requires_grad_(True)
    with pytest.raises(ValueError):v.verify_loaded_tensors(a,{'flow':'flow','aerodynamic':'aero'},digest)


def test_cpu_execution_requires_explicit_gpu_invisibility(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    with pytest.raises(RuntimeError,match='CUDA_VISIBLE_DEVICES'):
        v.execute_cpu(SimpleNamespace(),{})
