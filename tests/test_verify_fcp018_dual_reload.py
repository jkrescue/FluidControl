import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import verify_fcp018_dual_reload as v


@pytest.fixture
def files(tmp_path):
    source=tmp_path/'chain/numerical_source';source.mkdir(parents=True)
    for name in ('src/fluid_control/dual_fno.py','src/fluid_control/calibrated_checkpoint.py','scripts/train_tandem_fno.py'):
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
    for name in ('scripts/train_fcp013_independent_force_fno.py','scripts/verify_fcp018_dual_reload.py','receipt.json'):
        path=source.parent/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
    config=tmp_path/'config';config.write_text('fixture')
    manifest=tmp_path/'candidate/dual_model_manifest.json';manifest.parent.mkdir()
    protocol=Path(__file__).resolve().parents[1]/'docs/FC_P018_TRAINING_PROTOCOL_20261005.json'
    (manifest.parent/'training_protocol.json').write_bytes(protocol.read_bytes())
    payload=dict(kind='FC_P018_REDUCED_RATE_FORCE_FNO',**v.EXPERIMENT,config_sha256=v.sha(config))
    for role in ('flow','aerodynamic'):
        folder=manifest.parent/role;folder.mkdir();part=dict(checkpoint_relative_directory=role)
        for field in ('model','state'):
            path=folder/field;path.write_text(role+field)
            part.update({field+'_file':field,field+'_sha256':v.sha(path)})
        payload[role]=part
    manifest.write_text(json.dumps(payload))
    result=manifest.parent/'result.json';result.write_text(json.dumps(dict(
        status='FC_P018_REDUCED_RATE_TRAINING_COMPLETE_NOT_ADMISSION',**v.EXPERIMENT,
        dual_model_manifest_sha256=v.sha(manifest),flow_tensor_sha256_before='a'*64,
        flow_tensor_sha256_after='a'*64,aerodynamic_tensor_sha256_after='b'*64)))
    return manifest,config,result,source


def test_receipt_protocol_and_explicit_profile(files,tmp_path):
    receipt=tmp_path/'receipt.json';payload=v.bindings(*files);receipt.write_text(json.dumps(payload))
    assert v.validate_receipt(receipt,*files)==payload
    assert payload['training_protocol_sha256']==v.PROTOCOL_SHA and payload['actual_learning_rate']==1.5625e-7
    assert payload['scientific_admission'] is False and payload['gpu_used'] is False
    assert "configure_profile('p018')" in Path(v.__file__).read_text()


@pytest.mark.parametrize('field,value',[('kind','FC_P015_WINDOW_ACCUMULATION_FORCE_FNO'),
    ('actual_learning_rate',1e-5),('training_protocol_file','../training_protocol.json'),('optimizer_steps',1368)])
def test_wrong_profile_or_lr_rejected(files,field,value):
    path=files[0];payload=json.loads(path.read_text());payload[field]=value;path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):v.bindings(*files)


def test_actual_protocol_mutation_rejected(files):
    path=files[0].parent/'training_protocol.json';path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='protocol'):v.bindings(*files)


def test_cpu_gpu_visibility_guard(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    with pytest.raises(RuntimeError):v.execute_cpu(SimpleNamespace(),{})
