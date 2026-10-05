"""Synthetic provenance fixtures only; never a surrogate admission result."""
import json
from types import SimpleNamespace

import pytest

import fluid_control.dual_control_contract as c

IDENTITY = dict(dual_manifest_sha256='a'*64,flow_model_sha256='b'*64,flow_state_sha256='c'*64,
                checkpoint_sha256='d'*64,checkpoint_state_sha256='e'*64)
EXPERIMENT = dict(training_experiment='FC-P015',accumulation_windows=8,training_windows=1368,optimizer_steps=171)


@pytest.mark.parametrize('manifest_kind', [c.SYSTEM_KIND,c.P015_SYSTEM_KIND])
def test_only_exact_receipt_profile_is_accepted(manifest_kind):
    status,kind,_=c.receipt_profile(manifest_kind)
    receipt=dict(status=status,candidate_kind=kind,checkpoint_epoch=1,protocol=c.PROTOCOL,
                 frozen_test_accessed=False,ppo_auto_launched=False,**IDENTITY)
    c.check_receipt_identity(receipt,IDENTITY,manifest_kind=manifest_kind)
    other=c.P015_SYSTEM_KIND if manifest_kind == c.SYSTEM_KIND else c.SYSTEM_KIND
    with pytest.raises(ValueError):c.check_receipt_identity(receipt,IDENTITY,manifest_kind=other)
    for key in IDENTITY:
        with pytest.raises(ValueError):c.check_receipt_identity({**receipt,key:'wrong'},IDENTITY,manifest_kind=manifest_kind)
    with pytest.raises(ValueError):c.receipt_profile('FC_P999_ARBITRARY')


@pytest.fixture
def evidence(tmp_path,monkeypatch):
    root=tmp_path/'candidate/posteval_fc_p015';root.mkdir(parents=True)
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    identity=SimpleNamespace(payload={'kind':c.P015_SYSTEM_KIND,'config_sha256':'f'*64,'normalization_sha256':'0'*64},
        manifest_sha256='a'*64,flow=SimpleNamespace(model_sha256='b'*64,state_sha256='c'*64),
        aerodynamic=SimpleNamespace(directory=tmp_path/'aero',model_sha256='d'*64,state_sha256='e'*64))
    monkeypatch.setattr(c,'validate_dual_fno_manifest',lambda *a,**kw:identity)
    monkeypatch.setattr(c,'validate_dual_runtime_files',lambda *a,**kw:None)
    result_path=root.parent/'candidate/result.json'
    write(result_path,{**EXPERIMENT,'flow_tensor_sha256_after':'1'*64,'aerodynamic_tensor_sha256_after':'2'*64})
    write(root.parent/'completion_receipt.json',{'software_fixture_only':True})
    lineage={'status':'FC_P015_DUAL_CANDIDATE_LINEAGE_PASS_NOT_ADMISSION',
        'candidate_kind':'fcp015_window_accumulation_dual_fno','checkpoint_epoch':1,**EXPERIMENT,**IDENTITY,
        'candidate_result_sha256':c.sha256(result_path)}
    write(root/'lineage.json',lineage)
    write(root/'precision.json',{'software_fixture_only':True})
    reload={'status':'FC_P015_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION',**EXPERIMENT,
        'device':'cpu','dual_manifest_sha256':'a'*64,'flow_model_sha256':'b'*64,'flow_state_sha256':'c'*64,
        'aerodynamic_model_sha256':'d'*64,'aerodynamic_state_sha256':'e'*64,'config_sha256':'f'*64,
        'training_result_sha256':c.sha256(result_path),'posteval_chain_receipt_sha256':'9'*64,
        'tensor_sha256':{'flow':'1'*64,'aerodynamic':'2'*64},'forward_performed':False,'optimizer_created':False,
        'model_saved':False,'gpu_used':False,'scientific_admission':False,'ppo_authorized':False}
    write(root.parent/'dual_reload_receipt.json',reload)
    approval={'status':'FC_P015_FORMAL_EVALUATION_APPROVED',**EXPERIMENT,
        'candidate_kind':lineage['candidate_kind'],'checkpoint_epoch':1,'candidate_model_sha256':'d'*64,
        'candidate_state_sha256':'e'*64,'dual_manifest_sha256':'a'*64,'flow_model_sha256':'b'*64,
        'flow_state_sha256':'c'*64,'candidate_result_sha256':c.sha256(result_path),
        'candidate_completion_receipt_sha256':c.sha256(root.parent/'completion_receipt.json'),
        'dual_reload_receipt_sha256':c.sha256(root.parent/'dual_reload_receipt.json'),
        'formal_evaluation_authorized':True,'protocol':c.PROTOCOL,'frozen_test_accessed':False,'ppo_auto_launch':False}
    write(root/'evidence/formal_evaluation_approval.json',approval)
    hashes={'lineage_sha256':c.sha256(root/'lineage.json'),'precision_sha256':c.sha256(root/'precision.json'),
        'formal_evaluation_approval_sha256':c.sha256(root/'evidence/formal_evaluation_approval.json'),
        'posteval_chain_receipt_sha256':'9'*64}
    for name in ('validation10','dynamic6','force_window'):
        write(root/f'step_receipts/{name}.json',{'status':'FC_P015_POSTEVAL_STEP_COMPLETE','candidate_kind':lineage['candidate_kind'],
            'checkpoint_epoch':1,'step':name,**IDENTITY,**hashes})
    for name in ('validation10/evaluation.json','validation10/segments.json','validation10/endpoint_gate.json',
                 'validation10/diagnostic.json','dynamic6/diagnostic.json','dynamic6/evaluation.json','dynamic6/segments.json'):
        write(root/name,{'software_fixture_only':True})
    for name in ('force_window/result.json','development_gate.json'):
        write(root/name,{'status':'DYNAMIC_FNO_DEVELOPMENT_ADMISSION_PASS'})
    auditor=tmp_path/'fake_numerical_auditor.py'
    auditor.write_text('import json\ndef audit(path, checkpoint): return json.loads(path.read_text())\n')
    monkeypatch.setattr(c,'DEVELOPMENT_AUDITOR_SHA',c.sha256(auditor))
    receipt=root/'receipt.json'
    def refresh():
        write(receipt,{'status':'FC_P015_POSTEVAL_COMPLETE','candidate_kind':lineage['candidate_kind'],
            'checkpoint_epoch':1,'protocol':c.PROTOCOL,'frozen_test_accessed':False,'ppo_auto_launched':False,
            **IDENTITY,**hashes,'sha256':{str(p.relative_to(root)):c.sha256(p) for p in root.rglob('*') if p.is_file() and p!=receipt}})
    def run():
        return c.verify_dual_control_binding(manifest_path=tmp_path/'manifest.json',expected_manifest_sha256='a'*64,
            training_config=tmp_path/'config',normalization_path=tmp_path/'normalization',
            checkpoint_dir=identity.aerodynamic.directory,expected_checkpoint_sha256='d'*64,
            posteval_receipt=receipt,expected_posteval_receipt_sha256=c.sha256(receipt),development_auditor=auditor)
    refresh()
    return root,receipt,identity,refresh,run


def test_valid_p015_recomputes_original_gate_without_authorizing_policy(evidence):
    root,receipt,identity,refresh,run=evidence
    binding=run()
    assert binding['status']=='DUAL_CONTROL_IDENTITY_VERIFIED_NOT_CONTROL_SUCCESS'
    assert binding['canonical_endpoint_window_dynamic_gates_still_required']
    assert binding['policy_trained'] is False and binding['real_cfd_control_validated'] is False
    for name in ('force_window/result.json','development_gate.json'):
        (root/name).write_text(json.dumps({'status':'DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL'}))
    refresh()
    with pytest.raises(ValueError,match='not passed'):run()


@pytest.mark.parametrize('path,key,value',[
    ('lineage.json','optimizer_steps',1368),('lineage.json','accumulation_windows',1),
    ('lineage.json','training_experiment','FC-P013'),('lineage.json','training_windows',171),
    ('step_receipts/dynamic6.json','status','FC_P013_POSTEVAL_STEP_COMPLETE'),
    ('step_receipts/validation10.json','candidate_kind','fcp013_independent_force_dual_fno'),
    ('step_receipts/force_window.json','lineage_sha256','0'*64),
    ('step_receipts/dynamic6.json','posteval_chain_receipt_sha256','0'*64),
    ('evidence/formal_evaluation_approval.json','dual_reload_receipt_sha256','0'*64),
])
def test_p015_cross_profile_and_provenance_tampering_rejected(evidence,path,key,value):
    root,receipt,identity,refresh,run=evidence
    p=root/path; payload=json.loads(p.read_text());payload[key]=value;p.write_text(json.dumps(payload))
    refresh()  # Even a refreshed outer file table cannot change the declared identity.
    with pytest.raises(ValueError):run()


def test_manifest_not_receipt_selects_profile(evidence):
    root,receipt,identity,refresh,run=evidence
    identity.payload['kind']=c.SYSTEM_KIND
    with pytest.raises(ValueError,match='receipt identity'):run()


def test_reload_cannot_be_swapped_or_deleted(evidence):
    root,receipt,identity,refresh,run=evidence
    path=root.parent/'dual_reload_receipt.json';payload=json.loads(path.read_text())
    payload['aerodynamic_model_sha256']='0'*64;path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):run()
    path.unlink()
    with pytest.raises(FileNotFoundError):run()
