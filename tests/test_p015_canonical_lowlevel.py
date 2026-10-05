"""CPU-only P015 route guards; no policy or scientific result."""
import json
from types import SimpleNamespace

import pytest
import torch

from test_full40_canonical_ppo_dry_run import MODULE
from test_p013_canonical_endpoint_path_view import evidence


@pytest.mark.parametrize('payload',[
    {'status':'FC_P015_WINDOW_ACCUMULATION_FORCE_FNO_AERODYNAMIC_CHECKPOINT'},
    {'kind':'FC_P015_WINDOW_ACCUMULATION_FORCE_FNO'},
    {'candidate_kind':'fcp015_window_accumulation_dual_fno'},
    {'metadata':{'training_experiment':'FC-P015'}},
    {'nested':[{'metadata_kind':'FC_P015_WINDOW_ACCUMULATION_FORCE_FNO_AERODYNAMIC_CHECKPOINT'}]},
])
def test_p015_evidence_cannot_select_single_model(payload):
    with pytest.raises(ValueError,match='complete dual arguments'):
        MODULE.require_single_model_identity(payload)


def test_p015_actual_official_metadata_without_dual_flags_is_rejected(tmp_path):
    checkpoint=tmp_path/'aerodynamic';checkpoint.mkdir()
    torch.save({'metadata':{'status':'FC_P015_WINDOW_ACCUMULATION_FORCE_FNO_AERODYNAMIC_CHECKPOINT',
        'training_experiment':'FC-P015','optimizer_steps':171,'training_windows':1368,'accumulation_windows':8}},
        checkpoint/'checkpoint.0.1.pt')
    args=SimpleNamespace(checkpoint_dir=checkpoint)
    with pytest.raises(ValueError,match='complete dual arguments'):
        MODULE.verify_single_model_selection(args,{})


@pytest.mark.parametrize('status,expected',[
    ('FC_P013_POSTEVAL_COMPLETE','P013_ENDPOINT_PATH_VIEW_IDENTITY_VERIFIED'),
    ('FC_P015_POSTEVAL_COMPLETE','P015_ENDPOINT_PATH_VIEW_IDENTITY_VERIFIED'),
])
def test_endpoint_label_matches_verified_receipt_and_metrics_unchanged(evidence,status,expected):
    args,gate,_=evidence
    path=args['dual_options']['dual_posteval_receipt']
    receipt=MODULE.read_json(path);receipt['status']=status;path.write_text(json.dumps(receipt))
    before={name:args[name].read_bytes() for name in ('validation_report','validation_segments','validation_gate')}
    actual,mapping=MODULE.recompute_p013_endpoint_view(**args)
    assert mapping['status']==expected
    assert actual == gate and mapping['numerical_evidence_changed'] is False
    assert all(args[name].read_bytes()==content for name,content in before.items())


def test_unknown_endpoint_profile_rejected_before_temporary_view(evidence,monkeypatch):
    args,_,_=evidence
    path=args['dual_options']['dual_posteval_receipt'];receipt=MODULE.read_json(path)
    receipt['status']='FC_P999_POSTEVAL_COMPLETE';path.write_text(json.dumps(receipt))
    monkeypatch.setattr(MODULE.tempfile,'TemporaryDirectory',lambda **kw:pytest.fail('no unknown profile rewrite'))
    with pytest.raises(ValueError,match='verified dual experiment'):
        MODULE.recompute_p013_endpoint_view(**args)


def test_genuine_legacy_single_model_remains_allowed():
    MODULE.require_single_model_identity({'status':'HISTORICAL_SINGLE_FNO','optimizer_steps':171})
