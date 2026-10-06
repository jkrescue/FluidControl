import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('b00_dashboard',ROOT/'scripts/serve_live_research_dashboard.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)

def test_running_and_wrong_invocation(tmp_path):
    base=tmp_path/'artifacts/b00_controlled_train_conversion_20261006';base.mkdir(parents=True)
    (base/'progress.json').write_text(json.dumps({'written_frames':100}))
    (base/'resources.jsonl').write_text(json.dumps({'MemAvailable':100*2**30}))
    approval=tmp_path/'docs/B00_CONTROLLED_TRAIN_CONVERSION_APPROVAL_20261006.json';approval.parent.mkdir()
    approval.write_text(json.dumps({'driver':{'path':'driver.py'}}))
    hashes=['4fa7192e13bf7ad3a141bffb483710e2400fd8ee243caa60a6e67ab695927686','f96c882a90e7ecaf4a2f8a5fc327764909ab42b4e6bbb11cde8e99205488b075']
    live='InvocationID=f5ee31dd92624f0980a509084de9c756\nActiveState=active\nSubState=running\nExecMainStatus=0\n'
    with patch.object(m,'_small_file_sha256',side_effect=hashes),patch.object(m.subprocess,'check_output',return_value=live):
        out=m._b00_train_conversion(tmp_path)
    assert out['label']=='转换中' and out['written_frames']==100 and out['available_gib']==100
    with patch.object(m,'_small_file_sha256',side_effect=hashes),patch.object(m.subprocess,'check_output',return_value=live.replace('f5ee31dd92624f0980a509084de9c756','wrong')):
        assert m._b00_train_conversion(tmp_path)['label']=='身份未验证'

def test_render_no_training_claim():
    assert '仅 CPU Curator→训练 HDF，无模型训练/新 CFD' in m.PAGE
    assert 'b00_train_conversion' in m.PAGE
