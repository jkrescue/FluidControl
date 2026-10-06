import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).parents[1]/'scripts/serve_live_research_dashboard.py'
spec = importlib.util.spec_from_file_location('dashboard_replay', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(tmp_path, monkeypatch):
    result_path=tmp_path/'artifacts/projected_policy_h1_h5_inference_20261006/result.json'
    result_path.parent.mkdir(parents=True)
    item={'velocity_relative_l2':.0421266833,'field_relative_l2_u_v_p':[.031,.134,.1358591903],
          'force_channel_mae':[.001,.02,.03,.1499264352]}
    zero={'velocity_relative_l2':.0070497739,'field_relative_l2_u_v_p':[.005,.022,.0187875023],
          'force_channel_mae':[.0001,.001,.007,.0108058974]}
    result_path.write_text(json.dumps({'status':'PROJECTED_POLICY_H1_H5_REPLAY_COMPLETE_NOT_ADMISSION',
        'records':[{}]*16,'endpoints':80,'optimizer_steps':0,'model_tensors_unchanged':True,
        'scientific_admission':False,'branches':{'mpc':{'5':item},'zero':{'5':zero}}}))
    review=tmp_path/'docs/PROJECTED_POLICY_H1_H5_INFERENCE_TERMINAL_REVIEW_20261006.md'
    review.parent.mkdir();review.write_text('review')
    preview=tmp_path/'artifacts/projected_policy_h1_h5_field_preview_20261006_v3';preview.mkdir()
    manifest={'source_inference_result_sha256':'247af0405d9e622f0b3b3b5dbc64e46c20890439fcd5216682b8d973957fd00d',
        'saved_predictions_only':True,'model_rerun':False,'cfd_rerun':False,'scientific_admission':False}
    (preview/'result.json').write_text(json.dumps(manifest))
    (preview/'mpc_start_0000_h1_h5.png').write_bytes(b'png0')
    (preview/'mpc_start_0700_h1_h5.png').write_bytes(b'png7')
    hashes={result_path.read_bytes():'247af0405d9e622f0b3b3b5dbc64e46c20890439fcd5216682b8d973957fd00d',
        review.read_bytes():'197b385617420e5f0e9cb7c8dfb25d957e98f85ea93f280bb2d6c0effc898faa',
        (preview/'result.json').read_bytes():'e2a08c418ceeeb7b0eca648c9e8e2945faf7db4e8146e42e4d9cebaa8fcb8258',
        b'png0':'41a02dd81a9850705acbd4e1f8c11c6cb1cf9f5aee7ec5885ea39b2fcfbf5eb9',
        b'png7':'0bc6530b425732454c1b8f0056137d7fe308bf9a42d71ff806eacff7c2b9bb40'}
    monkeypatch.setattr(m.hashlib,'sha256',lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'bad')))
    unit='\n'.join(['InvocationID=627cb6b8b59a40aca3bb9159fb617eb3','MainPID=0',
        'ActiveState=active','SubState=exited','Result=success','ExecMainStatus=0'])
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:unit)
    return result_path,review


def test_reviewed_replay_keeps_controlled_and_zero_separate(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    result=m._projected_policy_h1_h5_inference(tmp_path)
    assert result['verified'] and result['reviewed'] and not result['online_control']
    assert result['controlled_h5']['velocity_relative_l2']==.0421266833
    assert result['zero_h5']['velocity_relative_l2']==.0070497739
    assert result['images']['0000']['sha256'].startswith('41a02d')
    assert '不是在线未知未来动作预测' in m.PAGE


def test_wrong_review_or_unit_rejected(tmp_path,monkeypatch):
    _,review=fixture(tmp_path,monkeypatch)
    review.write_text('changed')
    assert not m._projected_policy_h1_h5_inference(tmp_path)['verified']
