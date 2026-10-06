import hashlib
import json
import pytest
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE if (HERE/'p064_dashboard_progress.py').exists() else HERE.parent/'scripts'))
from p064_dashboard_progress import parse_journal,status,PROFILES

def events():
    return '\n'.join(json.dumps({'MESSAGE':json.dumps(m),'__REALTIME_TIMESTAMP':'1791281947000000'}) for m in [
        {'event':'training_window_complete','history_k':1,'consumed':8},
        {'event':'accumulation_update_complete','history_k':1,'update':1}])

def test_parser():
    result=parse_journal(events());assert result['windows']==8 and result['updates']==1
    assert result['last_update_utc'].endswith('+00:00')

@pytest.mark.parametrize('field,value',[('history_k',True),('consumed',257),('consumed',False)])
def test_bad_event(field,value):
    row={'event':'training_window_complete','history_k':1,'consumed':8};row[field]=value
    with pytest.raises(ValueError):parse_journal(json.dumps({'MESSAGE':json.dumps(row)}))

def test_pending_never_queries(tmp_path):
    assert status(tmp_path,'A',run=lambda *a,**k:1/0)['status']=='尚未启动'

def test_actual_identity_and_zero_exit_not_success(tmp_path):
    p=tmp_path/'source';p.write_bytes(b'source')
    item={'path':'source','sha256':hashlib.sha256(b'source').hexdigest()}
    reg={**PROFILES['A'],'invocation':'a'*32,'approval':item,'driver':item}
    def running(cmd,**kw):
        return 'InvocationID='+('a'*32)+'\nActiveState=active\nSubState=running\nMainPID=10\nExecMainStatus=0' if cmd[0]=='systemctl' else events()
    assert status(tmp_path,'A',reg,running)['status']=='训练中'
    def ended(cmd,**kw):return running(cmd,**kw).replace('SubState=running','SubState=exited').replace('MainPID=10','MainPID=0')
    out=status(tmp_path,'A',reg,ended);assert '等待独立' in out['status'] and not out['terminal_verified']
    def failed(cmd,**kw):return ended(cmd,**kw).replace('ExecMainStatus=0','ExecMainStatus=1')
    assert status(tmp_path,'A',reg,failed)['status']=='失败/停止（未自动重试）'
    reg['invocation']='b'*32
    assert status(tmp_path,'A',reg,running)['status']=='身份或状态未验证'
def test_development_missing_evidence_fails_closed(tmp_path):
    from p064_dashboard_progress import development_summary
    assert development_summary(tmp_path)['verified'] is False


def test_actual_small_reviewed_development_json():
    from pathlib import Path
    from p064_dashboard_progress import development_summary
    root=Path('/workspace/fluid_control')
    data=development_summary(root)
    assert data['verified'] is True
    assert [row['label'] for row in data['rows']]==['K1','A','B']
    assert data['rows'][2]['h1_cl'] < data['rows'][1]['h1_cl']
    assert data['rows'][2]['h5_cd'] > data['rows'][0]['h5_cd']
    assert '尚未训练' in data['current_stage']
