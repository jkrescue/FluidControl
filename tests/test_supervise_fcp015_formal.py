import io
import json
import signal
from types import SimpleNamespace

import pytest
import supervise_fcp015_formal as m


def info():
    return {'Id':'a'*64,'Image':m.IMAGE,'State':{'Running':True},'Mounts':[
        {'Type':'bind','Destination':d,'Source':str(s),'RW':rw}
        for d,s,rw in [('/workspace/output',m.OUTPUT,True),
                        ('/workspace/src',m.CHAIN/'numerical_source/src',False),
                        ('/workspace/scripts',m.CHAIN/'numerical_source/scripts',False)]]}


def test_memory():
    assert m.parse_memory('MemFree: 20971520 kB\nMemAvailable: 20971521 kB') == {
        'MemFree':20971520,'MemAvailable':20971521}
    with pytest.raises(m.GuardFailure): m.parse_memory('MemAvailable: 20971521 kB')


def test_ownership():
    assert m.owns_container(info())


@pytest.mark.parametrize('mutation', ['image','running','output','source','scripts','rw','type'])
def test_reject_other_container(mutation):
    x=info()
    if mutation=='image': x['Image']='other'
    elif mutation=='running': x['State']['Running']=False
    elif mutation=='output': x['Mounts'][0]['Source']+='-other'
    elif mutation=='source': x['Mounts'][1]['Source']+='-other'
    elif mutation=='scripts': x['Mounts'][2]['Source']+='-other'
    elif mutation=='rw': x['Mounts'][0]['RW']=False
    else: x['Mounts'][0]['Type']='volume'
    assert not m.owns_container(x)


def test_environment_overrides(monkeypatch,tmp_path):
    monkeypatch.setenv('FCP_POSTEVAL_CANDIDATE','/wrong')
    monkeypatch.setenv('FCP008_HOST_PYTHON','/wrong')
    args=SimpleNamespace(training_approval_sha256='a'*64,observation_sha256='b'*64,
                         approval=tmp_path/'approval',approval_sha256='c'*64)
    env=m.child_environment(args)
    assert env['FCP_POSTEVAL_CANDIDATE']==str(m.CANDIDATE)
    assert env['FCP008_HOST_PYTHON']=='/home/USER/env_isaaclab/bin/python'
    assert env['FCP_POSTEVAL_PROFILE']=='p015'
    assert env['FCP008_EXECUTION_APPROVAL_SHA256']=='a'*64
    assert env['FCP015_EXECUTION_OBSERVATION_SHA256']=='b'*64


@pytest.mark.parametrize('scenario',['success','free','available','signal','timeout','exit','orphan','query'])
def test_monitor(scenario):
    stream=io.StringIO()
    memory={'MemFree':m.FLOOR_KIB,'MemAvailable':m.FLOOR_KIB}
    if scenario in ('free','available'): memory['MemFree' if scenario=='free' else 'MemAvailable']-=1
    proc=SimpleNamespace(poll=lambda: 2 if scenario=='exit' else 0)
    ticks=iter([0,m.LIMIT_SECONDS if scenario=='timeout' else 0])
    def discover():
        if scenario=='query': raise RuntimeError('daemon unavailable')
        return ['a'*64] if scenario=='orphan' else []
    def run():
        m.monitor(proc,stream,lambda:scenario=='signal',clock=lambda:next(ticks),
                  memory_reader=lambda:memory,discover=discover)
    if scenario=='success': run()
    else:
        with pytest.raises((m.GuardFailure,RuntimeError)): run()
    if scenario!='query': assert json.loads(stream.getvalue())['mem_free_kib']==memory['MemFree']


def test_poll_interval():
    calls=[]
    poll=iter([None,0])
    m.monitor(SimpleNamespace(poll=lambda:next(poll)),io.StringIO(),lambda:False,
              clock=lambda:0,sleep=calls.append,
              memory_reader=lambda:{'MemFree':m.FLOOR_KIB,'MemAvailable':m.FLOOR_KIB},discover=lambda:[])
    assert calls==[2]


def test_cleanup_owned_ids_only(monkeypatch):
    commands=[]; signals=[]
    scans=iter([['a'*64],[],[],[],[]])
    monkeypatch.setattr(m,'discover_owned',lambda:next(scans))
    monkeypatch.setattr(m.time,'sleep',lambda _:None)
    monkeypatch.setattr(m.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    monkeypatch.setattr(m.subprocess,'run',lambda cmd,**kw:commands.append(cmd) or SimpleNamespace(returncode=0))
    proc=SimpleNamespace(pid=456,poll=lambda:None,wait=lambda **kw:0)
    m.stop_owned(proc)
    assert commands==[['docker','rm','--force','a'*64]]
    assert signals==[(456,signal.SIGSTOP),(456,signal.SIGTERM),(456,signal.SIGCONT)]


def test_cleanup_query_failure_still_stops_child(monkeypatch):
    signals=[]
    def fail(): raise RuntimeError('daemon unavailable')
    monkeypatch.setattr(m,'discover_owned',fail)
    monkeypatch.setattr(m.os,'killpg',lambda pid,sig:signals.append(sig))
    with pytest.raises(m.GuardFailure):
        m.stop_owned(SimpleNamespace(pid=456,poll=lambda:None,wait=lambda **kw:0))
    assert signals==[signal.SIGSTOP,signal.SIGTERM,signal.SIGCONT]


def test_discovery_exit_race(monkeypatch):
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'a'*64+'\n'+'b'*64)
    def inspect(cmd,**kw):
        return SimpleNamespace(returncode=1,stderr='Error: No such object: b',stdout=json.dumps([info()]))
    monkeypatch.setattr(m.subprocess,'run',inspect)
    assert m.discover_owned()==['a'*64]


def test_created_owned_container():
    x=info();x['State']={'Running':False,'Status':'created'}
    assert m.owns_container(x)


def test_cleanup_late_start(monkeypatch):
    commands=[]
    scans=iter([[],[],['a'*64],[],[]])
    monkeypatch.setattr(m,'discover_owned',lambda:next(scans))
    monkeypatch.setattr(m.time,'sleep',lambda _:None)
    monkeypatch.setattr(m.subprocess,'run',lambda cmd,**kw:commands.append(cmd) or SimpleNamespace(returncode=0))
    m.stop_owned(SimpleNamespace(poll=lambda:0))
    assert commands==[['docker','rm','--force','a'*64]]


def test_discovery_daemon_failure(monkeypatch):
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'a'*64)
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stderr='permission denied',stdout='[]'))
    with pytest.raises(m.GuardFailure): m.discover_owned()
