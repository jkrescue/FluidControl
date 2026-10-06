import importlib.util
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest

STAGE=Path(__file__).parents[1]
REPO=Path(os.environ.get('PROJECT_REPO','/workspace/fluid_control'))

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

core=module(STAGE/'scripts/diagnose_projected_policy_h1_h5_replay.py','replay_core_test')
driver=module(STAGE/'scripts/convert_projected_policy_h1_h5_frames.py','replay_convert_test')


def progress():
    def obs(force,omega):
        x=np.zeros(69);x[64:68]=force;x[68]=omega;return x.tolist()
    rows=[];prior=0.
    for index in range(800):
        applied=float(np.clip(np.sin(index/20)*.6,prior-.1,prior+.1))
        rows.append({'step':index+1,'start_time':148+.1*index,'end_time':148+.1*(index+1),
          'requested_omega':applied,'applied_omega':applied,
          'input_observation':obs(index,prior),'output_observation':obs(index+1,applied),
          'zero_observation':obs(index+2,0.)});prior=applied
    return {'completed_cycles':800,'rows':rows}


def packet(time):
    state=np.zeros((3,128,256),np.float32);state[2,0,0]=1;state[2,0,1]=-1
    return {'state':state,'mask':np.ones((1,128,256),np.uint8),
      'time':np.asarray([time],np.float64),'x':np.linspace(8,25,256,dtype=np.float32),
      'y':np.linspace(4,11,128,dtype=np.float32)}


def test_actual_installed_official_reader_roundtrip(tmp_path):
    path=tmp_path/'actual.h5'
    with h5py.File(path,'x') as handle:
        handle['state']=np.zeros((6,3,128,256),np.float32)
        handle['mask']=np.ones((6,1,128,256),np.uint8)
        handle['omega']=np.arange(6,dtype=np.float32)[:,None]/10
        handle['force']=np.arange(24,dtype=np.float32).reshape(6,4)
        handle['time']=np.arange(6,dtype=np.float64)[:,None]/10+148
        handle['x']=np.linspace(8,25,256,dtype=np.float32)
        handle['y']=np.linspace(4,11,128,dtype=np.float32)
    code='''import importlib.util,sys,numpy as np
from pathlib import Path
adapter=Path(sys.argv[1]);data=Path(sys.argv[2])
s=importlib.util.spec_from_file_location("official_actual",adapter);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
fields,static=m.read_trajectory(data)
assert tuple(fields["state"].shape)==(6,3,128,256)
assert fields["omega"].dtype.__str__()=="torch.float32"
assert np.array_equal(fields["force"].numpy(),np.arange(24,dtype=np.float32).reshape(6,4))
assert static["x"].shape==(256,) and static["y"].shape==(128,)
print("ACTUAL_OFFICIAL_HDF5READER_SYNTHETIC_PASS")'''
    done=subprocess.run([str(REPO/'.venv-curator-py312/bin/python'),'-c',code,
      str(REPO/'scripts/official_evaluation_input.py'),str(path)],check=True,text=True,capture_output=True)
    assert done.stdout.strip()=='ACTUAL_OFFICIAL_HDF5READER_SYNTHETIC_PASS'


def test_dry_run_does_not_create_output_or_touch_payload(tmp_path):
    spec=tmp_path/'spec.json';spec.write_text('{}')
    digest=driver.sha(spec);output=tmp_path/'never-created'
    done=subprocess.run([str(REPO/'.venv-curator-py312/bin/python'),
      str(STAGE/'scripts/convert_projected_policy_h1_h5_frames.py'),
      '--spec',str(spec),'--spec-sha256',digest,'--output',str(output)],
      check=True,text=True,capture_output=True)
    assert json.loads(done.stdout)['status']=='PREPARATION_ONLY_NOT_EXECUTED'
    assert not output.exists()


def test_series_mapping_binds_all_48_unique_times(tmp_path):
    root=tmp_path/'case/VTK_replay';root.mkdir(parents=True)
    rows=[]
    for index in range(48):
        name=f'case_{index:05d}';folder=root/name;folder.mkdir()
        (folder/'internal.vtu').write_bytes(b'vtu'+bytes([index]))
        rows.append({'name':name+'.vtm','time':148+.1*index})
    (root/'case.vtm.series').write_text(json.dumps({'files':rows}))
    mapping=driver.series_mapping(tmp_path/'case')
    assert len(mapping)==48 and mapping[148.0].name=='internal.vtu'


def test_sample_view_copies_identical_bytes_without_hardlink(tmp_path,monkeypatch):
    sources=[];mapping={}
    for index in range(6):
        path=tmp_path/f'source-{index}.vtu';path.write_bytes(b'VTU'+bytes([index]))
        sources.append(path);mapping[148.0+.1*index]=path
    monkeypatch.setattr(driver,'series_mapping',lambda case:mapping)
    monkeypatch.setattr(driver.os,'link',lambda *a,**k:(_ for _ in ()).throw(
        AssertionError('hardlink must not be used')))
    class Sampler:
        @staticmethod
        def sample_frame(view,output):
            index=int(view.name)
            copied=view/'frame/internal.vtu'
            assert copied.read_bytes()==sources[index].read_bytes()
            np.savez(output,state=np.zeros((3,128,256),np.float32),
                     mask=np.ones((1,128,256),np.uint8),time=np.asarray([148+.1*index]),
                     x=np.linspace(8,25,256,dtype=np.float32),
                     y=np.linspace(4,11,128,dtype=np.float32))
    selection={'records':[{'branch':'mpc','start_index':0,'frames':[
      {'time':148+.1*i,'global_index':i} for i in range(6)]}]}
    packets=driver.sample_branch(tmp_path/'unused','mpc',selection,Sampler,tmp_path/'out',lambda:None)
    assert len(packets[0])==6
    for index,path in enumerate(sources):
        copied=tmp_path/'out/sample_views/mpc'/str(index)/'frame/internal.vtu'
        assert driver.sha(copied)==driver.sha(path)
        assert copied.stat().st_ino!=path.stat().st_ino


def test_bound_paths_reject_escape(tmp_path):
    outside=tmp_path.parent/'outside';outside.write_text('x')
    try:
        try:driver.checked(tmp_path,{'path':'../outside','sha256':driver.sha(outside)})
        except RuntimeError as error:assert 'escape' in str(error)
        else:raise AssertionError('escape accepted')
    finally:
        outside.unlink()


def test_recovered_foreign_container_is_never_adopted_for_cleanup(monkeypatch,tmp_path):
    removed=[]
    candidate='c'*64
    lists=iter(['',candidate+'\n'])
    def output(command,**kwargs):
        if command[:3]==['docker','container','ls']:
            return next(lists)
        if command[:2]==['docker','inspect']:
            return json.dumps([{'Name':'/projected-h1h5-convert-mpc-20261006',
              'Image':driver.EXPECTED_IMAGE,'Config':{'Cmd':['foreign']},
              'Mounts':[{'Source':str(tmp_path),'Destination':'/case','RW':True}]}])
        if command[:2]==['docker','create']:
            raise subprocess.TimeoutExpired(command,30)
        raise AssertionError(command)
    monkeypatch.setattr(driver.subprocess,'check_output',output)
    monkeypatch.setattr(driver.subprocess,'run',lambda command,**kwargs:removed.append(command))
    with pytest.raises(RuntimeError,match='owned container identity'):
        driver.export_branch(tmp_path,'mpc',driver.EXPECTED_IMAGE,tmp_path/'out',1e99)
    assert removed==[]


def test_final_spec_and_outer_unit_contract(tmp_path,monkeypatch):
    monkeypatch.setattr(driver,'ROOT',tmp_path);(tmp_path/'artifacts').mkdir()
    spec={'status':driver.EXPECTED_STATUS,'execution_authorized':True,
      'starts':[0,100,200,300,400,500,600,700],'horizon':5,'frames':96,
      'deadlines':{'conversion_seconds':1200,'future_inference_seconds':600},
      'resources':{'startup_available_gib':50,'runtime_available_gib':22,
        'supervisor_memory_gib':12,'supervisor_swap_gib':0},
      'image':driver.EXPECTED_IMAGE,'python':'.venv-curator-py312/bin/python',
      'runtime_versions':{'torch':'2.14.1','numpy':'2.5.3','pyvista':'0.49.0',
        'physicsnemo-curator':'0.1.0','nvidia-physicsnemo':'2.2.2'},
      'selected_source_inventory':{'case_mpc/148/U':{'size':1,'sha256':'a'*64}},
      'output':'artifacts/exclusive','unit':'fixed.service',
      'required_systemd':{'MemoryMax':str(12*2**30),'MemorySwapMax':'0',
        'CPUQuotaPerSecUSec':'1s','TasksMax':'64','RuntimeMaxUSec':'20min'}}
    driver.validate_spec(spec,tmp_path/'artifacts/exclusive')
    unit='\n'.join([f'MainPID={os.getpid()}',f'MemoryMax={12*2**30}',
      'MemorySwapMax=0','CPUQuotaPerSecUSec=1s','TasksMax=64','RuntimeMaxUSec=20min'])
    monkeypatch.setattr(driver.subprocess,'check_output',lambda *a,**k:unit)
    driver.verify_outer_unit(spec)
    spec['selected_source_inventory']={'../escape':{'size':1,'sha256':'a'*64}}
    with pytest.raises(RuntimeError,match='inventory escape'):
        driver.validate_spec(spec,tmp_path/'artifacts/exclusive')
