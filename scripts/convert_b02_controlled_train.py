"""Project adapter: one entire saved b02 canonical-PPO trajectory, train only.

Reuses reviewed R2 export/lifecycle, Curator sampling and official reader.
No solver advance, inference, normalization fit, or dataset split search.
"""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path('/workspace/fluid_control')
BASE_PATH = ROOT/'artifacts/projected_policy_h1_h5_conversion_source_r2_20261006_immutable/scripts/convert_projected_policy_h1_h5_frames.py'
BASE_SHA = '304fece8b7c205dbd8182b9cdae24701fc917b102884406b54734f7e0d0cad68'
SOURCE = ROOT/'artifacts/p064_b_symmetry_canonical_ppo_b02_train_acquisition_20261007'
NORM_SHA = 'f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1'
STATUS = 'B02_CANONICAL_CONTROLLED_TRAIN_CONVERSION_EXECUTION_APPROVED'
SECONDS = 3600

def require(value, message):
    if not value: raise ValueError(message)

def load_base():
    require(hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()==BASE_SHA,'R2 source')
    spec=importlib.util.spec_from_file_location('reviewed_r2_conversion',BASE_PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def selection(document, core):
    rows=core.validate_progress(document)
    frames=[dict(global_index=i,time=106+.1*i,
                 files={k:f'case_mpc/{core.time_name(106+.1*i)}/{k}' for k in ('U','p')}) for i in range(801)]
    omega=np.asarray([0.]+[r['applied_omega'] for r in rows],dtype=np.float32)[:,None]
    force=np.asarray([rows[0]['input_observation'][64:68]]+[r['output_observation'][64:68] for r in rows],dtype=np.float32)
    return {'records':[{'branch':'mpc','frames':frames}]},omega,force

def validate_source_result(document):
    require(document.get('status')=='P064_B_SYMMETRY_CANONICAL_32768_PPO_LONG_CFD_COMPLETE_NOT_ADMISSION','source result status')
    require(document.get('cycles')==800 and document.get('owned_containers_cleaned') is True and document.get('source_restart_unchanged') is True,'source result terminal')

def validate_source_approval(document):
    require(document.get('execution_authorized') is True and document.get('steps')==800,'source approval execution')
    require(document.get('start_time')==106 and document.get('end_time')==186,'source approval time')
    require(document.get('output')==str(SOURCE.relative_to(ROOT)),'source approval output')

def batches(frames):
    return [frames[i:i+48] for i in range(0,len(frames),48)]

def series(case, frames):
    doc=json.loads((case/'VTK_replay/case.vtm.series').read_text())
    result={}
    for frame in frames:
        matches=[r for r in doc['files'] if abs(float(r['time'])-frame['time'])<=1e-8]
        require(len(matches)==1,'VTK time uniqueness')
        path=case/'VTK_replay'/Path(matches[0]['name']).stem/'internal.vtu'
        require(path.is_file() and not path.is_symlink(),'VTK member')
        result[frame['global_index']]=path
    require(len(doc['files'])==len(frames),'extra VTK times')
    return result

def write_packet(handle, packet, index, core, anchor):
    state,mask=core.validate_packet(packet,expected_time=106+.1*index,expected_mask=anchor)
    if anchor is None:
        handle.create_dataset('x',data=packet['x']);handle.create_dataset('y',data=packet['y'])
        anchor=mask.copy()
    else:
        require(np.array_equal(handle['x'][:],packet['x']) and np.array_equal(handle['y'][:],packet['y']),'grid')
    handle['state'][index]=state;handle['mask'][index]=mask
    handle['time'][index]=packet['time']
    return anchor

def progress(base, output, written, last, batch, completed=False):
    base.atomic_json(output/'progress.json',dict(
        status='COMPLETE_NOT_TRAINING' if completed else 'CONVERTING_NOT_TRAINING',
        written_frames=len(written),expected_frames=801,last_global_index=last,
        current_batch=batch,official_reader_verified=completed))

def validate(spec, output):
    require(spec['status']==STATUS and spec['execution_authorized'] is True,'not authorized')
    require(spec['frames']==801 and spec['split']=='train' and spec['branch']=='mpc','whole controlled train only')
    require(spec['normalization']['sha256']==NORM_SHA and spec['normalization_refit'] is False,'normalization')
    require(spec['deadline_seconds']==SECONDS and spec['source_root']==str(SOURCE.relative_to(ROOT)),'scope')
    require(spec['scratch_retained'] is True and spec['disk_budget_gib']==10,'scratch disk contract')
    require(spec['resources']=={'memory_gib':12,'swap_gib':0,'cpu':1,'startup_available_gib':50,'runtime_available_gib':22},'resources')
    require(spec['python']=='.venv-curator-py312/bin/python','python')
    require(spec['runtime_versions']=={'torch':'2.14.1','numpy':'2.5.3','pyvista':'0.49.0','physicsnemo-curator':'0.1.0','nvidia-physicsnemo':'2.2.2'},'versions')
    require(output==ROOT/spec['output'] and ROOT in output.parents,'output')
    require(not output.exists() and not output.is_symlink(),'exclusive output')
    require(not Path(spec['output']).is_absolute() and '..' not in Path(spec['output']).parts,'output escape')
    for p in (output.parent,*output.parent.parents):
        require(not p.is_symlink(),'output ancestor symlink')
        if p==ROOT:break
    expected={f'case_mpc/{106+i*.1:.10g}/{k}' for i in range(801) for k in ('U','p')}
    inventory=spec['selected_source_inventory']
    require(expected<=set(inventory),'all1602 fields')
    require(all(p in expected or p.startswith(('case_mpc/constant/','case_mpc/system/')) for p in inventory),'no zero or extra fields')
    require(spec['reused_packets']=={},'b02 has no reusable packets')

def verify_unit(spec):
    expected={'MemoryMax':str(12*2**30),'MemorySwapMax':'0','CPUQuotaPerSecUSec':'1s','TasksMax':'64','RuntimeMaxUSec':'1h'}
    raw=subprocess.check_output(['systemctl','--user','show',spec['unit'],*[a for k in ('MainPID',*expected) for a in ('-p',k)]],text=True,timeout=10)
    actual=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
    require(actual.get('MainPID')==str(os.getpid()) and all(actual.get(k)==v for k,v in expected.items()),'actual outer unit')

def execute(spec, digest, output):
    import h5py
    base=load_base();validate(spec,output);verify_unit(spec)
    require(base.memory_available()>=50*2**30,'startup memory')
    require(shutil.disk_usage(output.parent).free>=20*2**30,'20GiB output headroom')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('NVIDIA_VISIBLE_DEVICES')=='void','CPU only')
    require(Path(sys.executable).resolve()==(ROOT/spec['python']).resolve(),'actual python')
    require({k:importlib.metadata.version(k) for k in spec['runtime_versions']}==spec['runtime_versions'],'runtime')
    require(base.checked(ROOT,spec['driver']).resolve()==Path(__file__).resolve(),'driver identity')
    paths={k:base.checked(ROOT,spec[k]) for k in ('core','sampler','official_reader_adapter','official_reader_source','normalization','progress','source_result','source_approval')}
    core=base.load_module(paths['core'],'b02_reviewed_core')
    sampler=base.load_module(paths['sampler'],'b02_reviewed_sampler')
    official=base.load_module(paths['official_reader_adapter'],'b02_official_reader')
    validate_source_approval(json.loads(paths['source_approval'].read_text()))
    source_result=json.loads(paths['source_result'].read_text())
    validate_source_result(source_result)
    selected,omega,force=selection(json.loads(paths['progress'].read_text()),core)
    output.mkdir();deadline=time.monotonic()+SECONDS
    guard=lambda:base.resource_row(output,'conversion',deadline)
    monitor=base.ContinuousGuard(output,deadline);monitor.start()
    scratch=None
    try:
        inventory=base.selected_inventory(SOURCE,selected,guard)
        require(inventory==spec['selected_source_inventory'],'source inventory')
        base.atomic_json(output/'source_inventory.json',inventory)
        base.atomic_json(output/'selection.json',selected)
        hdf=output/'b02_canonical_ppo_train.h5';written=set();anchor=None
        progress(base,output,written,None,None)
        with h5py.File(hdf,'x') as handle:
            handle.attrs['split']='train';handle.attrs['case']='b02_canonical_ppo_801'
            handle.attrs['normalization_sha256']=NORM_SHA
            handle.attrs['source_spec_sha256']=digest
            handle.create_dataset('state',(801,3,128,256),dtype='f4',chunks=(1,3,128,256))
            handle.create_dataset('mask',(801,1,128,256),dtype='u1',chunks=(1,1,128,256))
            handle.create_dataset('time',(801,1),dtype='f8')
            handle.create_dataset('omega',data=omega);handle.create_dataset('force',data=force)
            for batchno,frames in enumerate(batches(selected['records'][0]['frames'])):
                scratch=output/f'batch_{batchno:02d}'
                batchsel={'records':[{'branch':'mpc','frames':frames}]}
                subset=base.selected_inventory(SOURCE,batchsel,guard)
                require(all(inventory.get(k)==v for k,v in subset.items()),'batch source')
                base.copy_selected(SOURCE,scratch,batchsel,subset,guard)
                case=scratch/'case_mpc'
                base.export_branch(case,f'b02train-{batchno:02d}',base.EXPECTED_IMAGE,output,deadline)
                vtk=series(case,frames)
                for frame in frames:
                    index=frame['global_index'];view=scratch/'sample_view';(view/'frame').mkdir(parents=True)
                    shutil.copy2(vtk[index],view/'frame/internal.vtu')
                    require(base.sha(vtk[index],guard)==base.sha(view/'frame/internal.vtu',guard),'VTU copy')
                    packetpath=scratch/'sample.npz';sampler.sample_frame(view,packetpath);guard()
                    with np.load(packetpath,allow_pickle=False) as packet:
                        anchor=write_packet(handle,dict(packet),index,core,anchor)
                    require(index not in written,'duplicate frame');written.add(index)
                    progress(base,output,written,index,batchno)
                    shutil.rmtree(view);packetpath.unlink()
                # foamToVTK creates root-owned trees. Retain all17 batch exports;
                # do not mutate permissions or attempt unprivileged recursive deletion.
                scratch=None
            require(written==set(range(801)),'all801 exactly once')
        fields,static=official.read_trajectory(hdf);guard()
        with h5py.File(hdf,'r') as handle:
            for key in official.FIELDS:
                require(tuple(fields[key].shape)==handle[key].shape,'reader shape')
                for i in range(801):
                    actual=fields[key][i].numpy();expected=handle[key][i]
                    require(actual.dtype==expected.dtype and np.array_equal(actual,expected),'reader exact fields')
                    guard()
            require(all(np.array_equal(static[k],handle[k][:]) for k in ('x','y')),'reader static')
        require(base.selected_inventory(SOURCE,selected,guard)==inventory,'original unchanged')
        monitor.check()
        base.atomic_json(output/'result.json',dict(status='B02_CONTROLLED_TRAIN_HDF_COMPLETE_NOT_TRAINING',frames=801,trajectories=1,split='train',reused_frames=0,newly_sampled_frames=801,scratch_retained=True,disk_budget_gib=10,label_provenance='actual CFD endpoint observation coefficients; not legacy interpolated-HDF labels',hdf={'path':str(hdf),'sha256':base.sha(hdf,guard)},source_spec_sha256=digest,source_inventory_sha256=base.sha(output/'source_inventory.json'),normalization_sha256=NORM_SHA,normalization_refit=False,official_reader_verified=True,source_unchanged=True,model_loaded=False,cfd_executed=False,optimizer_steps=0,scientific_admission=False,owned_containers_cleaned=True))
        progress(base,output,written,800,16,completed=True)
    finally:
        monitor.stop()
        # Failed scratch is retained for diagnosis; exporter owns container cleanup.

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--spec-sha256',required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args();raw=a.spec.read_bytes();require(hashlib.sha256(raw).hexdigest()==a.spec_sha256,'spec SHA')
    spec=json.loads(raw);output=ROOT/spec['output']
    if not a.execute:print('PREPARATION_ONLY');return
    execute(spec,a.spec_sha256,output)

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(RuntimeError('SIGTERM')))
    main()
