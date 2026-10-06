"""Thin CPU conversion profile for opened b01/b03 controlled development frames."""
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

ROOT=Path('/workspace/fluid_control')
BASE=ROOT/'artifacts/projected_policy_h1_h5_conversion_source_r2_20261006_immutable/scripts/convert_projected_policy_h1_h5_frames.py'
BASE_SHA='304fece8b7c205dbd8182b9cdae24701fc917b102884406b54734f7e0d0cad68'
STATUS='DEVELOPMENT_PHASE_CONVERSION_EXECUTION_APPROVED'
VERSIONS={'torch':'2.14.1','numpy':'2.5.3','pyvista':'0.49.0','physicsnemo-curator':'0.1.0','nvidia-physicsnemo':'2.2.2'}
RESOURCES={'memory_gib':12,'swap_gib':0,'cpu':1,'startup_available_gib':50,'runtime_available_gib':22,'reserve_gib':20,'deadline_seconds':900}

def require(value,message):
    if not value:raise ValueError(message)

def base_module():
    require(hashlib.sha256(BASE.read_bytes()).hexdigest()==BASE_SHA,'R2 source')
    spec=importlib.util.spec_from_file_location('reviewed_phase_converter',BASE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def validate(spec):
    require(spec['status']==STATUS and spec['execution_authorized'] is True,'not approved')
    require(spec['phases']==['b01','b03'] and set(spec['phase_inputs'])=={'b01','b03'},'fixed phases')
    require(spec['frames']==96 and spec['trajectories']==16 and spec['split']=='development_already_opened','fixed scope')
    require(spec['resources']==RESOURCES and spec['runtime_versions']==VERSIONS,'resources/runtime')
    require(spec['python']=='.venv-curator-py312/bin/python','python')
    for phase,item in spec['phase_inputs'].items():
        require(item['source_root']==f'artifacts/exploratory_projected_32768_ppo_{phase}_long_cfd_20261006','source root')
        require(item['progress']['path']==item['source_root']+'/progress.json','progress path')
        require(item['result']['path']==item['source_root']+'/result.json','result path')
        require(all(k.startswith('case_mpc/') and '..' not in Path(k).parts for k in item['selected_source_inventory']),'controlled only')

def verify_unit(spec):
    expected={'MemoryMax':str(12*2**30),'MemorySwapMax':'0','CPUQuotaPerSecUSec':'1s','TasksMax':'64','RuntimeMaxUSec':'15min','KillMode':'control-group'}
    raw=subprocess.check_output(['systemctl','--user','show',spec['unit'],*[v for k in ('MainPID',*expected) for v in ('-p',k)]],text=True,timeout=10)
    actual=dict(x.split('=',1) for x in raw.splitlines() if '=' in x)
    require(actual.get('MainPID')==str(os.getpid()) and all(actual.get(k)==v for k,v in expected.items()),'actual unit')

def execute(spec,digest):
    validate(spec);base=base_module();verify_unit(spec)
    require(base.memory_available()>=50*2**30,'startup Available')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('NVIDIA_VISIBLE_DEVICES')=='void','CPU only')
    require(Path(sys.executable).resolve()==(ROOT/spec['python']).resolve(),'runtime python')
    require({k:importlib.metadata.version(k) for k in VERSIONS}==VERSIONS,'installed versions')
    require(base.checked(ROOT,spec['driver']).resolve()==Path(__file__).resolve(),'driver')
    paths={k:base.checked(ROOT,spec[k]) for k in ('selection_adapter','core','sampler','official_reader_adapter','official_reader_source')}
    selector=base.load_module(paths['selection_adapter'],'development_selection')
    core=base.load_module(paths['core'],'development_core')
    sampler=base.load_module(paths['sampler'],'development_sampler')
    reader=base.load_module(paths['official_reader_adapter'],'development_reader')
    require(spec['image']==base.EXPECTED_IMAGE and subprocess.check_output(['docker','image','inspect',spec['image'],'--format','{{.Id}}'],text=True,timeout=10).strip()==spec['image'],'image')
    output=base.output_path(spec)
    require(shutil.disk_usage(output.parent).free>=10*2**30,'disk headroom')
    selected={}
    for phase,item in spec['phase_inputs'].items():
        path=base.checked(ROOT,item['progress']);base.checked(ROOT,item['result'])
        selected[phase]=selector.make_selection(json.loads(path.read_text()),ROOT/item['source_root'],phase)
    output.mkdir(exist_ok=False);deadline=time.monotonic()+900
    guard=lambda:base.resource_row(output,'conversion',deadline)
    monitor=base.ContinuousGuard(output,deadline);monitor.start()
    receipts=[]
    try:
        base.atomic_json(output/'selection.json',dict(source_spec_sha256=digest,phases=selected))
        for phase in spec['phases']:
            item=spec['phase_inputs'][phase];source=ROOT/item['source_root'];selection=selected[phase]
            child=output/phase;child.mkdir(exist_ok=False)
            inventory=base.selected_inventory(source,selection,guard)
            require(inventory==item['selected_source_inventory'],'approved inventory')
            base.atomic_json(child/'source_inventory.json',inventory)
            copied=child/'selected_cases';base.copy_selected(source,copied,selection,inventory,guard)
            case=copied/'case_mpc'
            base.export_branch(case,'development-'+phase,base.EXPECTED_IMAGE,child,deadline)
            packets=base.sample_branch(case,'mpc',selection,sampler,child,monitor.check)
            for record in selection['records']:
                hdf=child/'hdf'/f"{phase}_start_{record['start_index']:04d}.h5"
                receipt=core.pack_and_verify_mini_hdf(record,packets[record['start_index']],hdf,reader.read_trajectory)
                receipts.append(dict(receipt,phase=phase,start_index=record['start_index'],split='development_already_opened'))
                guard();monitor.check()
                base.atomic_json(output/'progress.json',dict(status='CONVERTING_NOT_TRAINING',completed_trajectories=len(receipts),expected_trajectories=16))
            require(base.selected_inventory(source,selection,guard)==inventory,'source changed')
        require(len(receipts)==16 and len({(x['mask_sha256'],x['x_sha256'],x['y_sha256']) for x in receipts})==1,'all16 common grid')
        monitor.check()
        base.atomic_json(output/'result.json',dict(status='DEVELOPMENT_PHASE_CONVERSION_COMPLETE_NOT_ADMISSION',source_spec_sha256=digest,selection_sha256=base.sha(output/'selection.json'),frames=96,trajectories=16,endpoints=80,hdf=receipts,split='development_already_opened',source_unchanged=True,owned_containers_cleaned=True,scratch_retained=True,model_loaded=False,cfd_executed=False,optimizer_steps=0,scientific_admission=False))
    finally:monitor.stop()

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--spec-sha256',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    raw=a.spec.read_bytes();require(hashlib.sha256(raw).hexdigest()==a.spec_sha256,'spec hash');spec=json.loads(raw)
    if not a.execute:print('PREPARATION_ONLY');return
    execute(spec,a.spec_sha256)

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(RuntimeError('SIGTERM')))
    main()
