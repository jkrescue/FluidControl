"""One approved saved-CFD-only t228 comparison; no solver advance or model."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path('/workspace/fluid_control')
OUTPUT = ROOT/'artifacts/p064_real_cfd_t228_comparison_20261007'
BASE = ROOT/'scripts/convert_projected_policy_h1_h5_frames.py'
SAMPLER = ROOT/'scripts/persistent_curator_frame.py'
PINS = {BASE: '304fece8b7c205dbd8182b9cdae24701fc917b102884406b54734f7e0d0cad68', SAMPLER: '6c1ae12c0b7382347547b2049a3f82d9f18cc9b5c5c4ff71a5b9d5895d067416'}
OLD = ROOT/'artifacts/p064_b_projected_ppo_long_cfd_20261006'
NEW = ROOT/'artifacts/p064_b_seed20261007_projected_ppo_long_cfd_20261006'
CASES = {'zero': OLD/'case_zero', 'trained_seed20261006': OLD/'case_mpc', 'trained_seed20261007': NEW/'case_mpc'}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def inventory(case):
    paths = [case/'228/U', case/'228/p']
    for directory in ('constant','system'):
        for p in (case/directory).rglob('*'):
            assert not p.is_symlink(), p
            if p.is_file(): paths.append(p)
    result = {}
    for p in paths:
        assert p.is_file() and not p.is_symlink()
        assert all(not x.is_symlink() for x in p.parents)
        result[str(p.relative_to(case))] = {'sha256':sha(p),'size':p.stat().st_size}
    return result

def masked_quantity(packet, kind):
    import numpy as np
    mask = packet['mask'][0].astype(bool)
    assert packet['state'].shape == (3,128,256) and mask.shape == (128,256)
    assert np.isfinite(packet['state']).all() and mask.any()
    state = packet['state'].astype(np.float64)
    value = np.hypot(state[0],state[1]) if kind == 'speed' else state[2]
    return np.where(mask,value,np.nan)

def common_range(fields):
    import numpy as np
    low = min(float(np.nanmin(f)) for f in fields)
    high = max(float(np.nanmax(f)) for f in fields)
    assert np.isfinite([low,high]).all() and high > low
    return low,high

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    for p,d in PINS.items(): assert sha(p)==d
    base=load(BASE,'reviewed_conversion'); assert base.memory_available()>=50*2**30
    cgroup=Path('/sys/fs/cgroup')/next(l.split('::',1)[1] for l in Path('/proc/self/cgroup').read_text().splitlines() if l.startswith('0::')).lstrip('/')
    assert 0<int((cgroup/'memory.max').read_text())<=4*2**30
    assert int((cgroup/'memory.swap.max').read_text())==0
    assert not OUTPUT.exists(); OUTPUT.mkdir()
    start=time.monotonic(); deadline=start+280
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(RuntimeError('TERM')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(RuntimeError('deadline')))
    signal.alarm(280)
    guard=base.ContinuousGuard(OUTPUT,deadline); guard.start()
    try:
        inventories={name:inventory(case) for name,case in CASES.items()}
        zero_duplicate=inventory(NEW/'case_zero')
        for f in ('228/U','228/p'): assert inventories['zero'][f]==zero_duplicate[f]
        base.atomic_json(OUTPUT/'source_inventory.json',{'cases':inventories,'duplicate_zero':zero_duplicate})
        image=base.EXPECTED_IMAGE
        assert subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True,timeout=10).strip()==image
        sampler=load(SAMPLER,'reviewed_persistent_curator')
        import numpy as np
        packets={}
        for name,case in CASES.items():
            scratch=OUTPUT/'scratch'/name; scratch.mkdir(parents=True)
            for relative,item in inventories[name].items():
                dest=scratch/relative; dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(case/relative,dest); assert sha(dest)==item['sha256']; guard.check()
            base.export_branch(scratch,'physical228-'+name,image,OUTPUT,deadline)
            series=json.loads((scratch/'VTK_replay/case.vtm.series').read_text())['files']
            assert len(series)==1 and abs(float(series[0]['time'])-228)<1e-8
            vtk=scratch/'VTK_replay'/Path(series[0]['name']).stem/'internal.vtu'
            view=OUTPUT/'views'/name/'frame'; view.mkdir(parents=True)
            shutil.copy2(vtk,view/'internal.vtu'); assert sha(vtk)==sha(view/'internal.vtu')
            target=OUTPUT/(name+'.npz'); sampler.sample_frame(view.parent,target)
            with np.load(target,allow_pickle=False) as data: packets[name]={k:data[k] for k in data.files}
            assert abs(float(packets[name]['time'][0])-228)<1e-6
            guard.check()
        reference=packets['zero']
        for p in packets.values():
            for k in ('mask','x','y'): assert np.array_equal(p[k],reference[k]),k
            assert abs(float(p['state'][2,p['mask'][0].astype(bool)].mean(dtype=np.float64)))<1e-6
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(15,6),constrained_layout=True)
        ranges={}
        titles=['Paired zero','Trained seed20261006 (prior benefit)','Trained seed20261007 (prior drag failure)']
        for row,kind in enumerate(('speed','pressure')):
            fields=[masked_quantity(p,kind) for p in packets.values()]; low,high=common_range(fields);ranges[kind]=[low,high]
            for col,(field,title) in enumerate(zip(fields,titles)):
                im=axes[row,col].imshow(field,origin='lower',extent=[8,25,4,11],aspect='equal',cmap='magma' if kind=='speed' else 'coolwarm',vmin=low,vmax=high)
                axes[row,col].set_title(title);axes[row,col].set_xlabel('x (solver coordinates)');axes[row,col].set_ylabel('y')
            fig.colorbar(im,ax=list(axes[row]),shrink=.8,label='|U| (solver units)' if kind=='speed' else "p′ (same ROI per-frame mean removed; solver units)")
        fig.suptitle('Saved real OpenFOAM fields at t=228 — identical grid and shared row color scale\nInstantaneous morphology, NOT a drag-window measurement; no FNO/prediction')
        png=OUTPUT/'real_cfd_t228_comparison.png';fig.savefig(png,dpi=150);plt.close(fig)
        for name,case in CASES.items(): assert inventory(case)==inventories[name]
        assert inventory(NEW/'case_zero')==zero_duplicate
        guard.check()
        base.atomic_json(OUTPUT/'figure_manifest.json',{'status':'REAL_SAVED_CFD_FIELD_VISUALIZATION_COMPLETE','time':228,'source_unchanged':True,'zero_fields_byte_identical':True,'source_inventory_sha256':sha(OUTPUT/'source_inventory.json'),'sources':{str(p):d for p,d in PINS.items()},'script_sha256':sha(__file__),'image':image,'grid':[128,256],'extent':[8,25,4,11],'pressure_gauge':'same ROI per-frame mean removed','ranges':ranges,'versions':{k:importlib.metadata.version(k) for k in ('numpy','torch','matplotlib','physicsnemo-curator','nvidia-physicsnemo')},'png_sha256':sha(png),'packets':{n:sha(OUTPUT/(n+'.npz')) for n in packets},'wall_seconds':time.monotonic()-start,'new_cfd_steps':0,'model_inference':False,'scratch_retained':True})
    finally:
        guard.stop();signal.alarm(0)

if __name__=='__main__': main()
