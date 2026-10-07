"""Read-only B04 first/last H100 check; project adapter + official HDF5Reader.

Run only after a separately authorized conversion completes. No model/optimizer.
The callable does not build a view, refit normalization, or write input files.
"""
import hashlib
import inspect
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from physicsnemo.datapipes.readers.hdf5 import HDF5Reader
from fluid_control.tandem_datapipe import TandemRolloutDataset

NORM_SHA = 'f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1'
FIELDS = ('state', 'mask', 'omega', 'force', 'time')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def exact(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.array_equal(a, b):
        raise ValueError(f'{label}: shape/value/finite mismatch')

def verify_b04_h100(root, hdf, expected_grid=(128, 256)):
    root, hdf = Path(root).resolve(), Path(hdf).resolve()
    norm, manifest = root/'normalization.json', root/'manifest.json'
    if sorted((root/'train').glob('*.h5')) != [hdf] or hdf.parent != root/'train':
        raise ValueError('requires exclusive single train HDF view')
    if any((root/s).exists() for s in ('validation','dev','test','frozen_test')):
        raise ValueError('train-only view required')
    if sha(norm) != NORM_SHA or json.loads(manifest.read_text())['max_abs_omega'] != .75:
        raise ValueError('original normalization/action scale required')
    before = {str(p): sha(p) for p in (hdf,norm,manifest)}
    ds = TandemRolloutDataset(root,'train',100,stride=1,num_workers=0,force_indices=(0,1,2,3))
    reader = HDF5Reader(hdf,fields=list(FIELDS))
    records=[]
    try:
        if len(reader)!=801 or len(ds)!=701 or ds.index[0]!=(0,0) or ds.index[-1]!=(0,700):
            raise ValueError('801 frames / 701 H100 windows required')
        with h5py.File(hdf,'r') as raw:
            if raw.attrs.get('split')!='train' or raw.attrs.get('case')!=hdf.stem:
                raise ValueError('HDF train/case identity')
            height,width=expected_grid
            shapes={'state':(801,3,height,width),'mask':(801,1,height,width),'omega':(801,1),'force':(801,4),'time':(801,1)}
            if any(raw[k].shape!=shape for k,shape in shapes.items()):raise ValueError('HDF shape contract')
            for start in (0,700):
                # Only these two 101-frame slices are decoded, not the complete trajectory.
                fields={k:raw[k][start:start+101] for k in FIELDS}
                clock=(120+np.arange(start,start+101)*.1).astype(raw['time'].dtype).reshape(101,1)
                exact(fields['time'],clock,'physical time')
                for offset in range(101):
                    values,metadata=reader[start+offset]
                    for k in FIELDS:exact(values[k].cpu().numpy(),fields[k][offset],f'official/{k}/{start+offset}')
                sample,metadata=ds[start]
                if metadata != dict(case=hdf.stem,step=start,rollout_steps=100,split='train'):
                    raise ValueError('adapter metadata')
                mask=torch.from_numpy(fields['mask'][0]).float()
                # Match the actual project adapter's FP32 arithmetic/order exactly.
                states=torch.from_numpy(fields['state']).float()
                normalized=(states-ds.state_mean[None])/ds.state_std[None]
                normalized*=mask[None]
                omega=torch.from_numpy(fields['omega']).float()/.75
                force=(torch.from_numpy(fields['force'][1:]).float()-ds.force_mean[None])/ds.force_std[None]
                expected=dict(state=normalized[0],target_state=normalized[1:],mask=mask,omega=omega,target_force=force,time=torch.from_numpy(fields['time'][0]).float())
                for k,value in expected.items():exact(sample[k].cpu().numpy(),value.numpy(),f'adapter/{k}/{start}')
                records.append(dict(start=start,end=start+100,physical_times=fields['time'].reshape(-1).tolist(),physical_omega=fields['omega'].reshape(-1).tolist(),physical_target_force=fields['force'][1:].tolist(),sample_shapes={k:list(v.shape) for k,v in sample.items()},decoded_frame_indices=[start,start+100]))
    finally:
        reader.close();ds.close()
    if any(sha(Path(p))!=h for p,h in before.items()):raise ValueError('input bytes changed')
    sources={str(Path(inspect.getfile(cls)).resolve()):sha(inspect.getfile(cls)) for cls in (HDF5Reader,TandemRolloutDataset)}
    return dict(status='B04_H100_READER_ENDPOINTS_VERIFIED_NOT_TRAINING',inputs_sha256=before,source_sha256=sources,normalization_refit=False,windows=records,decoded_frames=202,model_forward=0,optimizer_steps=0,scope='first/last H100 only; no full-trajectory or physical-quality admission')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--hdf',type=Path,required=True)
    args=p.parse_args();print(json.dumps(verify_b04_h100(args.root,args.hdf),sort_keys=True))
