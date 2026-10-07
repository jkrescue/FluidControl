"""Read-only compact-label audit; no flow fields, models, or conversion."""
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
from physicsnemo.datapipes.readers.hdf5 import HDF5Reader

ROOT = Path('/workspace/fluid_control')
OUT = ROOT/'artifacts/p064_force_component_sidecars_20261007'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest_path = OUT/'manifest.json'
assert sha(manifest_path) == 'f8b2f6564e761d360c6a0101c6d6d036f49f7ce9643ecb7755b86cd43e6f9c3a'
m = json.loads(manifest_path.read_text())
approval = ROOT/'docs/P064_FORCE_COMPONENT_SIDECARS_APPROVAL_20261007.json'
assert sha(approval) == m['spec_sha256'] == '06ae2e41c8e22153408bc23e3c0e1f2f1dbbbb5b244ceb40638f3f5de3494ff0'
spec = json.loads(approval.read_text())
for k in ('worker', 'helper', 'supervisor'):
    assert sha(Path(spec[k]['path'])) == spec[k]['sha256']
families = {'base20': (20,16020), 'train8': (8,1608), 'train16': (16,2064), 'controlled_b00': (1,801)}
counts = {k:[0,0] for k in families}
values = {k:{c:[] for c in ('pressure','viscous')} for k in families}
max_sum = max_time = max_hdf = 0.
reader_samples = source_samples = 0
for row in m['rows']:
    p = Path(row['sidecar_path']); assert p.is_relative_to(OUT)
    assert sha(p) == row['sidecar_sha256']
    with h5py.File(p,'r') as f:
        data = {k:f[k][:] for k in ('time','hdf_total','raw_total','pressure','viscous')}
        assert f.attrs['split'] == 'train'
        for k in ('family','case','hdf_path','hdf_sha256','raw_root'):
            assert f.attrs[k] == row[k]
    n = row['frames']; assert data['time'].shape == (n,1)
    for k in ('hdf_total','raw_total','pressure','viscous'):
        assert data[k].shape == (n,4) and np.isfinite(data[k]).all()
    assert np.isfinite(data['time']).all() and np.all(np.diff(data['time'][:,0]) > 0)
    assert len(row['property_inventory']) == n
    for i,item in enumerate(row['property_inventory']):
        assert item['index'] == i and item['time'] == float(data['time'][i,0])
        rawp = Path(item['property_path']); assert rawp.is_relative_to(Path(row['raw_root']))
        dt = abs(float(rawp.parent.parent.parent.name)-item['time'])
        assert dt <= 2e-5; max_time = max(max_time,dt)
    err = float(abs(data['pressure']+data['viscous']-data['raw_total']).max())
    assert err <= 1e-10; max_sum = max(max_sum,err)
    with h5py.File(row['hdf_path'],'r') as f:
        assert np.array_equal(data['time'],f['time'][:])
        assert np.array_equal(data['hdf_total'],f['force'][:])
    max_hdf = max(max_hdf,float(abs(data['hdf_total']-data['raw_total']).max()))
    reader = HDF5Reader(p,fields=list(data))
    try:
        assert len(reader) == n
        for i in (0,n-1):
            sample,_ = reader[i]
            for k in data: assert np.array_equal(sample[k].numpy(),data[k][i])
            reader_samples += 1
            item = row['property_inventory'][i]; path = Path(item['property_path'])
            assert sha(path) == item['property_sha256']
            text = path.read_text()
            for body,offset in (('forceFront',0),('forceRear',2)):
                start = re.search(r'(?m)^\s*'+body+r'\s*$',text).end()
                opening=text.index('{',start); depth=0
                for end in range(opening,len(text)):
                    depth += (text[end]=='{')-(text[end]=='}')
                    if depth==0: break
                block=text[opening:end+1]
                for j,direction in enumerate(('Cd','Cl')):
                    for key,suffix in (('raw_total',''),('pressure','Pressure'),('viscous','Viscous')):
                        match=re.findall(r'(?m)^\s*'+direction+suffix+r'\s+([-+0-9.eE]+)\s*;',block)
                        assert len(match)==1 and float(match[0])==data[key][i,offset+j]
            source_samples += 1
    finally: reader.close()
    counts[row['family']][0]+=1; counts[row['family']][1]+=n
    for k in values[row['family']]:values[row['family']][k].append(data[k])
assert {k:tuple(v) for k,v in counts.items()} == families
assert len(m['rows'])==45 and sum(v[1] for v in counts.values())==20493
def check_moments(arr, saved):
    assert saved['count']==len(arr)
    np.testing.assert_allclose(arr.mean(0),saved['mean'],rtol=0,atol=1e-13)
    np.testing.assert_allclose(arr.std(0),saved['std_population'],rtol=0,atol=1e-13)
for family in families:
    for k in ('pressure','viscous'):
        check_moments(np.concatenate(values[family][k]),m['family_moments'][family][k])
for k in ('pressure','viscous'):
    check_moments(np.concatenate([a for f in families for a in values[f][k]]),m['all_train_'+k+'_moments'])
print(json.dumps({'status':'INDEPENDENT_COMPACT_SIDECAR_AUDIT_PASS','manifest_sha256':sha(manifest_path),
    'family_counts':counts,'frames':20493,'official_reader_samples':reader_samples,
    'raw_boundary_samples':source_samples,'max_component_sum_abs':max_sum,'max_time_abs':max_time,
    'max_hdf_total_minus_raw_total_abs':max_hdf,'original_label_arrays_exact':True,
    'field_payload_read':False,'large_hdf_rehashed':False},indent=2))
