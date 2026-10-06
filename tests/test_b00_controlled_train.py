import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE if (HERE/'convert_b00_controlled_train.py').exists() else HERE.parent/'scripts'))
import convert_b00_controlled_train as m

class Core:
    @staticmethod
    def validate_progress(d):return d['rows']
    @staticmethod
    def time_name(t):return f'{t:.10g}'

def test_all801_clock_and_no_zero():
    rows=[{'applied_omega':i/1000,'input_observation':[0]*64+[1,2,3,4]+[0],
           'output_observation':[0]*64+[i,i+1,i+2,i+3]+[0]} for i in range(800)]
    sel,a,f=m.selection({'rows':rows},Core)
    frames=sel['records'][0]['frames']
    assert len(frames)==801 and frames[-1]['time']==228
    assert all(p.startswith('case_mpc/') for v in frames for p in v['files'].values())
    assert a.dtype==np.float32 and a[0,0]==0 and a[-1,0]==np.float32(.799)
    assert f[0].tolist()==[1,2,3,4] and f[-1].tolist()==[799,800,801,802]

def test_bounded_batches_exact753():
    frames=[{'global_index':i} for i in range(801)]
    reuse={i+j for i in range(0,800,100) for j in range(6)}
    batch=m.batches(frames,reuse)
    ids=[f['global_index'] for b in batch for f in b]
    assert len(ids)==753 and len(set(ids))==753 and max(map(len,batch))==48
    assert set(ids)|reuse==set(range(801)) and not set(ids)&reuse

def test_series_partial_batch_and_stale(tmp_path):
    import json
    root=tmp_path/'VTK_replay';(root/'x').mkdir(parents=True)
    (root/'x/internal.vtu').touch()
    p=root/'case.vtm.series';p.write_text(json.dumps({'files':[{'time':228,'name':'x.vtm'}]}))
    assert m.series(tmp_path,[{'time':228,'global_index':800}])[800].name=='internal.vtu'
    p.write_text(json.dumps({'files':[{'time':228,'name':'x.vtm'}]*2}))
    with pytest.raises(ValueError):m.series(tmp_path,[{'time':228,'global_index':800}])

def valid_spec():
    return dict(status=m.STATUS,execution_authorized=True,frames=801,split='train',branch='mpc',
        normalization={'sha256':m.NORM_SHA},normalization_refit=False,deadline_seconds=3600,scratch_retained=True,disk_budget_gib=10,
        source_root=str(m.SOURCE.relative_to(m.ROOT)),output='artifacts/test-exclusive-b00',
        resources={'memory_gib':12,'swap_gib':0,'cpu':1,'startup_available_gib':50,'runtime_available_gib':22},
        python='.venv-curator-py312/bin/python',runtime_versions={'torch':'2.14.1','numpy':'2.5.3','pyvista':'0.49.0','physicsnemo-curator':'0.1.0','nvidia-physicsnemo':'2.2.2'},
        selected_source_inventory={f'case_mpc/{148+i*.1:.10g}/{k}':{} for i in range(801) for k in ('U','p')},
        reused_packets={str(i+j):{} for i in range(0,800,100) for j in range(6)})

def test_contract():
    s=valid_spec();m.validate(s,m.ROOT/s['output'])

@pytest.mark.parametrize('key,value',[('split','validation'),('branch','zero'),('frames',800),('normalization_refit',True),('deadline_seconds',1200)])
def test_reject_contract(key,value):
    s=valid_spec();s[key]=value
    with pytest.raises(ValueError):m.validate(s,m.ROOT/s['output'])

def test_packet_roundtrip(tmp_path):
    import h5py
    class PacketCore:
        @staticmethod
        def validate_packet(p,expected_time,expected_mask):
            assert p['time'][0]==expected_time
            if expected_mask is not None:assert np.array_equal(expected_mask,p['mask'])
            return p['state'],p['mask']
    p={'state':np.zeros((3,2,2),np.float32),'mask':np.ones((1,2,2),np.uint8),'time':np.array([148.]),'x':np.array([1,2]),'y':np.array([3,4])}
    with h5py.File(tmp_path/'tiny.h5','x') as h:
        h.create_dataset('state',(1,3,2,2),dtype='f4');h.create_dataset('mask',(1,1,2,2),dtype='u1');h.create_dataset('time',(1,1),dtype='f8')
        m.write_packet(h,p,0,PacketCore,None)
        assert np.array_equal(h['state'][0],p['state']) and h['time'][0,0]==148

def test_reuse_links_receipt_arrays():
    import torch
    packet={'state':np.zeros((3,2,2),np.float32),'mask':np.ones((1,2,2),np.uint8),'time':np.array([148.]),'x':np.array([1]),'y':np.array([2])}
    fields={k:torch.from_numpy(packet[k][None]) for k in ('state','mask','time')}
    fields.update(omega=torch.tensor([[.1]]),force=torch.tensor([[1.,2.,3.,4.]]))
    static={k:packet[k] for k in ('x','y')}
    m.verify_reused_packet(packet,fields,static,0,np.array([.1],np.float32),np.array([1,2,3,4],np.float32))
    packet['state'][0,0,0]=9
    fields['state']=torch.zeros((1,3,2,2))
    with pytest.raises(ValueError):m.verify_reused_packet(packet,fields,static,0,np.array([.1],np.float32),np.array([1,2,3,4],np.float32))

def test_no_root_owned_export_delete():
    import ast,inspect
    tree=ast.parse(inspect.getsource(m.execute))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='rmtree']
    assert len(calls)==1 and ast.unparse(calls[0].args[0])=='view'

def test_truthful_progress():
    class Base:
        @staticmethod
        def atomic_json(path,value):Base.value=value
    m.progress(Base,Path('/tmp'),set(range(48)),705,None)
    assert Base.value['written_frames']==48 and not Base.value['official_reader_verified']
    m.progress(Base,Path('/tmp'),set(range(801)),800,15,True)
    assert Base.value['written_frames']==801 and Base.value['status']=='COMPLETE_NOT_TRAINING'
