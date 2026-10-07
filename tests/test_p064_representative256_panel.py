import importlib.util
from pathlib import Path
from types import SimpleNamespace

def module():
    path=(Path(__file__).resolve().parents[1]
          / 'artifacts/p064_representative256_source_20261007_immutable/panel256.py')
    spec=importlib.util.spec_from_file_location('panel256test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_two_source_counters_and_exact_order():
    m=module();rows=[dict(consumed_position=i,original_global_index=255-i,source='controlled_b00' if i%4==0 else 'original44',b00_start=i if i%4==0 else None) for i in range(256)]
    out=m.select(rows)
    assert [x['original_global_index'] for x in out]==list(range(255,-1,-1))
    for key,n in [('original44',192),('controlled_b00',64)]:
        source=[r for r in out if r['source']==key]
        assert [r['occurrence_k'] for r in source]==list(range(n))
        assert [r['lead'] for r in source]==[1+37*k%100 for k in range(n)]
        assert all(r['weight_denominator']==256 for r in source)

def test_original_global_index_boundaries():
    m=module();children=[SimpleNamespace(index=[(j,i) for i in range(n)]) for j,n in enumerate((720,408,240))]
    for idx,family,local in [(0,0,0),(719,0,719),(720,1,0),(1127,1,407),(1128,2,0),(1367,2,239)]:
        got=m.resolve_original(children,idx)
        assert got[0]==family and got[2:]==(family,local)

def test_last_six_gradient_has_exact_256_denominator():
    import torch
    x=torch.arange(1024,dtype=torch.float64).reshape(256,4)/1000
    w=x.new_tensor([.125,.125,.125,.625])
    a=torch.tensor([.4,.8,1.2,1.6],dtype=torch.float64,requires_grad=True)
    b=a.detach().clone().requires_grad_(True)
    full=((x*a-x.square()).square()*w).sum()/256;full.backward()
    value=0;counts=[]
    for i in range(0,256,10):
        xx=x[i:i+10];counts.append(len(xx));loss=((xx*b-xx.square()).square()*w).sum()/256
        loss.backward();value+=float(loss.detach())
    assert counts==[10]*25+[6]
    torch.testing.assert_close(a.grad,b.grad,rtol=1e-12,atol=1e-12)
    assert abs(value-float(full.detach()))<1e-12
