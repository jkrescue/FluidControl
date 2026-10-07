"""Read-only saved-prediction decomposition; no model imports or inference."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
import h5py

ROOT=Path('/workspace/fluid_control')
OUT=ROOT/'artifacts/p064_representative256_failure_map_20261007'
def read(p): return json.loads((ROOT/p).read_text())
def sha(p): return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def ah(v):
    h=hashlib.sha256();h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes());return h.hexdigest()
rp='artifacts/p064_representative256_training_20261007/result.json'
sp='artifacts/p064_fit256_fixed_six_20261007/result.json'
assert sha(rp)=='50617c1c49cebcdc2198fb1cc427f7a7289dc1912846d3f882b0a257025b6d6a'
assert sha(sp)=='57add3a45f4cedcde94fbc242337e64cd6d54d51156e4d2be27c29cb8c309289'
r=read(rp);s=read(sp);rows=[]
def add(scope,group,mode,old,new,std=None):
    old=np.asarray(old);new=np.asarray(new)
    if std is not None: old=old*std;new=new*std
    for j,ch in enumerate(['frontCd','frontCl','rearCd','rearCl','totalCd']):
        a=old[:,j] if j<4 else old[:,0]+old[:,2]
        b=new[:,j] if j<4 else new[:,0]+new[:,2]
        rows.append(dict(scope=scope,group=group,mode=mode,channel=ch,n=len(a),before_mae=float(abs(a).mean()),after_mae=float(abs(b).mean()),delta_mae=float(abs(b).mean()-abs(a).mean())))
truth=[]
for p in r['panel']:
    with h5py.File(p['hdf_path'],'r') as f: v=np.asarray(f['force'][p['target_index']],dtype=np.float32)
    assert ah(v)==p['target_physical_sha256'];truth.append(v)
truth=np.asarray(truth);old=np.asarray(r['initial']['prediction'])-truth;new=np.asarray(r['final']['prediction'])-truth
groups={'all':list(range(256))}
for i,p in enumerate(r['panel']):
    family=['base20','train8','train16','controlled_b00'][p['dataset_index']]
    action='zero' if p['omega_current']==p['omega_next']==0 else ('changing' if p['omega_current']!=p['omega_next'] else 'constant_nonzero')
    for g in ['family:'+family,'action:'+action,'lead:'+str(p['lead']),'lead_quarter:'+str((p['lead']-1)//25+1)]:groups.setdefault(g,[]).append(i)
for g,idx in groups.items():add('selected256',g,'h1',old[idx],new[idx])
six_stats=[]
for b,c in zip(s['panels']['B']['rows'],s['panels']['candidate']['rows']):
    assert b['identity']==c['identity']; ba=b['arrays'];ca=c['arrays']
    assert ba['target_force']==ca['target_force'] and ba['force_std']==ca['force_std']
    target=np.asarray(ba['target_force']);std=np.asarray(ba['force_std']);case=b['identity']['case'];start=b['identity']['start']
    for mode in ['h1','ar']:
        be=np.asarray(ba[mode])-target;ce=np.asarray(ca[mode])-target
        add('six',case,mode,be,ce,std)
        for q in range(4):add('six_quarter',case+':lead'+str(q*25+1)+'-'+str((q+1)*25),mode,be[q*25:(q+1)*25],ce[q*25:(q+1)*25],std)
        selected={p['target_index'] for p in r['panel'] if p['case']==case}
        ix=[i for i in range(100) if start+i+1 in selected];other=[i for i in range(100) if i not in ix]
        for name,inds in [('selected_overlap',ix),('not_selected',other)]:
            if inds:add('six_overlap',case+':'+name,mode,be[inds],ce[inds],std)
        six_stats.append(dict(case=case,mode=mode,selected_overlap_count=len(ix),physical_mae_before=np.mean(abs(be*std),axis=0).tolist(),physical_mae_after=np.mean(abs(ce*std),axis=0).tolist()))
for case in {x['group'] for x in rows if x['scope']=='six'}:
    for ch in ['frontCd','frontCl','rearCd','rearCl','totalCd']:
        h=next(x for x in rows if x['scope']=='six' and x['group']==case and x['mode']=='h1' and x['channel']==ch)
        a=next(x for x in rows if x['scope']=='six' and x['group']==case and x['mode']=='ar' and x['channel']==ch)
        rows.append(dict(scope='six_ar_minus_h1',group=case,mode='mae_gap_not_error_of_difference',channel=ch,n=100,before_mae=a['before_mae']-h['before_mae'],after_mae=a['after_mae']-h['after_mae'],delta_mae=(a['after_mae']-h['after_mae'])-(a['before_mae']-h['before_mae'])))
OUT.mkdir(exist_ok=True)
with (OUT/'metrics.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'summary.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs={rp:sha(rp),sp:sha(sp)},six=six_stats,rows=rows,model_forwards=0),indent=2))
print(json.dumps(dict(output=str(OUT),rows=len(rows),six=six_stats)))
