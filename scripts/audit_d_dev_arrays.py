"""Independent saved-array arithmetic; no project/model inference imports."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/fluid_control')
parser=argparse.ArgumentParser()
parser.add_argument('--arm',choices=['D'],required=True)
parser.add_argument('--approval',type=Path,required=True)
parser.add_argument('--approval-sha256',required=True)
args=parser.parse_args()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
approval=args.approval
assert sha(approval)==args.approval_sha256
s=json.loads(approval.read_text())
OUT=ROOT/s['output']
r=json.loads((OUT/'result.json').read_text())
BOUT=ROOT/'artifacts/p064_arm_b_development_h1_h5_20261006'
assert sha(BOUT/'result.json')=='47e7d4c6931fadc62730790500bb9a8a07f44792d1bef82c900d10c36f9a8665'
br=json.loads((BOUT/'result.json').read_text())
for item in [s['driver'],s['base'],*s['sources'].values(),*s['runtime_sources'].values(),*s['inputs'].values()]:
    assert sha(ROOT/item['path'])==item['sha256'],item['path']
assert r['endpoints']==80 and r['candidate_label']==args.arm and r['optimizer_steps']==0 and r['model_tensors_unchanged'] and not r['scientific_admission']
assert r['manifest_sha256']==s['inputs']['manifest']['sha256']
assert r['inference_precision']['effective']==dict(float32_matmul_precision='highest',cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False)
selection=json.loads((ROOT/s['inputs']['selection']['path']).read_text())
expected=[(p,j) for p in ('b01','b03') for j in range(0,701,100)]
assert [(x['branch'],x['start_index']) for x in r['records']]==expected
data=[]; maxdelta=0.
def compare(actual,expected):
    global maxdelta
    for k,v in actual.items():
        b=np.asarray(expected[k],dtype=float);a=np.asarray(v,dtype=float)
        maxdelta=max(maxdelta,float(np.max(np.abs(a-b))))
        assert np.allclose(a,b,rtol=1e-12,atol=1e-10),(k,a,b)
def metrics(indices):
    x=[data[i] for i in indices];n=len(x)
    sums=[sum((a[k] for a in x),np.zeros_like(x[0][k])) for k in range(7)]
    se,ref,ps,fa,pfa,cd,pcd=sums
    return {str(j+1):dict(field_relative_l2_u_v_p=np.sqrt(se[j]/ref[j]).tolist(),
        velocity_relative_l2=float(np.sqrt(se[j,:2].sum()/ref[j,:2].sum())),
        persistence_field_relative_l2_u_v_p=np.sqrt(ps[j]/ref[j]).tolist(),
        force_channel_mae=(fa[j]/n).tolist(),persistence_force_channel_mae=(pfa[j]/n).tolist(),
        total_drag_mae=float(cd[j]/n),persistence_total_drag_mae=float(pcd[j]/n),
        field_squared_error_sums_u_v_p=se[j].tolist(),field_reference_squared_sums_u_v_p=ref[j].tolist(),
        persistence_field_squared_error_sums_u_v_p=ps[j].tolist(),segments=n) for j in range(5)}
for index,row in enumerate(r['records']):
    p=Path(row['physical_field_artifact']['path']);assert p.parent==OUT and sha(p)==row['physical_field_artifact']['sha256']
    with np.load(p,allow_pickle=False) as a:
        brow=br['records'][index];bp=Path(brow['physical_field_artifact']['path'])
        assert sha(bp)==brow['physical_field_artifact']['sha256']
        assert (brow['branch'],brow['start_index'])==(row['branch'],row['start_index'])
        with np.load(bp,allow_pickle=False) as b:
            for key in ('initial_state','truth_states','mask','x','y','time','omega','truth_forces','predicted_states'):
                assert np.array_equal(a[key],b[key]),('B/D matched array',index,key)
        assert all(np.isfinite(a[k]).all() for k in a.files)
        pred=a['predicted_states'].astype(np.float64);truth=a['truth_states'].astype(np.float64);initial=a['initial_state'].astype(np.float64)
        assert pred.shape==truth.shape==(5,3,128,256)
        mask=a['mask'][0].astype(bool);pf=a['predicted_forces'].astype(np.float64);tf=a['truth_forces'].astype(np.float64)
        assert pf.shape==(5,4) and tf.shape==(6,4)
        record=selection['phases'][row['branch']]['records'][index%8]
        assert record['start_index']==row['start_index'] and record['transitions']==row['transitions']
        assert np.array_equal(pf,np.array(row['predicted_forces']))
        assert np.array_equal(a['truth_forces'],np.asarray([record['initial_force']]+[t['truth_force'] for t in record['transitions']],np.float32))
        assert np.array_equal(a['omega'],np.asarray([record['transitions'][0]['omega_now']]+[t['omega_next'] for t in record['transitions']],np.float32)[:,None])
        assert np.max(np.abs(a['time'].ravel()-np.array([f['time'] for f in record['frames']])))<=1e-5
        se=((pred[:,:,mask]-truth[:,:,mask])**2).sum(-1)
        ref=(truth[:,:,mask]**2).sum(-1)
        ps=((initial[None,:,mask]-truth[:,:,mask])**2).sum(-1)
        force_error=pf-tf[1:];persist_error=tf[:1]-tf[1:]
        data.append([se,ref,ps,abs(force_error),abs(persist_error),abs(force_error[:,0]+force_error[:,2]),abs(persist_error[:,0]+persist_error[:,2])])
    m=metrics([index])
    for lead in m:compare(m[lead],row['metrics'][lead])
for indices,target in [(range(16),r['summary']),(range(8),r['phases']['b01']),(range(8,16),r['phases']['b03'])]:
    for lead,m in metrics(indices).items():compare(m,target[lead])
mem=[json.loads(x) for x in (OUT/'memory.jsonl').read_text().splitlines()]
supervisor=json.loads((OUT/'supervisor_result.json').read_text());assert supervisor['error'] is None and supervisor['returncode']==0
assert min(x['MemAvailable'] for x in mem)>=22*2**30
compact={p:{h:{k:(v[3] if k.endswith('channel_mae') else v[2] if k=='field_relative_l2_u_v_p' else v) for k,v in st[h].items() if k in ('velocity_relative_l2','field_relative_l2_u_v_p','force_channel_mae','persistence_force_channel_mae','total_drag_mae','persistence_total_drag_mae')} for h in ('1','5')} for p,st in [('pooled',r['summary']),*r['phases'].items()]}
wins={p:{str(h+1):dict(rear_cl=sum(d[3][h,3]<d[4][h,3] for d in data[a:b]),total_cd=sum(d[5][h]<d[6][h] for d in data[a:b])) for h in (0,4)} for p,a,b in [('b01',0,8),('b03',8,16)]}
comparison={phase:{h:{'B_rearCl':old[h]['force_channel_mae'][3],'D_rearCl':new[h]['force_channel_mae'][3],'B_totalCd':old[h]['total_drag_mae'],'D_totalCd':new[h]['total_drag_mae'],'persistence_rearCl':new[h]['persistence_force_channel_mae'][3],'persistence_totalCd':new[h]['persistence_total_drag_mae']} for h in ('1','2','3','4','5')} for phase,old,new in [('pooled',br['summary'],r['summary']),('b01',br['phases']['b01'],r['phases']['b01']),('b03',br['phases']['b03'],r['phases']['b03'])]}
print(json.dumps(dict(result_sha256=sha(OUT/'result.json'),max_absolute_roundoff=maxdelta,source_count=len(s['sources']),runtime_count=len(s['runtime_sources']),inputs=len(s['inputs']),minimum_available_bytes=min(x['MemAvailable'] for x in mem),memory_samples=len(mem),wall_seconds=mem[-1]['elapsed'],summary=compact,comparison=comparison,matched_B_arrays_and_frozen_fields=True,wins=wins,origins=[dict(phase=x['branch'],start=x['start_index'],cl=[float(v) for v in data[i][3][:,3]]) for i,x in enumerate(r['records'])]),indent=2,default=lambda v:int(v)))
