"""Independent saved-array arithmetic only; no model, torch or GPU imports."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

INDICES=[160,816,923,975,1077,1233]
CASES=['matched_start_acquisition_train_b00_zero','dynamic_train8_b00_prbs','dynamic_train8_b02_prbs','dynamic_train8_b04_prbs','dynamic_train8_b06_prbs','direct_cfd_directppo2048_v1_env0_ep0009_b00']
STARTS=[320,90,100,0,0,0]
STAT_KEYS=('bias_mse','rms_error_mse','absolute_rms_error','centered_residual_mse')
def close(a,b,objective=False):
    assert np.allclose(a,b,rtol=2e-6 if objective else 1e-10,atol=2e-8 if objective else 1e-12), (a,b)
def calc(pred,target,std):
    p,y,s=map(lambda x:np.asarray(x,dtype=np.float64),(pred,target,std))
    assert p.shape==y.shape==(100,4) and s.shape==(4,)
    assert np.isfinite(p).all() and np.isfinite(y).all() and np.isfinite(s).all() and (s>0).all()
    e=p-y;r=e*s;tail=r[38:,3]
    pr=np.std(p[38:,3])*s[3];tr=np.std(y[38:,3])*s[3]
    channels=np.mean(e*e,axis=0)
    return dict(balanced=float(channels@np.array([.125,.125,.125,.625])),channels=channels,
        stats=dict(bias_mse=float(tail.mean()**2),rms_error_mse=float((pr-tr)**2),absolute_rms_error=float(abs(pr-tr)),
        centered_residual_mse=float(np.mean((tail-tail.mean())**2)),signed_mean_error=float(tail.mean()),
        predicted_tail_rms=float(pr),truth_tail_rms=float(tr),four_force_physical_mae=np.abs(r).mean(0).tolist(),four_force_physical_mse=(r*r).mean(0).tolist()),
        # Descriptive residual metrics, not replacements for original gates.
        total_cd=dict(mae=float(np.abs(r[:,0]+r[:,2]).mean()),rmse=float(np.sqrt(np.mean((r[:,0]+r[:,2])**2)))),
        rear_cl=dict(mae=float(np.abs(r[:,3]).mean()),rmse=float(np.sqrt(np.mean(r[:,3]**2)))))
def audit(data):
    assert data['status']=='P064_FIT256_FIXED_SIX_COMPLETE_NOT_ADMISSION'
    assert data['optimizer_steps']==0 and data['candidate_saved'] is False
    assert set(data['panels'])=={'B','candidate'}
    computed={};descriptive={}
    for label,panel in data['panels'].items():
        assert panel['inference_precision']==dict(matmul='highest',cuda_tf32=False,cudnn_tf32=False)
        rows=panel['rows'];assert len(rows)==6
        assert [x['global_index'] for x in rows]==INDICES
        assert [x['identity']['case'] for x in rows]==CASES
        assert [x['identity']['start'] for x in rows]==STARTS
        assert all(x['identity']['split']=='train' for x in rows)
        values=[]
        for row in rows:
            a=row['arrays'];objective=row['panel']['objective'];v={}
            for domain in ('h1','ar'):
                v[domain]=calc(a[domain],a['target_force'],a['force_std'])
                close(v[domain]['balanced'],objective[domain+'_balanced'],True)
                close(v[domain]['channels'],objective[domain+'_channel_mse'],True)
                for k,x in v[domain]['stats'].items():close(x,row['panel']['domains'][domain][k])
            close(.5*(v['h1']['balanced']+v['ar']['balanced']),objective['total'],True)
            assert objective['chunks']==10 and objective['chunk_size']==10
            values.append(v)
        computed[label]={}
        descriptive[label]=[{d:{k:v[d][k] for k in ('total_cd','rear_cl')} for d in ('h1','ar')} for v in values]
        for domain in ('h1','ar'):
            score=float(np.mean([v[domain]['balanced'] for v in values]));computed[label][domain]=score
            close(score,panel['aggregate'][domain]['six_window_original_objective'],True)
            for k in STAT_KEYS:close(np.mean([v[domain]['stats'][k] for v in values[1:]]),panel['aggregate'][domain]['five_nonzero'][k])
        for name,warm in [('warm',True),('padded',False)]:
            idx=[i for i,r in enumerate(rows) if (r['identity']['start']>=3)==warm]; nz=[i for i in idx if i!=0]
            saved=panel['history_subgroups'][name]
            assert saved['window_count']==len(idx) and saved['nonzero_count']==len(nz)
            for d in ('h1','ar'):
                close(np.mean([values[i][d]['balanced'] for i in idx]),saved['domains'][d]['mean_original_objective'],True)
                for k in STAT_KEYS:close(np.mean([values[i][d]['stats'][k] for i in nz]),saved['domains'][d]['nonzero_statistics'][k])
    for b,c in zip(data['panels']['B']['rows'],data['panels']['candidate']['rows']):
        for key in ('identity','history','flow_history_sha256','global_index'):assert b[key]==c[key],key
        for key in ('target_force','force_std'):assert b['arrays'][key]==c['arrays'][key],key
    saved={l:{d:data['panels'][l]['aggregate'][d]['six_window_original_objective'] for d in ('h1','ar')} for l in ('B','candidate')}
    gate=all(saved['candidate'][d]<=saved['B'][d] for d in ('h1','ar'))
    assert gate==all(computed['candidate'][d]<=computed['B'][d] for d in ('h1','ar')),'rounding changes decision'
    return dict(status='SAVED_ARRAY_ARITHMETIC_PASS_NOT_ADMISSION',rows=12,endpoints=2400,computed= computed,
        original_and_nonregression=gate,descriptive_residual_metrics=descriptive,
        limitation='Saved arrays and producer-recorded flow-history digest equality only; no independent model forward or tensor digest reconstruction.')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--invocation',required=True);p.add_argument('--approval-sha256',required=True);a=p.parse_args()
    # Completion receipt is written only after the owned child exits. No live result read.
    launch=json.loads((a.result.parent/'launch.json').read_text());receipt=json.loads((a.result.parent/'supervisor_receipt.json').read_text())
    assert launch['unit']['InvocationID']==a.invocation
    assert receipt['returncode']==0 and receipt['error'] is None
    assert launch['approval_sha256']==receipt['approval_sha256']==a.approval_sha256
    assert sha(a.result)==a.sha256
    out=audit(json.loads(a.result.read_text()));out.update(result_sha256=a.sha256,invocation=a.invocation)
    print(json.dumps(out,allow_nan=False))
if __name__=='__main__':main()
