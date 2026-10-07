"""Bounded post-terminal saved-array/checkpoint inspection, no model construction."""
import argparse,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/fluid_control')
def require(v,m):
    if not v:raise RuntimeError(m)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def time_scalar(value):return np.asarray(value).reshape(()).item()
def bound(x):
    p=ROOT/x['path'];require(sha(p)==x['sha256'],'binding '+str(p));return p
def module(p,n):
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
def terminal(unit,inv,approval_sha):
    p=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show',unit,'-p','LoadState','-p','MainPID','-p','InvocationID','-p','Result','-p','ExecMainStatus'],text=True).splitlines())
    require(p['MainPID']=='0','training still live')
    if p['LoadState']!='not-found':
        require(p['InvocationID']==inv and p['Result']=='success' and p['ExecMainStatus']=='0','terminal identity/success')
        return dict(basis='retained unit',properties=p)
    events=[json.loads(x) for x in subprocess.check_output(['journalctl','--user','-u',unit,'-o','json','--no-pager'],text=True).splitlines()]
    exact=[e for e in events if e.get('USER_INVOCATION_ID')==inv]
    starts=[e for e in exact if str(e.get('MESSAGE','')).startswith('Started ')]
    ends=[e for e in exact if 'Consumed ' in str(e.get('MESSAGE',''))]
    require(len(starts)==len(ends)==1,'unique exact-inv manager lifecycle')
    require(int(starts[0]['__REALTIME_TIMESTAMP'])<int(ends[0]['__REALTIME_TIMESTAMP']),'manager order')
    require(not any(any(x in str(e.get('MESSAGE','')) for x in ('Failed with result','Main process exited','oom-kill')) for e in exact),'manager failure')
    return dict(basis='GC: manager exact invocation plus bound supervisor receipt',current_limits_unavailable=True,start=starts[0]['__REALTIME_TIMESTAMP'],end=ends[0]['__REALTIME_TIMESTAMP'])
def metrics(m,truth,std):
    err=np.asarray(m['prediction'],dtype=float)-truth
    require(err.shape==(256,4) and np.isfinite(err).all(),'finite256 prediction')
    z=err/std;calc=dict(loss=float(np.mean(np.sum(z*z*np.array([.125,.125,.125,.625]),axis=1))),normalized_rmse=np.sqrt(np.mean(z*z,axis=0)),physical_mae=np.mean(abs(err),axis=0),physical_rmse=np.sqrt(np.mean(err*err,axis=0)),physical_bias=np.mean(err,axis=0))
    difference=max(float(np.max(abs(np.asarray(v)-m[k]))) for k,v in calc.items())
    require(difference<2e-5,'saved physical float32 roundtrip metric mismatch; not admission tolerance')
    cd=err[:,0]+err[:,2];total=dict(mae=float(np.mean(abs(cd))),rmse=float(np.sqrt(np.mean(cd*cd))),bias=float(np.mean(cd)))
    require(all(abs(v-m['total_cd_physical'][k])<1e-6 for k,v in total.items()),'totalCd arithmetic')
    return {k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in calc.items()}|dict(total_cd_physical=total),difference
def records(r):
    accepted=[x for x in r['records'] if 'measurement' in x]
    require(0<=len(accepted)<=200 and 0<=r['closures']<=300,'bounded optimization')
    require([x['outer'] for x in accepted]==list(range(1,len(accepted)+1)),'accepted sequence')
    require([x['closure'] for x in r['trials']]==list(range(1,r['closures']+1)),'closure sequence')
    require(all(x['trial_not_accepted'] and np.isfinite([x['loss'],x['gradient_l2'],x['gradient_inf']]).all() for x in r['trials']),'finite trials')
    require(r['final']==(accepted[-1]['measurement'] if accepted else r['initial']),'final last accepted')
    tail=r['records'][len(accepted):]
    require(len(tail)<=1,'at most one terminal interrupted transaction')
    for x in tail:require(x['restored_previous_accepted'] and x['error_type']=='BudgetStop' and x['outer']==len(accepted)+1 and x['closures']==r['closures'],'unsuccessful transaction')
    prior_count=0
    for x in accepted:
        require(prior_count<x['closures']<=r['closures'] and x['closures_this_call']==x['closures']-prior_count,'accepted closure accounting')
        prior_count=x['closures']
    require(r['no_grad_evaluations']==len(accepted)+1,'initial plus accepted measurements')
    require(r['counts']==dict(aero=26*(r['closures']+r['no_grad_evaluations']),flow=0),'26 microbatch counts')
    fit=max(r['final']['normalized_rmse'])<=.01
    require((r['status']=='TRAIN_PANEL_FITTED_NOT_ADMISSION')==fit,'fit status exact .01')
    require(r['status'] in ('TRAIN_PANEL_FITTED_NOT_ADMISSION','BUDGET_STOP_NOT_FITTED','LOCAL_STAGNATION_NOT_FITTED'),'engineering success status')
    return accepted
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--approval',type=Path,required=True);ap.add_argument('--approval-sha256',required=True);ap.add_argument('--invocation',required=True);ap.add_argument('--receipt',type=Path,required=True);a=ap.parse_args()
    require(sha(a.approval)==a.approval_sha256,'approval SHA');s=read(a.approval)
    life=terminal(s['unit'],a.invocation,a.approval_sha256) # BEFORE result/checkpoint
    out=ROOT/s['output'];sup=read(out/'supervisor_receipt.json');child=read(out/'child.json')
    require(sup==dict(error=None,returncode=0,approval_sha256=a.approval_sha256),'supervisor success')
    require(child['execution']['InvocationID']==a.invocation,'owned child binding')
    for k in ('driver','core','consumer','panel_adapter','panel_receipt','b_training_approval','reference_approval','reference_result','reference_driver'):bound(s[k])
    for x in s['data_files']:bound(x)
    prior=read(bound(s['panel_receipt']));r=read(out/'result.json')
    require(r['panel']==prior['rows'] and r['coverage']==prior['summary'],'exact256 panel/coverage')
    require(r['protocol']==s['protocol'] and r['scientific_admission'] is False,'protocol/no admission')
    old=read(bound(s['reference_approval']));norm=read(bound(old['data']['normalization']));std=np.asarray(norm['all_force_std'],dtype=np.float32).astype(float)
    import torch,h5py
    torch.set_num_threads(1)
    truth=[]
    for row in r['panel']:
        with h5py.File(row['hdf_path'],'r') as f:
            force=np.asarray(f['force'][row['target_index']],dtype=np.float32)
            require(time_scalar(f['time'][row['target_index']])==row['time_target'],'actual target time')
        def ah(v):
            h=hashlib.sha256();h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes());return h.hexdigest()
        require(ah(force)==row['target_physical_sha256'],'actual target force')
        normalized=((torch.from_numpy(force)-torch.tensor(norm['all_force_mean'],dtype=torch.float32))/torch.tensor(norm['all_force_std'],dtype=torch.float32)).numpy()
        require(ah(normalized)==row['target_normalized_sha256'],'actual original normalization')
        truth.append(force)
    truth=np.asarray(truth,dtype=float);accepted=records(r);diff=0.;summary={}
    for name,m in [('initial',r['initial']),('final',r['final'])]+[(str(i),x['measurement']) for i,x in enumerate(accepted)]:
        result,d=metrics(m,truth,std);diff=max(diff,d)
        if name in ('initial','final'):summary[name]=result
    consumer=module(bound(s['consumer']),'rep256_review_consumer');mp=out/'candidate/dual_model_manifest.json'
    ident=consumer.validate_dual_fno_manifest(mp,expected_sha256=r['candidate']['manifest_sha256']);manifest=ident.payload
    require(r['candidate']['official_fresh_reload'] is True,'producer official reload')
    for k in ('driver','core','panel_adapter','consumer'):require(manifest['source_sha256'][k]==s[k]['sha256'],'manifest source')
    aerodir=mp.parent/manifest['aerodynamic']['checkpoint_relative_directory'];cp=torch.load(aerodir/manifest['aerodynamic']['state_file'],map_location='cpu',weights_only=True)
    require(cp['epoch']==1 and cp['metadata']==consumer.rep256_checkpoint_metadata(manifest),'official metadata')
    opt=cp['optimizer_state_dict'];group=opt['param_groups'][0]
    require(len(opt['param_groups'])==1 and len(group['params'])==28,'28 LBFGS params')
    for k,v in dict(lr=1.,max_iter=1,max_eval=300,history_size=5,line_search_fn='strong_wolfe').items():require(group[k]==v,'LBFGS '+k)
    require(len(opt['state'])<=1,'LBFGS first-parameter state, not Adam28')
    if accepted:
        state=next(iter(opt['state'].values()));require(state['n_iter']==accepted[-1]['optimizer_n_iter'],'saved accepted optimizer iterate')
        require(state['func_evals']==accepted[-1]['closures'],'optimizer rollback closure counter')
        require(len(state['old_dirs'])==len(state['old_stps'])<=5,'LBFGS history bound')
    def finite(v):
        if torch.is_tensor(v):require(bool(torch.isfinite(v).all()),'nonfinite optimizer tensor')
        elif isinstance(v,dict):
            for x in v.values():finite(x)
        elif isinstance(v,(list,tuple)):
            for x in v:finite(x)
    finite(opt);del cp,opt
    basepath=ROOT/'scripts/check_p064_terminal_original.py';require(sha(basepath)=='8b3cd86756f4fc5b36f19d0049969793bbb3e403bbeba8327324eb6b4e9e9587','original archive reader')
    base=module(basepath,'original_archive_reader');parentpath=bound(old['models']['B']['manifest']);parent=read(parentpath)
    parent_aero=parentpath.parent/parent['aerodynamic']['checkpoint_relative_directory']/parent['aerodynamic']['model_file']
    base.frozen_bias_cpu(parent_aero,aerodir/manifest['aerodynamic']['model_file'])
    def digest_state(values):
        h=hashlib.sha256()
        for name,t in sorted(values.items()):
            v=t.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
        return h.hexdigest()
    require(digest_state(base.model_state_cpu(parent_aero))==r['model_before'],'parent tensor digest')
    require(digest_state(base.model_state_cpu(aerodir/manifest['aerodynamic']['model_file']))==r['model_after']==r['candidate']['aerodynamic_tensor_digest'],'candidate tensor digest')
    for k in ('model_sha256','state_sha256'):require(manifest['flow'][k]==parent['flow'][k],'frozen flow exact files')
    memory=[json.loads(x) for x in (out/'memory.jsonl').read_text().splitlines()];minmem=min(x['available_bytes'] for x in memory)/2**30;require(minmem>=22,'memory floor')
    losses=[x['measurement']['loss'] for x in accepted]
    receipt=dict(status='REP256_SAVED_ARITHMETIC_CHECKPOINT_REVIEW_NOT_ADMISSION',lifecycle=life,result_sha256=sha(out/'result.json'),manifest_sha256=sha(mp),training_status=r['status'],metrics=summary,metric_roundtrip_maxdiff=diff,accepted_points=len(accepted),closures=r['closures'],counts=r['counts'],minimum_available_gib=minmem,last10_accepted_losses=losses[-10:],accepted_loss_increases=sum(b>a for a,b in zip(losses,losses[1:])),limits='No model construction/forward. Official fresh reload is producer evidence; archive tensors/metadata and saved-array arithmetic independently inspected. No six/dev admission conclusion.')
    require(not a.receipt.exists(),'unique receipt');a.receipt.parent.mkdir(parents=True,exist_ok=True)
    with a.receipt.open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps(receipt))
if __name__=='__main__':main()
