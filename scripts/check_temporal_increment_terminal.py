"""Pinned B engineering audit plus saved temporal auxiliary arithmetic; no forward."""
import contextlib, hashlib, importlib.util, inspect, io, json, math, sys
from pathlib import Path
ROOT=Path('/workspace/fluid_control')
BASE=ROOT/'scripts/check_p064_terminal_original.py'
assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='8b3cd86756f4fc5b36f19d0049969793bbb3e403bbeba8327324eb6b4e9e9587'
s=importlib.util.spec_from_file_location('temporal_terminal_base',BASE)
b=importlib.util.module_from_spec(s);sys.modules[s.name]=b;s.loader.exec_module(b)
for name,before,after in [
 ('check_records',"f'FC_P064_ARM_{arm}_TRAINING_COMPLETE_NOT_ADMISSION'","'FC_P064_TEMPORAL_INCREMENT_AUX_TRAINING_COMPLETE_NOT_ADMISSION'"),
 ('checkpoint_cpu',"f'FC_P064_ARM_{arm}_CONTROLLED_AERO_CHECKPOINT'","'FC_P064_TEMPORAL_INCREMENT_AUX_AERODYNAMIC_CHECKPOINT'"),
 ('main','f"FC_P064_ARM_{approval[\'arm\']}_DUAL_FNO_MANIFEST_VERIFIED"',"'FC_P064_TEMPORAL_INCREMENT_AUX_DUAL_FNO_MANIFEST_VERIFIED'")]:
    source=inspect.getsource(getattr(b,name));assert source.count(before)==1
    exec(source.replace(before,after),b.__dict__)

def close(x,y,label):
    b.require(math.isfinite(x) and math.isfinite(y) and math.isclose(x,y,rel_tol=3e-6,abs_tol=1e-9),label)

def check_approval_protocol(protocol,declared):
    aliases={'controlled_b00_windows':'b00_windows','controlled_b00_weight':'b00_weight'}
    for key,value in declared.items():
        if key=='fresh_adamw':
            # Constructor source is pinned; base inspects all 28 final Adam step32 states.
            b.require(value is True,'fresh Adam approval')
        else:b.require(protocol.get(aliases.get(key,key))==value,'approval protocol '+key)

def check_aux(result,protocol,events):
    expected=dict(weight=1.0,dt=.1,divide_by_dt=False,edges=99,within_chunk_edges=90,
      recomputed_boundary_edges=9,extra_aero_calls_per_window=9,extra_aero_samples_per_window=18,
      both_endpoints_require_gradient=True,normalization='unchanged_train_force_std')
    b.require(protocol['temporal_increment']==expected,'exact auxiliary protocol')
    b.require(protocol['objective']=='original_mixed_absolute_plus_H1_temporal_increment','training objective')
    b.require(protocol['diagnostic_objective']=='equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE','diagnostic objective')
    windows=[e for e in events if e.get('event')=='training_window_complete']
    updates=[e for e in events if e.get('event')=='accumulation_update_complete']
    b.require(len(windows)==256 and len(updates)==32,'aux journal counts')
    b.require([e['consumed'] for e in windows]==list(range(1,257)),'aux window order')
    b.require([e['update'] for e in updates]==list(range(1,33)),'aux update order')
    b.require([e['consumed_windows'] for e in updates]==[8*i for i in range(1,33)],'aux update consumed')
    for g,ev in zip(result['records'],updates):
        for key,target in [('original_total_loss',g['mean_objective']['total']),('temporal_increment_loss',g['mean_temporal_increment_loss']),('training_objective',g['mean_training_objective']),('preclip_mean_gradient_norm',g['preclip_mean_gradient_norm']),('applied_clip_scale',g['applied_clip_scale'])]:close(ev[key],target,key)
        for key,target in [('temporal_increment_loss','mean_temporal_increment_loss'),('training_objective','mean_training_objective')]:close(sum(r[key] for r in g['records'])/8,g[target],target)
    for row,event in zip([r for g in result['records'] for r in g['records']],windows):
        original=row['total'];aux=row['temporal_increment_loss'];training=row['training_objective']
        b.require(min(original,aux,training)>=0,'negative loss')
        close(original,.5*(row['h1_balanced']+row['ar_balanced']),'absolute mix')
        close(training,original+aux,'total plus lambda1aux')
        for key,value in [('original_total_loss',original),('temporal_increment_loss',aux),('temporal_increment_weighted_loss',aux),('training_objective',training)]:close(event[key],value,'window journal '+key)
    return dict(windows=256,updates=32,aux_weight=1.,saved_scalar_arithmetic=True,independent_loss_forward=False)

def retention(candidate,baseline):
    panels=[baseline['fixed_train_panels'][-1],candidate['fixed_train_panels'][-1]]
    b.require(all(p['consumed_windows']==256 and len(p['rows'])==6 for p in panels),'six final rows')
    for old,new in zip(panels[0]['rows'],panels[1]['rows']):
        for k in ('global_index','identity','history','flow_history_sha256'):b.require(old[k]==new[k],'same six identity')
    evidence={}
    for domain in ('h1','ar'):
        means=[]
        for panel in panels:
            values=[]
            for row in panel['rows']:
                obj=row['panel']['objective'];m=obj[domain+'_channel_mse'];v=.125*sum(m)+.5*m[3]
                close(v,obj[domain+'_balanced'],'balanced total-only');values.append(v)
            mean=sum(values)/6;close(mean,panel['aggregate'][domain]['six_window_original_objective'],'six mean');means.append(mean)
        evidence[domain]=dict(B=means[0],candidate=means[1],nondegrading=means[1]<=means[0],relative_change=means[1]/means[0]-1)
    evidence['retention_pass']=all(evidence[d]['nondegrading'] for d in ('h1','ar'))
    return evidence

def main():
    path=Path(sys.argv[sys.argv.index('--approval')+1])
    b.require(b.sha(path)=='98f5638ae454e532b9ddb22b90937c95468214400b0cfe31b3031f7e3dc565f2','actual approval')
    approval=b.read(path);stream=io.StringIO()
    # Original main rejects live invocation before touching result/checkpoint.
    with contextlib.redirect_stdout(stream):b.main()
    evidence=json.loads(stream.getvalue());unit=evidence['unit']
    b.require(unit['InvocationID']=='0d2508de79b646f08c87d7c0f0c1d53c','actual invocation')
    b.require(unit['MemoryMax']==str(12*2**30) and unit['MemorySwapMax']=='0','training limits')
    b.require(evidence['observed_min_available_gib']>=22,'reserve')
    for p,h in approval['overlay_sources'].items():b.require(b.sha(p)==h,'overlay SHA')
    runner=next(Path(p) for p in approval['overlay_sources'] if Path(p).name=='train_p064_temporal_increment.py')
    b.require(b.sha(runner)=='19e0799b73f218edf4a1e59f1695532154c3dec9fcf4641b07108c7da82c16c0','reviewed fresh optimizer source')
    text=runner.read_text()
    b.require('optimizer = torch.optim.AdamW(' in text and 'check_optimizer(aero, optimizer, 0)' in text,'fresh constructor and empty state check')
    evidence['fresh_optimizer_evidence']='pinned reviewed AdamW constructor plus initial step0 check; independently saved28 states step32'
    out=Path(approval['planned_output']);result=b.read(out/'result.json');protocol=b.read(out/'training_protocol.json')
    manifest=b.read(out/'dual_model_manifest.json')
    b.require(manifest['kind']=='FC_P064_TEMPORAL_INCREMENT_AUX_K1_FRESH_FORCE_FNO','kind')
    check_approval_protocol(protocol,approval['protocol'])
    consumer_path=next(Path(p) for p in approval['overlay_sources'] if Path(p).name=='dual_fno_temporal_increment.py')
    cs=importlib.util.spec_from_file_location('terminal_exact_temporal_consumer',consumer_path)
    consumer=importlib.util.module_from_spec(cs);sys.modules[cs.name]=consumer;cs.loader.exec_module(consumer)
    consumer._validate_p026_protocol(out,manifest,consumer._experiment_contract(manifest['kind']))
    text=b.subprocess.check_output(['journalctl','--user','_SYSTEMD_INVOCATION_ID='+unit['InvocationID'],'-o','cat','--no-pager'],text=True)
    events=[]
    for line in text.splitlines():
        try: row=json.loads(line)
        except json.JSONDecodeError:continue
        if isinstance(row,dict):events.append(row)
    old=ROOT/'artifacts/fcp064_controlled_aero_arm_b_20261006/result.json'
    b.require(b.sha(old)=='9167e8d811f64cf001cc87bfd45d9ed2d48f5c588a19b951f7be2c826637b980','B result comparator')
    evidence.update(auxiliary=check_aux(result,protocol,events),retention=retention(result,b.read(old)),model_forward=0)
    print(json.dumps(evidence,indent=2,allow_nan=False))

if __name__=='__main__':main()
