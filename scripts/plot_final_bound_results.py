"""CPU-only plotting of three completed, source-pinned results; no scientific run."""
import hashlib,json,csv,io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/workspace/fluid_control')
OUT=R/'docs/report_20261007/assets'
PINS={
 'e114':('artifacts/p064_b_continuation_328_408_cfd_20261007/result.json','752b92d1063e51a8fb6a45ea539b173c3c5ffbd24c6a83255e0fa649f392063e'),
 'fit':('artifacts/p064_representative256_training_20261007/result.json','50617c1c49cebcdc2198fb1cc427f7a7289dc1912846d3f882b0a257025b6d6a'),
 'six':('artifacts/p064_fit256_fixed_six_20261007/result.json','57add3a45f4cedcde94fbc242337e64cd6d54d51156e4d2be27c29cb8c309289')}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
data={}
for key,(name,digest) in PINS.items():
    p=R/name;assert sha(p)==digest;data[key]=json.loads(p.read_text())
failure_path='artifacts/p064_representative256_failure_map_20261007/metrics.csv'
failure_sha='96e8784c587176c2d46a2b543de849d9ea2b1cd569bffd9b7c9e6e70c5ca8824'
assert sha(R/failure_path)==failure_sha
failure_rows=list(csv.DictReader(io.StringIO((R/failure_path).read_text())))
PINS['failure_map']=(failure_path,failure_sha)
ppo_path='artifacts/p064_b_symmetry_canonical_h5_32768_ppo_20261007/payload/progress.json'
ppo_sha='5d2b2a330548c66406647ffb7db06ba8fa40ad140c992e13dddd9d9418601b2f'
assert sha(R/ppo_path)==ppo_sha
ppo_rows=[json.loads(line) for line in (R/ppo_path).read_text().splitlines() if line.strip()]
PINS['ppo_history']=(ppo_path,ppo_sha)
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
assets={}
def save(fig,name,description,counts):
    for ext in ('png','svg'):
        p=OUT/(name+'.'+ext)
        fig.savefig(p,dpi=160,bbox_inches='tight',metadata={'Creator':'Bound saved-results CPU plotting'})
        assets[p.name]={'sha256':sha(p),'bytes':p.stat().st_size,'description':description,'counts':counts,'smoothing':'none','downsampling':'none'}
    plt.close(fig)

rows=data['e114']['rows'];assert len(rows)==800 and [x['step'] for x in rows]==list(range(1,801))
t=np.array([x['end_time'] for x in rows]);assert np.allclose(t,328+np.arange(1,801)*.1,atol=1e-8,rtol=0)
controlled=np.array([x['output_observation'] for x in rows]);zero=np.array([x['zero_observation'] for x in rows]);omega=np.array([x['applied_omega'] for x in rows])
assert controlled.shape==zero.shape==(800,69)
fig,ax=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
for obs,label,col in [(zero,'Paired zero rotation','#777777'),(controlled,'Frozen B PPO / real CFD','#176ba0')]:
    ax[0].plot(t,obs[:,64]+obs[:,66],label=label,color=col,lw=1.15)
    ax[1].plot(t,obs[:,67],label=label,color=col,lw=1.15)
ax[2].plot(t,omega,color='#176ba0',lw=1.15,label='Applied endpoint omega* (B)')
ax[2].plot(t,np.zeros(800),color='#777777',lw=1,label='Zero branch')
ax[2].axhline(.75,ls=':',color='black',lw=.7);ax[2].axhline(-.75,ls=':',color='black',lw=.7)
for a,y in zip(ax,['Total Cd = front Cd + rear Cd','Rear-cylinder Cl','omega* = Omega D / U']):
    a.set_ylabel(y);a.grid(alpha=.2);a.legend(loc='upper right',ncol=2)
ax[-1].set_xlabel('Nondimensional time t* = t U / D');ax[-1].set_xlim(328,408)
fig.suptitle('E114: 800 new real-CFD feedback intervals (328 to 408)\nEndpoint observations; resumed trajectory, not a new physical condition',fontsize=13)
save(fig,'e114_closed_loop','Only E114 saved endpoint forces and applied endpoint actions; lines connect samples, not raw solver-rate forces.',{'endpoints_per_branch':800,'start':328,'first_endpoint':328.1,'end':408})

d=data['fit'];records=[r for r in d['records'] if 'measurement' in r];assert records
x=np.array([0]+[r['outer'] for r in records]);measurements=[d['initial']]+[r['measurement'] for r in records]
loss=np.array([m['loss'] for m in measurements]);rmse=np.array([m['normalized_rmse'] for m in measurements]);assert np.isfinite(loss).all() and np.isfinite(rmse).all()
assert measurements[-1]==d['final'];assert np.all(np.diff(x)>0)
fig,ax=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
ax[0].plot(x,loss,color='#176ba0',lw=1.7,label='Original balanced normalized H1 loss');ax[0].set_ylabel('Training objective');ax[0].legend()
for i,label in enumerate(['Front Cd','Front Cl','Rear Cd','Rear Cl']):ax[1].plot(x,rmse[:,i],lw=1.5,label=label)
ax[1].axhline(.01,color='black',ls='--',label='Train-fit target (not admission): 0.01')
ax[1].set_ylabel('Normalized force RMSE');ax[1].set_xlabel('Optimizer-returned outer point (0 = initial)');ax[1].legend(ncol=3)
for a in ax:a.grid(alpha=.2);a.set_xlim(0,x[-1])
fig.suptitle(f'Representative256: train-panel fit only / NOT ADOPTED\nAll {len(records)} returned accepted points; {d["closures"]} closures; trial losses excluded',fontsize=13)
save(fig,'rep256_accepted_fit','Initial point plus every saved optimizer-returned measurement; no trial minimum, no best-point selection.',{'accepted_records':len(records),'initial_points':1,'closures':d['closures'],'force_channels':4})

d=data['six'];labels=['B (retained)','Representative256 (rejected)'];colors=['#176ba0','#d46a28'];domains=['h1','ar']
fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for j,domain in enumerate(domains):
    for off,(key,label,col) in enumerate(zip(['B','candidate'],labels,colors)):
        vals=[r['panel']['objective'][domain+'_balanced'] for r in d['panels'][key]['rows']]
        assert len(vals)==6
        ax[j].bar(np.arange(6)+(off-.5)*.36,vals,width=.36,label=label,color=col)
        mean=d['panels'][key]['aggregate'][domain]['six_window_original_objective']
        ax[j].axhline(mean,color=col,ls='--',lw=1,label=f'{label}: mean {mean:.6f}')
    ax[j].set_ylim(0,max(r['panel']['objective'][domain+'_balanced'] for key in ('B','candidate') for r in d['panels'][key]['rows'])*1.6)
    ax[j].set_xticks(range(6),['Zero\nbase b00','PRBS\nb00','PRBS\nb02','PRBS\nb04','PRBS\nb06','PPO\nb00']);ax[j].set_xlabel('Fixed training-window source');ax[j].set_ylabel('Balanced normalized force MSE');ax[j].set_title('One-step teacher-forced H1' if domain=='h1' else '100-step continuous rollout (AR100)');ax[j].grid(axis='y',alpha=.2);ax[j].legend(fontsize=8)
fig.suptitle('Same precision: highest / no TF32; both original retention objectives worsen\nH1 +15.8169%, AR100 +14.2703%; not physical drag percentages',fontsize=12)
save(fig,'rep256_fixed_six','Original fixed-six objectives per window plus unweighted six-window means, B and candidate at the same highest/noTF32 precision.',{'models':2,'windows_each':6,'domains':['h1','ar'],'endpoints_per_domain_window':100,'window_identities':[{**r['identity'],'global_index':r['global_index']} for r in d['panels']['B']['rows']]})
families=['base20','train8','train16','controlled_b00'];channels=['frontCd','frontCl','rearCd','rearCl','totalCd'];ns=[109,45,38,64]
fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
for ax,channel in zip(axes.flat,channels):
    selected=[next(r for r in failure_rows if r['scope']=='selected256' and r['group']=='family:'+f and r['channel']==channel and r['mode']=='h1') for f in families]
    assert [int(r['n']) for r in selected]==ns
    before=np.array([float(r['before_mae']) for r in selected]);after=np.array([float(r['after_mae']) for r in selected]);delta=np.array([float(r['delta_mae']) for r in selected]);assert np.allclose(after-before,delta,rtol=1e-12,atol=1e-15)
    ax.bar(range(4),delta,color=['#176ba0' if v<=0 else '#d46a28' for v in delta]);ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(4),['Base20\nN=109','Train8\nN=45','Train16\nN=38','Control b00\nN=64'],fontsize=8);ax.set_title(channel);ax.set_ylabel('MAE change: candidate - B');ax.grid(axis='y',alpha=.2)
    ax.bar_label(ax.containers[0],labels=[f'{v:+.3g}' for v in delta],padding=3,fontsize=8);lo=min(delta.min(),0);hi=max(delta.max(),0);span=max(hi-lo,1e-8);ax.set_ylim(lo-.23*span,hi+.25*span)
axes.flat[-1].axis('off');axes.flat[-1].text(.03,.85,'Selected training panel only\n\nBlue: lower MAE\nOrange: higher MAE\n\n256 weighted points / 255 unique\nNo causal or generalization claim\nAxes have different physical scales',va='top',fontsize=12)
fig.suptitle('Representative256: improvement is uneven across training sources\nPhysical force-coefficient MAE changes; no smoothing or resampling',fontsize=13)
save(fig,'rep256_failure_map','Selected256 family H1 physical MAE differences from pinned saved-array CSV, candidate minus B; negative means improvement, positive deterioration.',{'families':dict(zip(families,ns)),'channels':channels,'bars':20,'unique_points':255,'weighted_points':256})
fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
ppo_counts={}
for ax,field,label,color in zip(axes,['rollout/ep_rew_mean','train/value_loss'],['Logged mean episode reward','Logged PPO value loss'],['#176ba0','#d46a28']):
    points=[(r['time/total_timesteps'],r[field]) for r in ppo_rows if field in r and 'time/total_timesteps' in r]
    assert points and all(np.isfinite(v) for _,v in points)
    xx,yy=zip(*points);assert all(b>=a for a,b in zip(xx,xx[1:]));ax.plot(xx,yy,color=color,lw=1.5,marker='.',ms=3);ax.set_ylabel(label);ax.grid(alpha=.2);ppo_counts[field]={'records':len(points),'first_step':xx[0],'last_step':xx[-1]}
axes[-1].set_xlabel('Cumulative environment interaction steps');fig.suptitle('E082: frozen-B FNO-environment PPO learning history\nSaved SB3 logger values, not real-CFD drag reduction; no extra smoothing',fontsize=13)
save(fig,'e082_ppo_learning','Actual saved SB3 logger sequence. Episode reward is logger rolling episode mean, not physical performance; value loss is PPO training diagnostic. No final-policy reconstruction.',{'log_rows':len(ppo_rows),'series':ppo_counts})
manifest={'status':'SAVED_RESULT_VISUALIZATION_NOT_NEW_SCIENCE','script_sha256':sha(Path(__file__)),'sources':{k:{'path':p,'sha256':h} for k,(p,h) in PINS.items()},'assets':assets,'source_unchanged':all(sha(R/p)==h for p,h in PINS.values()),'no_model_loading':True,'no_solver':True}
manifest['engineering_notes']='R1 stopped after E114 plots because records also contained a final budget-stop row; only measurement-bearing returned points are plotted. One terminal restored-budget record and all trial closures are excluded. Final layout adds legend headroom; no data values changed.'
p=OUT/'source_manifest.json';p.write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'manifest_sha256':sha(p),'files':list(assets)},indent=2))
