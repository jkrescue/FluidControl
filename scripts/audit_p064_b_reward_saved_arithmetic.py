"""Independent saved-ledger arithmetic; no model or producer invocation."""
import json,math,hashlib
from pathlib import Path
ROOT=Path('/workspace/fluid_control')
p=ROOT/'artifacts/p064_b_canonical_reward_sequences_20261007/result.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='af39c877bba7f990b35ea9802b58e966b91dbc78e9a0664292e798871060f5f1'
r=json.loads(p.read_text());maxdiff=0
def check(a,b):
 global maxdiff
 maxdiff=max(maxdiff,abs(a-b));assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
summary=[]
for phase,record in r['phases'].items():
 costs={kind:{} for kind in ('truth','prediction')}
 for role,rows in record['branches'].items():
  for kind in costs:costs[kind][role]=[]
  for row in rows:
   for kind in costs:
    x=row[kind];l=x['ledger'];assert l['sample_count']==62 and l['window_ready']
    drag=1-l['mean_total_drag']/l['baseline_total_drag']
    fluct=l['rear_cl_fluctuation_rms']/l['baseline_rear_cl_fluctuation_rms']
    bias=abs(l['mean_rear_cl'])/l['baseline_rear_cl_fluctuation_rms']
    check(drag,l['total_drag_reduction']);check(fluct,l['rear_cl_fluctuation_ratio']);check(bias,l['abs_mean_rear_cl_over_baseline_clprime_rms'])
    components={'drag_screen':-max(-1,min(1,drag)),'drag_gate_violation':max(0,(.02-drag)/.02)**2,
     'rear_cl_fluctuation_gate_violation':max(0,(fluct-1.05)/(1.05-1))**2,
     'rear_cl_mean_bias_gate_violation':max(0,(bias-.1)/.1)**2,
     'actuation':.01*(row['omega']/.75)**2,'rate':.01*(row['delta_omega']/.1)**2}
    for k,v in components.items():check(v,x['components'][k])
    cost=sum(components[k] for k in sorted(components));check(cost,x['total_cost']);costs[kind][role].append(cost)
 for item in record['ranking']:
  h=item['horizon'];endpoint=item['endpoint_window_cost'];ret=item['truncated_discounted_reward_no_bootstrap'];assert ret['gamma']==.99
  calc={};returns={}
  for kind,label in (('truth','truth'),('prediction','predicted')):
   calc[label]={role:v[h-1] for role,v in costs[kind].items()}
   returns[label]={role:sum(-.1*v[j]*.99**j for j in range(h)) for role,v in costs[kind].items()}
   for role in calc[label]:check(calc[label][role],endpoint[label][role]);check(returns[label][role],ret[label][role])
   mins=[role for role in ('minus','zero','plus') if calc[label][role]==min(calc[label].values())]
   maxs=[role for role in ('minus','zero','plus') if returns[label][role]==max(returns[label].values())]
   assert mins==endpoint[label+'_minimizers_exact'];assert maxs==ret[label+'_maximizers_exact']
  for role,v in endpoint['truth_regret_by_predicted_minimizer'].items():check(calc['truth'][role]-min(calc['truth'].values()),v)
  for role,v in ret['truth_return_regret_by_predicted_maximizer'].items():check(max(returns['truth'].values())-returns['truth'][role],v)
  summary.append({'phase':phase,'h':h,'endpoint':endpoint['predicted_minimizers_exact'],'truth_endpoint':endpoint['truth_minimizers_exact'],'return':ret['predicted_maximizers_exact'],'truth_return':ret['truth_maximizers_exact'],'return_regret':ret['truth_return_regret_by_predicted_maximizer']})
print(json.dumps({'status':'SAVED_LEDGER_COST_AND_RETURN_ARITHMETIC_ACCEPT','max_difference':maxdiff,'scope':'60 ledger costs plus 10 endpoint and truncated-return rankings; underlying raw-history interpolation not independently repeated','summary':summary},indent=2))
