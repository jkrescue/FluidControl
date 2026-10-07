"""Read-only saved-evidence fairness inventory; no CFD/model imports."""
import datetime, hashlib, json
from pathlib import Path
ROOT=Path('/workspace/fluid_control')
OUT=ROOT/'artifacts/p064_paper_baseline_fairness_audit_20261007/receipt.json'
PATHS={
 'periodic_result':'artifacts/tandem_cylinders/periodic_rotation_benchmark_result_20261003.json',
 'periodic_actions':'artifacts/tandem_cylinders/periodic_rotation_action_audit_20261003.json',
 'periodic_source':'cfd/tandem_cylinders/make_periodic_rotation_benchmark.py',
 'constant_grid':'artifacts/tandem_cylinders/constant_p100_grid_comparison.json',
 'B_approval':'docs/P064_B_E109_CONTINUATION_328_408_APPROVAL_20261007.json',
 'B_result':'artifacts/p064_b_continuation_328_408_cfd_20261007/result.json',
 'B_review':'docs/P064_B_CONTINUATION_328_408_TERMINAL_REVIEW_20261007.md',
 'B_receipt':'artifacts/p064_b_continuation_328_408_independent_audit_20261007/receipt.json',
 'restart_audit':'docs/P064_INDEPENDENT_CONTROL_RESTART_AUDIT_20261007.md',
 'paper_reproduction':'docs/PAPER_REPRODUCTION.md',
 'paper_value':'docs/PAPER_VALUE_AUDIT.md',
 'pre_execution_plan':'docs/CLOSED_LOOP_PAPER_VALIDATION_GAPS_20261007.md'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(k):return json.loads((ROOT/PATHS[k]).read_text())
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
inputs={k:{'path':p,'bytes':(ROOT/p).stat().st_size,'sha256':sha(ROOT/p)} for k,p in PATHS.items()}
assert inputs['B_result']['sha256']=='752b92d1063e51a8fb6a45ea539b173c3c5ffbd24c6a83255e0fa649f392063e'
assert inputs['B_receipt']['sha256']=='a26368b60007b53a422351f1d086fcda095cd823a98ded331e513c324de07af1'
a,p,g,b,receipt=read('periodic_actions'),read('periodic_result'),read('constant_grid'),read('B_approval'),read('B_receipt')
assert p['status']=='PERIODIC_OPEN_LOOP_CFD_BENCHMARK_OK'
assert a['analysis_window']==p['analysis_window']==[120.,160.]
assert len(p['decisions'])==2 and all(not r['one_phase_coarse_grid_screen_pass'] for r in p['decisions'])
assert b['start_time']==328 and b['end_time']==408 and b['steps']==800
assert receipt['new_cycles']==800 and receipt['feedback_links']==799 and receipt['new_solver_logs']==1600
assert receipt['no_model_or_solver_rerun'] is True
assert receipt['max_abs_delta']<=.10000000001
assert g['reference']=='control_small_p100' and g['candidate']=='control_grid_p100_medium'
assert a['source_restart']=='tandem_backward_dt005/t=80'
assert all(r['max_abs_omega']==1 for r in a['cases'].values())
assert 'OMEGA_LIMIT = 1.0' in (ROOT/PATHS['periodic_source']).read_text()
result={'status':'SAVED_BASELINE_FAIRNESS_EVIDENCE_AUDIT_NOT_NEW_PHYSICS','started_at':started,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inputs':inputs,
 'periodic':{'source_restart':a['source_restart'],'analysis_window':p['analysis_window'],'action_table_summary':a['cases'],'decisions':p['decisions'],'fair_matched_B_comparison':False,'reason':'Different restart/time window and action amplitude; P10 also exceeds current B slew limit. P20 rate is below B limit but amplitude remains unequal.'},
 'constant_grid':g,
 'B_existing_audit_reused':{'receipt':inputs['B_receipt'],'new_cycles':receipt['new_cycles'],'feedback_links':receipt['feedback_links'],'solver_logs':receipt['new_solver_logs'],'max_abs_delta':receipt['max_abs_delta'],'action_clock_or_raw_solver_reaudit_performed':False},
 'unknown_or_unverified':['Final B feedback policy medium-grid and halved-dt validation','Best constant/periodic versus B with matched initial state, limits, action effort and statistics window','Unopened prospectively sealed independent phases','Actuator torque/net-power saving'],
 'scope':'Re100 L/D5 unchanged; no model/solver/optimizer; descriptive evidence compatibility only; no reclassification of prior physical gates',
 'source_unchanged':all(sha(ROOT/v['path'])==v['sha256'] for v in inputs.values())}
assert result['source_unchanged']
if OUT.exists():raise FileExistsError(OUT)
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'receipt':str(OUT),'sha256':sha(OUT),'source_count':len(inputs)}))
