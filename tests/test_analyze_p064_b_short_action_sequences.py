import importlib.util
from pathlib import Path

source=Path(__file__).with_name('analyze_p064_b_short_action_sequences.py')
if not source.is_file():
    source=Path(__file__).resolve().parents[1]/'scripts/analyze_p064_b_short_action_sequences.py'
spec=importlib.util.spec_from_file_location('audit',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_cumulative_cost_and_wrong_selection():
    b={r:{'truth':[[v,0,0,2] for _ in range(5)],'prediction':[[p,0,0,3] for _ in range(5)]} for r,v,p in [('minus',1,3),('zero',2,2),('plus',3,1)]}
    x=m.assess(b)
    assert len(x)==5 and x[4]['predicted_minimizer_set']==['plus']
    assert x[4]['cfd_minimizer_set']==['minus']
    assert x[4]['selected_realized_costs'][0]['cfd_regret_vs_exact_optimum']==2
    assert x[0]['components']['minus']['truth']['mean_squared_rear_cl']==4
    assert x[0]['action_minus_zero']['minus']['mean_total_cd']['prediction_delta_error']==2

def test_exact_ties_not_forced_selection():
    b={r:{'truth':[[1,0,0,0]]*5,'prediction':[[1,0,0,0]]*5} for r in m.ROLES}
    x=m.assess(b)[0]
    assert x['predicted_minimizer_set']==list(m.ROLES)
    assert all(p['truth_exact_sign']==p['prediction_exact_sign']==0 for p in x['total_cd_pairs'])

def test_horizons_exclude_q0_and_use_all_prefix_points():
    b={r:{'truth':[[i,0,0,(-1)**i] for i in range(1,6)],'prediction':[[i,0,0,(-1)**i] for i in range(1,6)]} for r in m.ROLES}
    x=m.assess(b)
    assert x[0]['components']['zero']['truth']['mean_total_cd']==1
    assert x[4]['components']['zero']['truth']['mean_total_cd']==3
    assert x[1]['components']['zero']['truth']['mean_rear_cl']==0
    assert x[1]['components']['zero']['truth']['mean_squared_rear_cl']==1
