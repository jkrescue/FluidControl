"""G-only extension of pinned B terminal checker; no model forward.

Training AR resets every five steps; fixed-six diagnostics remain continuous
AR100. Saved audit consistency is not independent replay of losses or states.
"""
import hashlib
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path

BASE = Path(__file__).with_name('check_p064_terminal_original.py')
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == '8b3cd86756f4fc5b36f19d0049969793bbb3e403bbeba8327324eb6b4e9e9587'
spec = importlib.util.spec_from_file_location('f_terminal_base', BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

IDENTITY_DELTAS = (
    ('check_records', "f'FC_P064_ARM_{arm}_TRAINING_COMPLETE_NOT_ADMISSION'", "'FC_P064_AR5_RESET_TRAINING_COMPLETE_NOT_ADMISSION'"),
    ('checkpoint_cpu', "f'FC_P064_ARM_{arm}_CONTROLLED_AERO_CHECKPOINT'", "'FC_P064_AR5_RESET_AERODYNAMIC_CHECKPOINT'"),
    ('main', 'f"FC_P064_ARM_{approval[\'arm\']}_DUAL_FNO_MANIFEST_VERIFIED"', "'FC_P064_AR5_RESET_DUAL_FNO_MANIFEST_VERIFIED'"),
)
for name, before, after in IDENTITY_DELTAS:
    source = inspect.getsource(getattr(base, name))
    assert source.count(before) == 1
    exec(source.replace(before, after), base.__dict__)


def check_reset_objective(result, protocol, manifest):
    required = dict(
        training_experiment='FC-P064-AR5-RESET', arm='B',
        candidate_profile='FC_P064_AR5_RESET_K1_FRESH',
        objective='equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE',
        training_ar_reset_every=5, training_ar_reset_indices=list(range(0,100,5)),
        training_flow_forward_calls_per_window=100, training_ar_supervised_points=100,
        diagnostic_ar_reset_every=None, diagnostic_ar_horizon=100,
        training_windows=256, optimizer_steps=32, accumulation_windows=8,
        b00_windows=64, b00_weight=0.25, replacement_within_each_update=[0, 4],
        validation_accessed=False, frozen_test_accessed=False, selection_performed=False,
    )
    for key, value in required.items():
        base.require(protocol[key] == value, 'G protocol: ' + key)
    base.require(manifest['kind'] == 'FC_P064_AR5_RESET_K1_FRESH_FORCE_FNO'
                 and manifest['training_experiment'] == 'FC-P064-AR5-RESET', 'G manifest kind')
    base.require('auxiliary_records' not in result and 'auxiliary_response' not in protocol,
                 'unexpected E auxiliary objective')
    base.require(len(result['records']) == 32, 'G update count')
    audit=result['training_state_audit']
    base.require(len(audit)==256, '256 state audit records')
    profile=dict(profile='training_ar_reset_every_5_true_current_states',
        reset_indices=list(range(0,100,5)), flow_forward_calls=100,
        supervised_ar_points=100, discarded_block_end_updates=20,
        future_state_inputs=False, optimizer_steps=0)
    objectives = []
    for group in result['records']:
        base.require(len(group['records']) == 8, 'G window group')
        for row in group['records']:
            state=audit[len(objectives)]
            base.require(row['ar_state_profile']==profile, 'training reset profile')
            base.require(state==dict(identity=row['identity'],flow_history_sha256=row['flow_history_sha256'],**profile), 'state/record pairing')
            digest=row['flow_history_sha256']
            base.require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest), 'state digest')
            values = [row[k] for k in ('h1_balanced', 'ar_balanced', 'total')]
            base.require(all(math.isfinite(x) and x >= 0 for x in values), 'G nonfinite/negative objective')
            base.require('training_objective' not in row, 'unexpected H1-only target')
            # Independent FP32 chunk sums can differ from half the saved sums.
            base.require(math.isclose(row['total'], 0.5 * (row['h1_balanced'] + row['ar_balanced']),
                                      rel_tol=2e-6, abs_tol=1e-9), 'diagnostic mixture')
            objectives.append(row['total'])
    for panel in result['fixed_train_panels']:
        base.require(len(panel['rows'])==6, 'six diagnostic windows')
        for row in panel['rows']:
            objective=row['panel']['objective']
            base.require(objective['ar_state_profile']==dict(profile='diagnostic_continuous_ar100',reset_indices=[0],flow_forward_calls=100,supervised_ar_points=100), 'diagnostic must remain continuous AR100')
    return dict(training_objective='equal H1 and reset-AR5', checked_windows=len(objectives),
                reset_indices_per_window=20, flow_calls_per_training_window=100,
                diagnostic_total='0.5 H1 + 0.5 continuous AR100',
                mean_saved_training_objective=sum(objectives) / len(objectives),
                independent_training_loss_recomputed=False,
                independent_model_forward=False)


def capture_read(original_read, approval_path, capture):
    """Bind full paths, so a same-basename parent cannot replace the candidate."""
    approval_path = Path(approval_path).resolve()
    def read(path):
        value = original_read(path)
        path = Path(path).resolve()
        if path == approval_path:
            capture['approval'] = value
        elif 'approval' in capture:
            output = Path(capture['approval']['planned_output']).resolve()
            if path.parent == output and path.name in ('result.json', 'training_protocol.json', 'dual_model_manifest.json'):
                capture[path.name] = value
        return value
    return read


def check_approval_protocol(protocol, approval):
    # Approval intentionally omits redundant supervised_points/reset_module
    # keys. They remain checked as fixed100 and against the actual bound file.
    for key in ('training_experiment', 'objective', 'training_ar_reset_every',
                'training_ar_reset_indices', 'training_flow_forward_calls_per_window',
                'diagnostic_ar_reset_every', 'diagnostic_ar_horizon'):
        base.require(protocol[key] == approval['protocol'][key], 'approval objective binding')


def main():
    # Base argparse, terminal-before-candidate gate and all old checks remain.
    approval_path = sys.argv[sys.argv.index('--approval') + 1]
    capture = {}
    original_read = base.read
    base.read = capture_read(original_read, approval_path, capture)
    base.main()
    approval = capture['approval']
    evidence = check_reset_objective(capture['result.json'], capture['training_protocol.json'],
                                  capture['dual_model_manifest.json'])
    base.require(len(approval['overlay_sources']) == 3, 'G overlay count')
    for path, digest in approval['overlay_sources'].items():
        base.require(base.sha(path) == digest, 'G overlay SHA')
    check_approval_protocol(capture['training_protocol.json'], approval)
    argv=approval['argv']
    cli={argv[i][2:].replace('-','_'):argv[i+1] for i in range(len(argv)-1) if argv[i].startswith('--')}
    base.require(capture['training_protocol.json']['reset_module_sha256'] == base.sha(cli['reset_module']), 'bound reset module')
    base.require(min(x['MemAvailable'] for x in capture['result.json']['resources']) >= 22, 'runtime reserve')
    evidence['overlay_sources_rehashed'] = 3
    print(json.dumps(evidence, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
