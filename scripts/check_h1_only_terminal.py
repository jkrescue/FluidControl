"""F-only extension of the pinned B terminal checker; no model forward.

Diagnostic total is not the backward objective. This checks saved scalar
consistency and source binding, not an independent replay of training loss.
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
    ('check_records', "f'FC_P064_ARM_{arm}_TRAINING_COMPLETE_NOT_ADMISSION'", "'FC_P064_H1_ONLY_TRAINING_COMPLETE_NOT_ADMISSION'"),
    ('checkpoint_cpu', "f'FC_P064_ARM_{arm}_CONTROLLED_AERO_CHECKPOINT'", "'FC_P064_H1_ONLY_AERODYNAMIC_CHECKPOINT'"),
    ('main', 'f"FC_P064_ARM_{approval[\'arm\']}_DUAL_FNO_MANIFEST_VERIFIED"', "'FC_P064_H1_ONLY_DUAL_FNO_MANIFEST_VERIFIED'"),
)
for name, before, after in IDENTITY_DELTAS:
    source = inspect.getsource(getattr(base, name))
    assert source.count(before) == 1
    exec(source.replace(before, after), base.__dict__)


def check_h1_objective(result, protocol, manifest):
    required = dict(
        training_experiment='FC-P064-H1-ONLY', arm='B',
        candidate_profile='FC_P064_H1_ONLY_K1_FRESH',
        objective='H1_only_half_equal_four_half_rearCl_normalized_MSE',
        diagnostic_objective='equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE',
        mixed_forward_preserved=True,
        h1_gradient_multiplier_vs_b_component=2.0,
        ar_gradient_multiplier_vs_b_component=0.0,
        training_windows=256, optimizer_steps=32, accumulation_windows=8,
        b00_windows=64, b00_weight=0.25, replacement_within_each_update=[0, 4],
        validation_accessed=False, frozen_test_accessed=False, selection_performed=False,
    )
    for key, value in required.items():
        base.require(protocol[key] == value, 'F protocol: ' + key)
    base.require(manifest['kind'] == 'FC_P064_H1_ONLY_K1_FRESH_FORCE_FNO'
                 and manifest['training_experiment'] == 'FC-P064-H1-ONLY', 'F manifest kind')
    base.require('auxiliary_records' not in result and 'auxiliary_response' not in protocol,
                 'unexpected E auxiliary objective')
    base.require(len(result['records']) == 32, 'F update count')
    objectives = []
    for group in result['records']:
        base.require(len(group['records']) == 8, 'F window group')
        for row in group['records']:
            values = [row[k] for k in ('training_objective', 'h1_balanced', 'ar_balanced', 'total')]
            base.require(all(math.isfinite(x) and x >= 0 for x in values), 'F nonfinite/negative objective')
            base.require(row['training_objective'] == row['h1_balanced'], 'backward target is not H1')
            base.require(row['diagnostic_total_semantics'] == 'unchanged_half_H1_half_AR', 'diagnostic semantics')
            # Independent FP32 chunk sums can differ from half the saved sums.
            base.require(math.isclose(row['total'], 0.5 * (row['h1_balanced'] + row['ar_balanced']),
                                      rel_tol=2e-6, abs_tol=1e-9), 'diagnostic mixture')
            objectives.append(row['training_objective'])
    return dict(training_objective='H1_only', checked_windows=len(objectives),
                diagnostic_total='0.5 H1 + 0.5 AR; not the backward objective',
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


def main():
    # Base argparse, terminal-before-candidate gate and all old checks remain.
    approval_path = sys.argv[sys.argv.index('--approval') + 1]
    capture = {}
    original_read = base.read
    base.read = capture_read(original_read, approval_path, capture)
    base.main()
    approval = capture['approval']
    evidence = check_h1_objective(capture['result.json'], capture['training_protocol.json'],
                                  capture['dual_model_manifest.json'])
    base.require(len(approval['overlay_sources']) == 3, 'F overlay count')
    for path, digest in approval['overlay_sources'].items():
        base.require(base.sha(path) == digest, 'F overlay SHA')
    for key in ('training_experiment', 'objective', 'diagnostic_objective',
                'mixed_forward_preserved', 'h1_gradient_multiplier_vs_b_component',
                'ar_gradient_multiplier_vs_b_component'):
        base.require(capture['training_protocol.json'][key] == approval['protocol'][key], 'approval objective binding')
    base.require(min(x['MemAvailable'] for x in capture['result.json']['resources']) >= 22, 'runtime reserve')
    evidence['overlay_sources_rehashed'] = 3
    print(json.dumps(evidence, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
