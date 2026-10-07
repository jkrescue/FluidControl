"""Read-only absolute64 extension of the pinned original B checker.

No model forward; saved scalar consistency is not independent loss replay.
The original terminal-before-candidate guard remains in force.
"""
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

BASE = Path(__file__).with_name('check_p064_terminal_original.py')
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == '8b3cd86756f4fc5b36f19d0049969793bbb3e403bbeba8327324eb6b4e9e9587'
spec = importlib.util.spec_from_file_location('absolute64_base', BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def replace_function(name, replacements):
    source = inspect.getsource(getattr(base, name))
    for before, after in replacements:
        assert source.count(before) == 1, (name, before)
        source = source.replace(before, after)
    exec(source, base.__dict__)


replace_function('check_records', [
    ("f'FC_P064_ARM_{arm}_TRAINING_COMPLETE_NOT_ADMISSION'", "f'FC_P064_ABSOLUTE64_ARM_{arm}_TRAINING_COMPLETE_NOT_ADMISSION'"),
    ("result['optimizer_steps'] == 32 and result['training_windows'] == 256", "result['optimizer_steps'] == 64 and result['training_windows'] == 512"),
    ("len(result['records']) == 32", "len(result['records']) == 64"),
    ('list(range(1, 33))', 'list(range(1, 65))'),
    ('len(ordered) == 288', 'len(ordered) == 576'),
    ('for i in range(32)', 'for i in range(64)'),
    ('== [0, 256]', '== [0, 512]'),
])
original_schedule = base.schedule


def repeated_schedule(order, arm):
    return [dict(row, consumed=epoch * 256 + row['consumed'])
            for epoch in range(2) for row in original_schedule(order, arm)]


base.schedule = repeated_schedule
replace_function('checkpoint_cpu', [
    ("checkpoint['epoch'] == 1", "checkpoint['epoch'] == 2"),
    ("f'FC_P064_ARM_{arm}_CONTROLLED_AERO_CHECKPOINT'", "f'FC_P064_ABSOLUTE64_ARM_{arm}_CONTROLLED_AERO_CHECKPOINT'"),
    ("float(state['step']) == 32", "float(state['step']) == 64"),
    ("'Adam step not 32'", "'Adam step not 64'"),
    ("'all_steps': 32", "'all_steps': 64"),
])
replace_function('main', [
    ('f"FC_P064_ARM_{approval[\'arm\']}_DUAL_FNO_MANIFEST_VERIFIED"', 'f"FC_P064_ABSOLUTE64_ARM_{approval[\'arm\']}_DUAL_FNO_MANIFEST_VERIFIED"'),
    ('records=32, consumed=256', 'records=64, consumed=512'),
    ('aerodynamic/checkpoint.0.1.pt', 'aerodynamic/checkpoint.0.2.pt'),
    ('aerodynamic/FNO.0.1.mdlus', 'aerodynamic/FNO.0.2.mdlus'),
])


def check_extension(result, protocol, manifest, comparison, historical_b):
    for key, value in dict(training_experiment='FC-P064-ABSOLUTE64', arm='B',
                           training_windows=512, optimizer_steps=64,
                           accumulation_windows=8, schedule_epochs=2,
                           diagnostic_counts=[0, 512], b00_windows=128,
                           b00_windows_per_schedule_epoch=64, b00_weight=0.25,
                           replacement_within_each_update=[0, 4],
                           objective='equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE',
                           allocator_bytes=16 * 1024**3).items():
        base.require(protocol[key] == value, 'absolute64 protocol: ' + key)
    expected = historical_b['aerodynamic_terminal_tensor_sha256']
    base.require(comparison == result['update32_comparison'] ==
                 dict(expected=expected, actual=expected, exact=True), 'update32 actual comparison')
    base.require(protocol['update32_reference_tensor_sha256'] == expected, 'protocol reference')
    base.require(manifest['kind'] == 'FC_P064_ABSOLUTE64_ARM_B_CONTROLLED_AERO_FORCE_FNO', 'manifest kind')
    aero = manifest['aerodynamic']
    base.require(aero['checkpoint_epoch'] == 2 and aero['model_file'] == 'FNO.0.2.mdlus'
                 and aero['state_file'] == 'checkpoint.0.2.pt', 'epoch2 saved paths')
    for index, group in enumerate(result['records']):
        base.require(group['schedule_epoch'] == index // 32 + 1, 'record schedule epoch')
    base.require(min(row['MemAvailable'] for row in result['resources']) >= 22, 'runtime reserve')
    return dict(update32_exact=True, adam_continuity_checked_by_final_state=True,
                schedule_epochs=2, final_only_saved_epoch=2,
                independent_training_loss_recomputed=False, independent_model_forward=False)


def main():
    approval_path = Path(sys.argv[sys.argv.index('--approval') + 1]).resolve()
    captured = {}
    original_read = base.read
    def read(path):
        value = original_read(path)
        path = Path(path).resolve()
        if path == approval_path:
            captured['approval'] = value
        elif 'approval' in captured:
            output = Path(captured['approval']['planned_output']).resolve()
            if path.parent == output and path.name in ('result.json', 'training_protocol.json', 'dual_model_manifest.json'):
                captured[path.name] = value
        return value
    base.read = read
    base.main()
    approval = captured['approval']
    output = Path(approval['planned_output'])
    historical = approval['historical_b_result']
    base.require(base.sha(historical['path']) == historical['sha256'], 'historical B result SHA')
    comparison_path = output.with_name(output.name + '_update32_comparison.json')
    evidence = check_extension(captured['result.json'], captured['training_protocol.json'],
                               captured['dual_model_manifest.json'], original_read(comparison_path),
                               original_read(historical['path']))
    evidence['comparison_sha256'] = base.sha(comparison_path)
    print(json.dumps(evidence, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
