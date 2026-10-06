"""G-policy b01: reuse prior raw audit; no model/CFD execution.

The retained B is a matched reference, not an expected identical trajectory.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

repo = Path('/workspace/fluid_control')
unit = 'fluid-control-p064-g-symmetry-canonical-b01-cfd-20261007.service'
props = dict(line.split('=', 1) for line in subprocess.check_output(
    ['systemctl', '--user', 'show', unit, '-p', 'InvocationID', '-p', 'MainPID',
     '-p', 'ExecMainStatus', '-p', 'Result'], text=True).splitlines())
assert props == {'InvocationID': 'c14a66a1464a4919a3904c1f4d2efb2d',
                 'MainPID': '0', 'ExecMainStatus': '0', 'Result': 'success'}
os.environ.update(
    CFD_AUDIT_OUTPUT='artifacts/p064_g_symmetry_canonical_b01_cfd_20261007',
    CFD_AUDIT_APPROVAL='docs/P064_G_SYMMETRY_CANONICAL_B01_CFD_APPROVAL_20261007.json',
    CFD_REFERENCE='artifacts/p064_b_symmetry_canonical_ppo_b01_long_cfd_20261007/result.json',
    CFD_START='130', CANONICAL_CFD='1')
base = Path('/tmp/audit_b02_base_readonly.py')
assert hashlib.sha256(base.read_bytes()).hexdigest() == 'cef05af9e430072300020a819ec320b328abf4754197b00701537fda018df2d8'
exec(compile(base.read_text(), str(base), 'exec'))
assert sha(approval) == 'f72cace2befb6aef72ed6878ec83d675cc92b4992ec0b9e6e45cd38be97bbb48'
progress = read(P / 'progress.json')
assert progress['completed_cycles'] == 800 and progress['rows'] == r['rows']
for folder, files in s['source_restart_tree_sha256'].items():
    for name, digest in files.items():
        assert sha(repo / s['source_restart'] / folder / name) == digest
print(json.dumps({'terminal_unit': props, 'progress_sha256': sha(P/'progress.json'),
                  'source_restart_hashes_verified': sum(map(len, s['source_restart_tree_sha256'].values())),
                  'historical_E085_result_sha256': sha(R/os.environ['CFD_REFERENCE'])}))

# Additional feedback continuity and raw endpoint checks, not covered by base.
for i in range(1, 800):
    assert r['rows'][i]['input_observation'] == r['rows'][i-1]['output_observation']
endpoint_max = 0.
for i, row in enumerate(r['rows']):
    j = (i+1)*20-1
    expected = np.asarray([arrays['ppo','front'][j,1], arrays['ppo','front'][j,2],
                           arrays['ppo','rear'][j,1], arrays['ppo','rear'][j,2]], dtype=np.float32)
    actual = np.asarray(row['output_observation'][64:68], dtype=np.float32)
    endpoint_max = max(endpoint_max, float(np.max(np.abs(actual-expected))))
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-7)
repeat = {}
for key in ('input_observation', 'output_observation', 'applied_omega', 'requested_omega'):
    a = np.asarray([row[key] for row in r['rows']], dtype=np.float64)
    b = np.asarray([row[key] for row in trained['rows']], dtype=np.float64)
    repeat[key] = {'exact': bool(np.array_equal(a,b)), 'maximum_absolute_difference': float(np.max(np.abs(a-b)))}
print(json.dumps({'feedback_continuity_exact_links':799,
                  'raw_force_endpoint_to_observation_max_fp32_difference':endpoint_max,
                  'matched_retained_B_trajectory_difference_not_bitwise_requirement':repeat}))
