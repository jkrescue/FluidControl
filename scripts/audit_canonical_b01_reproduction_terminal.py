"""Reuse the prior canonical CFD raw audit; no model/CFD execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

repo = Path('/workspace/fluid_control')
unit = 'fluid-control-canonical-reproduce-b01-20261007.service'
props = dict(line.split('=', 1) for line in subprocess.check_output(
    ['systemctl', '--user', 'show', unit, '-p', 'InvocationID', '-p', 'MainPID',
     '-p', 'ExecMainStatus', '-p', 'Result'], text=True).splitlines())
assert props == {'InvocationID': 'f6c3fc3464074493b19ea5418ddfada5',
                 'MainPID': '0', 'ExecMainStatus': '0', 'Result': 'success'}
os.environ.update(
    CFD_AUDIT_OUTPUT='artifacts/canonical_b01_reproduction_20261007',
    CFD_AUDIT_APPROVAL='docs/CANONICAL_B01_REPRODUCTION_APPROVAL_20261007.json',
    CFD_REFERENCE='artifacts/p064_b_symmetry_canonical_ppo_b01_long_cfd_20261007/result.json',
    CFD_START='130', CANONICAL_CFD='1')
base = repo / 'scripts/audit_canonical_cfd_raw.py'
assert hashlib.sha256(base.read_bytes()).hexdigest() == 'cef05af9e430072300020a819ec320b328abf4754197b00701537fda018df2d8'
exec(compile(base.read_text(), str(base), 'exec'))
assert sha(approval) == 'f10547240d5a89430f2967bfb9859868fc00ee625447fcc56ce68b2b93b4152d'
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
                  'historical_E085_repeatability':repeat}))
