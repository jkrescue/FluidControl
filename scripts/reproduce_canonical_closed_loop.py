"""Read-only default preflight / explicit, newly approved E085 reproduction launch."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

BASE = 'docs/P064_B_SYMMETRY_CANONICAL_B01_CFD_APPROVAL_20261007.json'
BASE_SHA = 'ee010bbe1932e0b48b2f2e90a1b9dd77d463f86dad6367e0c0dd49a46f0b45f1'
DRIVER = 'artifacts/p064_symmetry_canonical_b01_cfd_source_20261007_immutable/run_p064_symmetry_canonical_b_ppo_b01_long_cfd.py'
ALLOWED = {'output', 'lead_statement', 'interpretation', 'reproduction'}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def inside(repo, path):
    p = (repo / path).resolve()
    require(p.is_relative_to(repo), 'path escapes repository')
    return p

def compare_approval(base, spec, unit):
    require(set(spec) - set(base) <= {'reproduction'}, 'unexpected new fields')
    require({k: v for k, v in base.items() if k not in ALLOWED} ==
            {k: v for k, v in spec.items() if k not in ALLOWED}, 'scientific/resource binding changed')
    require(spec['output'] != base['output'], 'historical output cannot be reused')
    require(re.fullmatch(r'fluid-control-canonical-reproduce-[a-z0-9-]+\.service', unit or '') is not None,
            'unique reproduction unit required')
    r = spec.get('reproduction', {})
    require(set(r) == {'base_approval_sha256', 'unit', 'execution_authorized'}
            and r['base_approval_sha256'] == BASE_SHA and r['unit'] == unit
            and type(r['execution_authorized']) is bool, 'explicit reproduction binding required')
    require(spec['lead_statement'] != base['lead_statement'], 'historical one-run approval is not new authority')

def check_files(repo, spec):
    driver = inside(repo, DRIVER)
    require(sha(driver) == spec['driver_sha256'], 'driver SHA mismatch')
    for path, digest in spec['source_files'].items():
        require(sha(inside(repo, path)) == digest, 'source SHA mismatch: ' + path)
    for item in spec['inputs'].values():
        require(sha(inside(repo, item['path'])) == item['sha256'], 'input SHA mismatch: ' + item['path'])
    restart = inside(repo, spec['source_restart'])
    for folder, files in spec['source_restart_tree_sha256'].items():
        actual = {str(p.relative_to(restart / folder)): sha(p) for p in (restart / folder).rglob('*') if p.is_file()}
        require(actual == files, 'restart tree differs: ' + folder)
    training = json.loads(inside(repo, spec['inputs']['training_approval']['path']).read_text())
    overlay = repo / '.runtime/exploratory-h5-ppo-py312'
    for path, digest in training['runtime_sources'].items():
        if Path(path).is_relative_to(overlay):
            require(sha(path) == digest, 'policy runtime SHA mismatch: ' + path)
    return driver

def command(repo, spec, approval, digest, unit, driver):
    argv = ['systemd-run', '--user', '--unit=' + unit, '--property=Type=exec',
            '--property=RemainAfterExit=yes', '--property=TasksMax=512', '--working-directory=' + str(repo)]
    argv += ['--property=' + k + '=' + str(v) for k, v in spec['required_systemd'].items()]
    env = dict(spec['runtime_env'], OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    argv += ['--setenv=' + k + '=' + v for k, v in env.items()]
    return argv + [str(repo / '.venv-curator-py312/bin/python'), str(driver), '--spec', str(approval),
                   '--spec-sha256', digest, '--execute']

def execution_gate(spec, execute):
    require(not execute or spec['reproduction']['execution_authorized'] is True,
            'new reproduction execution is not authorized')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--approval', type=Path)
    ap.add_argument('--approval-sha256')
    ap.add_argument('--unit')
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    repo = args.repo.resolve()
    base_path = repo / BASE
    require(sha(base_path) == BASE_SHA, 'historical approval identity differs')
    base = json.loads(base_path.read_text())
    require(Path(base['repo']).resolve() == repo, 'run on original pinned Spark repository')
    spec, approval, digest = base, base_path, BASE_SHA
    if args.approval:
        approval = inside(repo, args.approval)
        digest = sha(approval)
        require(digest == args.approval_sha256, 'new approval SHA mismatch')
        spec = json.loads(approval.read_text())
        compare_approval(base, spec, args.unit)
        execution_gate(spec, args.execute)
        output = inside(repo, spec['output'])
        require(output.is_relative_to(repo / 'artifacts') and output != repo / 'artifacts', 'output must be artifacts child')
        require(not output.exists(), 'output already exists; no restart or overwrite')
        load = subprocess.check_output(['systemctl', '--user', 'show', args.unit, '-p', 'LoadState', '--value'], text=True).strip()
        require(load == 'not-found', 'unit already exists; no duplicate launch')
    else:
        require(not args.execute, '--execute requires a separately approved new spec and SHA/unit')
    driver = check_files(repo, spec)
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))) * 1024
    require(available >= 50 * 2**30, 'startup MemAvailable below 50 GiB')
    require(shutil.disk_usage(repo).free >= 20 * 2**30, 'disk free below 20 GiB')
    image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', spec['openfoam_image_id']], text=True).strip()
    require(image == spec['openfoam_image_id'], 'OpenFOAM image differs')
    # Import/metadata validation only: no policy deserialization and no CFD.
    code = 'import importlib.util,json,sys; s=importlib.util.spec_from_file_location("reviewed",sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.validate_spec(json.load(open(sys.argv[2])))'
    subprocess.run([str(repo / '.venv-curator-py312/bin/python'), '-B', '-c', code, str(driver), str(approval)], check=True)
    summary = {'status': 'PREFLIGHT_PASS_NOT_RUNNING', 'phase': 'b01', 'steps': 800,
               'approval_sha256': digest, 'driver_sha256': sha(driver), 'output': spec['output'],
               'resource_contract': spec['resource_contract'], 'unit': args.unit}
    if args.approval:
        summary['argv'] = command(repo, spec, approval, digest, args.unit, driver)
    print(json.dumps(summary, indent=2), flush=True)
    if args.execute:
        subprocess.run(summary['argv'], check=True)
        print(subprocess.check_output(['systemctl', '--user', 'show', args.unit, '-p', 'InvocationID', '-p', 'MainPID', '-p', 'SubState'], text=True))
        print('Launch is not scientific completion. Follow progress.json/result.json; no automatic retry.')

if __name__ == '__main__':
    main()
