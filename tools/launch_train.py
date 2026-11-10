#!/usr/bin/env python3
"""Create and start ONE training run (fsi_mgn_train.py) on this host in a container.

  python3 launch_train.py <project_root> <recipe.json>

recipe: {run_id, code_tag, split, roles:[TRAIN], ranks, init:{model_id | run_checkpoint, mode}, parent_model_id?, config:{...}}
* source = copy of code_releases/<code_tag> (read-only); SOURCE_MANIFEST verified by the job itself.
* train_cases come ONLY from the split file's TRAIN role, with its SHAs; each data file is re-hashed on this host.
* views: hard links under data/views/<split>/ (created if missing) -> mounted read-only at /views.
* run dir: experiments/runs/<run_id>/{source, contracts/config.json, results, control}; registered in experiments/index.json.
Refuses to overwrite an existing run dir. Host python only (no torch).
"""
import hashlib, json, os, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

IMAGE = os.environ.get('FSI_IMAGE', 'fluid-control-physicsnemo:2.2.2')   # PhysicsNeMo 26.06 container (torch, CUDA)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def main(root, recipe_path):
    root = Path(root); r = json.loads(Path(recipe_path).read_text())
    run = root / 'experiments/runs' / r['run_id']
    if run.exists(): raise SystemExit('run dir exists: ' + str(run))
    split = json.loads((root / 'data/splits' / (r['split'] + '.json')).read_text())
    cases = []
    views = root / 'data/views' / r['split']
    if r.get('roles', ['TRAIN']) != ['TRAIN']: raise SystemExit('training uses the TRAIN role of the split only')
    for role in ['TRAIN']:
        for c in split['roles'][role]:
            src = root / 'data/processed' / c['dataset'] / c['name']
            for f, key in (('sequence.npz', 'sequence_sha256'), ('report.json', 'report_sha256')):
                if sha(src / f) != c[key]: raise SystemExit('data SHA mismatch on this host: %s/%s' % (c['name'], f))
                dst = views / c['name'] / f; dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists(): os.link(src / f, dst)
            cases.append({'name': c['name'], 'sequence_sha256': c['sequence_sha256'], 'report_sha256': c['report_sha256']})
    init = r.get('init')   # None: train from scratch
    ck = None
    if init and 'model_id' in init:
        ck = root / 'models' / init['model_id'] / 'checkpoint.pt'
    elif init:
        ck = root / 'experiments/runs' / init['run_checkpoint']
    cfg = dict(r['config'], run_id=r['run_id'], code_tag=r['code_tag'], split=r['split'], train_cases=cases,
               init=None if ck is None else {'path': '/init/' + ck.name, 'sha256': sha(ck), 'mode': init['mode'], 'source': str(ck)})
    for d in ('contracts', 'results', 'control'): (run / d).mkdir(parents=True)
    shutil.copytree(root / 'code_releases' / r['code_tag'], run / 'source')
    (run / 'contracts/config.json').write_text(json.dumps(cfg, indent=1))
    cache = root / 'cache/arrays'; cache.mkdir(parents=True, exist_ok=True)   # host-level, content-addressed, deletable
    uid = '%d:%d' % (os.getuid(), os.getgid())
    cmd = ['docker', 'run', '-d', '--name', r['run_id'], '--gpus', 'all', '--ipc=host', '--user', uid,
           '-e', 'HOME=/tmp', '-e', 'USER=fsi', '-e', 'CUBLAS_WORKSPACE_CONFIG=:4096:8', '-e', 'PYTHONPATH=/work/src:/work',
           '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONFAULTHANDLER=1', '-e', 'OMP_NUM_THREADS=2', '-e', 'OPENBLAS_NUM_THREADS=2',
           '-e', 'FSI_ARRAY_CACHE=/cache', '-v', '%s:/cache' % cache,
           '-v', '%s:/work:ro' % (run / 'source'), '-v', '%s:/contracts:ro' % (run / 'contracts'), '-v', '%s:/results' % (run / 'results'),
           '-v', '%s:/control' % (run / 'control'), '-v', '%s:/views:ro' % views]
    if ck is not None: cmd += ['-v', '%s:/init/%s:ro' % (ck, ck.name)]
    cmd += ['-w', '/work', IMAGE, 'torchrun', '--standalone', '--nproc_per_node=%d' % int(r['ranks']), r.get('entry', 'fsi_mgn_train.py')]
    (run / 'launch.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'host': os.uname().nodename,
                                                  'recipe': r, 'docker': cmd}, indent=1))
    cid = subprocess.check_output(cmd, text=True).strip()
    idx = root / 'experiments/index.json'; data = json.loads(idx.read_text()) if idx.exists() else {}
    runs = data.setdefault('runs', {}) if isinstance(data, dict) else {}
    runs[r['run_id']] = {'kind': 'train', 'host': os.uname().nodename, 'run_dir': str(run), 'code_tag': r['code_tag'], 'split': r['split'],
                         'init': None if ck is None else str(ck), 'entry': r.get('entry', 'fsi_mgn_train.py'), 'parent_model_id': r.get('parent_model_id'), 'status': 'running', 'container': cid[:12],
                         'started_utc': datetime.now(timezone.utc).isoformat()}
    data['updated_utc'] = datetime.now(timezone.utc).isoformat()
    idx.write_text(json.dumps(data, indent=1))
    print('started', r['run_id'], cid[:12])


if __name__ == '__main__':
    main(*sys.argv[1:3])
