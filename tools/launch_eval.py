#!/usr/bin/env python3
"""Create and start ONE evaluation run (fsi_mgn_eval.py) of a checkpoint on this host in a container.

  python3 launch_eval.py <project_root> <eval_recipe.json>

recipe: {eval_id, code_tag, split, roles: [VAL | DEV | HELDOUT], cases?: [names],
         checkpoint: "<run_id>/results/checkpoint_uNNNNN.pt" | model_id, offsets | starts, steps, save_frames?}
* cases come ONLY from the split file (roles, optional name filter), with their SHAs; every data file is re-hashed here.
  HELDOUT requires recipe.final == true (final acceptance only).
* views: hard links under data/views/<split>__eval/ (kept apart from the TRAIN-only training views), mounted read-only.
* the checkpoint is hard-linked into the eval run dir (pinned by SHA); the model config is read from the checkpoint.
* run dir: experiments/runs/<eval_id>/{source, contracts/config.json, inputs/checkpoint.pt, results}; registered in index.json.
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
    run = root / 'experiments/runs' / r['eval_id']
    if run.exists(): raise SystemExit('run dir exists: ' + str(run))
    if 'HELDOUT' in r['roles'] and not r.get('final'): raise SystemExit('HELDOUT is for the final acceptance only')
    split = json.loads((root / 'data/splits' / (r['split'] + '.json')).read_text())
    views = root / 'data/views' / (r['split'] + '__eval'); cases = []
    for role in r['roles']:
        for c in split['roles'][role]:
            if r.get('cases') and c['name'] not in r['cases']: continue
            src = root / 'data/processed' / c['dataset'] / c['name']
            for f, key in (('sequence.npz', 'sequence_sha256'), ('report.json', 'report_sha256')):
                if sha(src / f) != c[key]: raise SystemExit('data SHA mismatch on this host: %s/%s' % (c['name'], f))
                dst = views / c['name'] / f; dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists(): os.link(src / f, dst)
            cases.append({'name': c['name'], 'sequence_sha256': c['sequence_sha256'], 'report_sha256': c['report_sha256'], 'role': role})
    if r.get('cases') and len(cases) != len(r['cases']): raise SystemExit('requested case not in the split roles')
    ck = root / 'models' / r['model_id'] / 'checkpoint.pt' if 'model_id' in r else root / 'experiments/runs' / r['checkpoint']
    for d in ('contracts', 'results', 'inputs'): (run / d).mkdir(parents=True)
    os.link(ck, run / 'inputs/checkpoint.pt')
    shutil.copytree(root / 'code_releases' / r['code_tag'], run / 'source')
    cfg = {'split': r['split'], 'cases': cases,
           'model_checkpoint': '/ckpt/checkpoint.pt', 'model_checkpoint_sha256': sha(run / 'inputs/checkpoint.pt'),
           'checkpoint_source': str(ck), 'code_tag': r['code_tag']}
    cfg.update({k: r[k] for k in ('offsets', 'starts', 'steps', 'save_frames') if k in r})
    (run / 'contracts/config.json').write_text(json.dumps(cfg, indent=1))
    cache = root / 'cache/arrays'; cache.mkdir(parents=True, exist_ok=True)
    uid = '%d:%d' % (os.getuid(), os.getgid())
    cmd = ['docker', 'run', '-d', '--name', r['eval_id'], '--gpus', 'all', '--ipc=host', '--user', uid,
           '-e', 'HOME=/tmp', '-e', 'USER=fsi', '-e', 'CUBLAS_WORKSPACE_CONFIG=:4096:8', '-e', 'PYTHONPATH=/work/src:/work',
           '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=2', '-e', 'OPENBLAS_NUM_THREADS=2',
           '-e', 'FSI_ARRAY_CACHE=/cache', '-v', '%s:/cache' % cache,
           '-v', '%s:/work:ro' % (run / 'source'), '-v', '%s:/contracts:ro' % (run / 'contracts'), '-v', '%s:/results' % (run / 'results'),
           '-v', '%s:/views:ro' % views, '-v', '%s:/ckpt:ro' % (run / 'inputs'),
           '-w', '/work', IMAGE, 'python', '-B', r.get('entry', 'fsi_mgn_eval.py')]
    (run / 'launch.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'host': os.uname().nodename,
                                                  'recipe': r, 'docker': cmd}, indent=1))
    cid = subprocess.check_output(cmd, text=True).strip()
    idx = root / 'experiments/index.json'; data = json.loads(idx.read_text()) if idx.exists() else {}
    data.setdefault('runs', {})[r['eval_id']] = {'kind': 'eval', 'host': os.uname().nodename, 'run_dir': str(run), 'code_tag': r['code_tag'],
                                                 'split': r['split'], 'checkpoint': str(ck), 'status': 'running', 'container': cid[:12],
                                                 'started_utc': datetime.now(timezone.utc).isoformat()}
    data['updated_utc'] = datetime.now(timezone.utc).isoformat()
    idx.write_text(json.dumps(data, indent=1))
    print('started', r['eval_id'], cid[:12])


if __name__ == '__main__':
    main(*sys.argv[1:3])
