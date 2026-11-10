"""Write a training recipe from a configuration file.

  python3 tools/make_recipe.py [--config configs/<name>.json] <root> <run_id> <code_tag> \
      [<init checkpoint, relative to experiments/runs> [warm|resume]]

The default configuration is configs/fsi_graphnet.json. A configuration may name a "base" file in the same directory;
its "train" and "model" entries then override the base recursively.
Output: <root>/experiments/recipes/<run_id>.json (input of tools/launch_train.py).
"""
import json, sys
from pathlib import Path


def merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path):
    c = json.loads(path.read_text())
    return merge(load(path.parent / c.pop('base')), c) if 'base' in c else c


args = sys.argv[1:]
cfg = Path(__file__).resolve().parents[1] / 'configs/fsi_graphnet.json'
if args[0] == '--config':
    cfg = Path(args[1]).resolve(); args = args[2:]
root, run_id, tag = args[:3]
init = {'run_checkpoint': args[3], 'mode': args[4] if len(args) > 4 else 'warm'} if len(args) > 3 else None
c = load(cfg); t = c['train']
config = {k: v for k, v in t.items() if k not in ('split', 'ranks')}; config['model'] = c['model']
recipe = {'run_id': run_id, 'code_tag': tag, 'split': t['split'], 'roles': ['TRAIN'], 'ranks': t['ranks'],
          'entry': 'fsi_mgn_train.py', 'init': init, 'config': config}
out = Path(root) / 'experiments/recipes' / (run_id + '.json'); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(recipe, indent=1)); print(out)
