"""Write a training recipe from configs/fsi_graphnet.json.

  python3 tools/make_recipe.py <root> <run_id> <code_tag> [<init checkpoint, relative to experiments/runs> [warm|resume]]

Output: <root>/experiments/recipes/<run_id>.json (input of tools/launch_train.py).
"""
import json, sys
from pathlib import Path

root, run_id, tag = sys.argv[1:4]
init = {'run_checkpoint': sys.argv[4], 'mode': sys.argv[5] if len(sys.argv) > 5 else 'warm'} if len(sys.argv) > 4 else None
c = json.loads((Path(__file__).resolve().parents[1] / 'configs/fsi_graphnet.json').read_text()); t = c['train']
config = {k: v for k, v in t.items() if k not in ('split', 'ranks')}; config['model'] = c['model']
recipe = {'run_id': run_id, 'code_tag': tag, 'split': t['split'], 'roles': ['TRAIN'], 'ranks': t['ranks'],
          'entry': 'fsi_mgn_train.py', 'init': init, 'config': config}
out = Path(root) / 'experiments/recipes' / (run_id + '.json'); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(recipe, indent=1)); print(out)
