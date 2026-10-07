"""Identity-only extension of the reviewed saved-evidence audit; no model loads."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

BASE_SHA = 'cfd12670b4d6c8f9c8192bea55cec23915113ebad581bdb1e0bc9c9e2a8d939a'
PROFILE = (
    'P064_ABSOLUTE64_SYMMETRY_CANONICAL_H5_32768_PPO_TRAINING_COMPLETE_NOT_ADMISSION',
    'FC_P064_ABSOLUTE64_ARM_B_CONTROLLED_AERO_FORCE_FNO',
    '2e3c778b9cec039487602eb38b54f78f42b0a2e0fb8e467790bc585aedc0545c',
)

def load_base(path):
    path = Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == BASE_SHA
    spec = importlib.util.spec_from_file_location('canonical_ppo_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROFILES['ABSOLUTE64'] = PROFILE
    return module

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, default=Path('scripts/audit_canonical_ppo_terminal.py'))
    p.add_argument('--approval', type=Path, required=True)
    p.add_argument('--approval-sha256', required=True)
    p.add_argument('--unit', required=True)
    p.add_argument('--invocation', required=True)
    args = p.parse_args()
    args.profile = 'ABSOLUTE64'
    print(json.dumps(load_base(args.base).audit(args), indent=2))
