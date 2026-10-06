"""Reuse the reviewed same-descriptor cache helper with one P028-only receipt."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path('/workspace/fluid_control')
HELPER = ROOT / 'scripts/advise_p026_verified_train_cache_once.py'
HELPER_SHA = '98efe4a3c7fd08268d82cac9be97b79f3eadfe69217d7f1943b250f28f43ae2c'
OUTPUT = ROOT / 'artifacts/fcp028_cache_advice_20261006/cache_advice_20261006_r1.jsonl'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if HELPER.is_symlink() or hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
        raise ValueError('reviewed cache helper differs')
    spec = importlib.util.spec_from_file_location('p028_verified_cache_helper', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only receipt routing changes. The44 files, same-fd SHA checks, guards,
    # POSIX_FADV_DONTNEED call and read-only descriptors remain unchanged.
    module.OUTPUTS = {1: OUTPUT, 4: OUTPUT}
    if args.execute:
        OUTPUT.parent.mkdir(exist_ok=False)
    sys.argv = [str(HELPER), '--history-k', '1', '--receipt-name', OUTPUT.name]
    if args.execute:
        sys.argv.append('--execute')
    module.main()

if __name__ == '__main__':
    main()
