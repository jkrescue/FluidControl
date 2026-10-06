"""Route one separately approved P029 cache pass to exclusive r1-r4 evidence."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path('/workspace/fluid_control')
HELPER = ROOT / 'scripts/advise_p026_verified_train_cache_once.py'
HELPER_SHA = '98efe4a3c7fd08268d82cac9be97b79f3eadfe69217d7f1943b250f28f43ae2c'
OUTPUT = ROOT / 'artifacts/fcp029_cache_advice_20261006/cache_advice_20261006_r1.jsonl'
RECEIPT_NAMES = ('cache_advice_20261006_r1.jsonl', 'cache_advice_20261006_r2.jsonl', 'cache_advice_20261006_r3.jsonl', 'cache_advice_20261006_r4.jsonl')


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--receipt-name', choices=RECEIPT_NAMES, default=RECEIPT_NAMES[0])
    return parser.parse_args(argv)


def prepare_receipt(receipt_name):
    if receipt_name not in RECEIPT_NAMES:
        raise ValueError('fixed r1-r4 receipt only')
    destination = OUTPUT.with_name(receipt_name)
    for component in (destination, *destination.parents):
        if component.is_symlink():
            raise ValueError('receipt path symlink')
    if destination.exists():
        raise FileExistsError(destination)
    if receipt_name == RECEIPT_NAMES[0]:
        destination.parent.mkdir(exist_ok=False)
    elif not destination.parent.is_dir():
        raise ValueError('r2-r4 require the existing nonsymlink P029 receipt directory')
    # Actual receipt remains exclusively opened with x by the unchanged helper.
    return destination


def main():
    args = arguments()
    if HELPER.is_symlink() or hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
        raise ValueError('reviewed cache helper differs')
    spec = importlib.util.spec_from_file_location('p029_verified_cache_helper', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only receipt routing changes. The44 files, same-fd SHA checks, guards,
    # POSIX_FADV_DONTNEED call and read-only descriptors remain unchanged.
    module.OUTPUTS = {1: OUTPUT, 4: OUTPUT}
    if args.execute:
        prepare_receipt(args.receipt_name)
    sys.argv = [str(HELPER), '--history-k', '1', '--receipt-name', args.receipt_name]
    if args.execute:
        sys.argv.append('--execute')
    module.main()


if __name__ == '__main__':
    main()
