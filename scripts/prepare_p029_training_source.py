"""Source-only P029 preparation. No data/model access or execution authorization."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('/workspace/fluid_control')
BASE = ROOT / 'artifacts/fcp028_resource_source_20261006_r2_immutable'
BASE_MANIFEST_SHA = '72c0513e03c154b032f162d9c4c3cbd5ddfad7c478694f4e652aeb949759012c'
OLD_SPEC_SHA = '655f4d924036ef1e23857c4bc1892b8f0bbce40af1a1b22f8bd4657eca097837'
OUTPUT = ROOT / 'artifacts/fcp029_training_source_20261006_immutable'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_checked(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError('regular source required: ' + str(path))
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError('source identity differs: ' + str(path))
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--runner-sha256', required=True)
    parser.add_argument('--objective', type=Path, required=True)
    parser.add_argument('--objective-sha256', required=True)
    parser.add_argument('--materialize', action='store_true')
    args = parser.parse_args()
    sources = json.loads(read_checked(BASE / 'source_manifest.json', BASE_MANIFEST_SHA))
    if len(sources) != 421:
        raise ValueError('expected fixed421 source basis')
    contents = {}
    for name, expected in sources.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('source path escape')
        contents[name] = read_checked(BASE / name, expected)
    additions = {
        'scripts/train_p029_control_aware_flow.py': (args.runner, args.runner_sha256),
        'scripts/p029_control_aware_flow_objective.py': (args.objective, args.objective_sha256),
    }
    for name, (path, expected) in additions.items():
        if name in sources:
            raise ValueError('new source unexpectedly exists in basis')
        contents[name] = read_checked(path, expected)
        sources[name] = expected
    old = json.loads(read_checked(ROOT / 'docs/FC_P028_TRAIN_EXECUTION_APPROVAL_20261006.json', OLD_SPEC_SHA))
    config = read_checked(BASE / 'training_config.yaml', old['config']['sha256'])
    if contents.get('training_config.yaml') != config:
        raise ValueError('source closure must contain the identical training config')
    definition = importlib.util.spec_from_file_location('p029_preparation_protocol', args.runner)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)  # Reviewed module: stdlib only, no execution at import.
    spec = {key: old[key] for key in ('parent_manifest', 'train_audit', 'data', 'resources')}
    spec.update(status='FC_P029_PREPARATION_ONLY_NOT_APPROVED', mode='scales',
                protocol=module.protocol(), source_root=str(OUTPUT), source_sha256=sources,
                config=dict(path=str(OUTPUT / 'training_config.yaml'), sha256=old['config']['sha256']),
                preparation_notes='Source-only preparation; no inherited P028 execution approval. Lead must review final formal outcome and issue a new per-mode approval.')
    if not args.materialize:
        print(json.dumps(dict(status='P029_SOURCE_DRY_RUN_ONLY', count=len(sources), output=str(OUTPUT))))
        return
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir()
    for name, raw in contents.items():
        path = OUTPUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    for name, value in [('source_manifest.json', sources), ('input_spec_preparation_only.json', spec)]:
        with (OUTPUT / name).open('x') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
    receipt = dict(status='P029_SOURCE_PREPARATION_ONLY_NOT_EXECUTION', source_count=len(sources),
                   basis_manifest_sha256=BASE_MANIFEST_SHA,
                   source_manifest_sha256=digest((OUTPUT / 'source_manifest.json').read_bytes()),
                   pending_spec_sha256=digest((OUTPUT / 'input_spec_preparation_only.json').read_bytes()),
                   helper_sha256=digest(Path(__file__).read_bytes()))
    with (OUTPUT / 'freeze_receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    for path in OUTPUT.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((p for p in OUTPUT.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        path.chmod(0o555)
    OUTPUT.chmod(0o555)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
