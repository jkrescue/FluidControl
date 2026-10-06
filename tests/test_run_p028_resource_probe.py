"""Independent synthetic launcher fixtures. No Docker or data operations."""
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load('review_p027_base', HERE / 'test_run_p027_short_horizon_diagnostic.py')
launcher = load('review_p028_launcher', HERE.parent / 'scripts/run_p028_resource_probe.py')
base.module = launcher
original_fixture = base.fixture


def fixture(tmp_path):
    path, payload = original_fixture(tmp_path)
    sources = payload.pop('source_files')
    root = Path(sources[0]['path']).parent.parent
    payload.update(status='FC_P028_EXECUTION_APPROVED', mode='resource-probe',
                   resources={'allocator_fraction': .06}, source_root=str(root),
                   source_sha256={str(Path(x['path']).relative_to(root)): x['sha256'] for x in sources},
                   parent_manifest={'path': payload['candidates']['1']['manifest']})
    # Retain synthetic extras solely for the inherited mocked SIGTERM test's guard lookup.
    payload['source_files'] = sources
    path.write_text(json.dumps(payload))
    return path, payload


base.fixture = fixture
for name in dir(base):
    if name.startswith('test_'):
        globals()[name] = getattr(base, name)


@pytest.mark.parametrize('mode,fraction', [('train', .06), ('resource-probe', .07)])
def test_resource_only_fixed_allocator(tmp_path, mode, fraction):
    path, payload = fixture(tmp_path)
    payload['mode'] = mode
    payload['resources']['allocator_fraction'] = fraction
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError):
        launcher.load_spec(path, base.digest(path))


def test_child_output_and_single_parent_mount(tmp_path):
    path, payload = fixture(tmp_path)
    output = tmp_path / 'output'
    command = launcher.create_command(payload, path, base.digest(path), output)
    assert command[command.index('--mode') + 1] == 'resource-probe'
    assert command[command.index('--output') + 1] == str(output / 'payload')
    mounts = launcher.readonly_mounts(payload, path)
    assert (Path(payload['parent_manifest']['path']).parent,) == tuple(
        source for source, _, _ in mounts if source.name == 'candidate1')
    assert all(source.name != 'candidate4' for source, _, _ in mounts)


def test_source_escape_rejected(tmp_path):
    path, payload = fixture(tmp_path)
    payload['source_sha256']['../escape.py'] = '0' * 64
    with pytest.raises(RuntimeError, match='escapes'):
        launcher.readonly_mounts(payload, path)

def test_training_mounts_match_unchanged_config_aliases(tmp_path):
    path, payload = fixture(tmp_path)
    mounts = launcher.readonly_mounts(payload, path)
    for family in ('base', 'train8', 'train16'):
        root = Path(payload['data'][family]['root'])
        for name in ('manifest.json', 'normalization.json', 'train'):
            assert (root / name, Path('/workspace') / family / name, True) in mounts
    assert all(target.name not in ('validation', 'frozen_test', 'test')
               for _, target, _ in mounts)
