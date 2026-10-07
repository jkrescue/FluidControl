"""Preparation-only source regressions; never imports the live CFD audit."""
import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'scripts' / 'audit_absolute64_cfd_terminal.py'


def test_identity_only_delta_from_g_audit_before_descriptive_g_extension():
    new = SOURCE.read_text()
    old = (REPO / 'scripts' / 'audit_g_cfd_terminal.py').read_text()
    prefix = new[:new.index('\ndef descriptive_difference')]
    replacements = {
        'Absolute64-policy': 'G-policy',
        ' and exploratory G runs are matched descriptive references': ' is a matched reference',
        'fluid-control-p064-absolute64-symmetry-canonical-b01-cfd-20261007.service':
            'fluid-control-p064-g-symmetry-canonical-b01-cfd-20261007.service',
        '6fa0baeb90034371b3b49f3d8d51192a': 'c14a66a1464a4919a3904c1f4d2efb2d',
        'artifacts/p064_absolute64_symmetry_canonical_b01_cfd_20261007':
            'artifacts/p064_g_symmetry_canonical_b01_cfd_20261007',
        'docs/P064_ABSOLUTE64_SYMMETRY_CANONICAL_B01_CFD_APPROVAL_20261007.json':
            'docs/P064_G_SYMMETRY_CANONICAL_B01_CFD_APPROVAL_20261007.json',
        'd6303082b0f97087961a9171505a2b4289623eb5f634f2502ac4f7d65639143b':
            'f72cace2befb6aef72ed6878ec83d675cc92b4992ec0b9e6e45cd38be97bbb48',
    }
    for a, b in replacements.items():
        assert prefix.count(a) == 1
        prefix = prefix.replace(a, b)
    # The reference G audit has its descriptive B comparison after this common prefix.
    old_prefix = old[:old.index('\nrepeat = {}')]
    new_tree, old_tree = ast.parse(prefix), ast.parse(old_prefix)
    new_tree.body.pop(0)
    old_tree.body.pop(0)
    assert ast.dump(new_tree) == ast.dump(old_tree)


def test_terminal_gate_precedes_raw_read_and_references_are_descriptive():
    source = SOURCE.read_text()
    assert source.index('assert props ==') < source.index('exec(compile(')
    assert "'MainPID': '0'" in source and "'ExecMainStatus': '0'" in source
    assert 'cef05af9e430072300020a819ec320b328abf4754197b00701537fda018df2d8' in source
    assert 'assert np.array_equal(a,b)' not in source
    assert 'matched_retained_B_descriptive_difference' in source
    assert 'matched_exploratory_G_descriptive_difference' in source


def test_fixed_windows_are_in_pinned_raw_base_not_selected_live():
    source = SOURCE.read_text()
    assert "CFD_START='130'" in source
    assert "progress['completed_cycles'] == 800" in source
    assert "len(g['rows']) == 800" in source
