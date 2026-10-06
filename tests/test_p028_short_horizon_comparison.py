"""Synthetic engineering fixtures only; no official model/data loads."""
import copy
import numpy as np
import pytest
import torch
import sys
from types import SimpleNamespace

import diagnose_p028_short_horizon_comparison as probe


class Flow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        raw = x.new_zeros((1, 7, *x.shape[-2:]))
        raw[:, :3] = 1
        return raw


class Aero(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.clone())
        raw = x.new_zeros((1, 7, *x.shape[-2:]))
        raw[:, 3:] = x[:, 3*(self.k-1):3*(self.k-1)+1] + x[:, -1:]
        return raw


def sample():
    return dict(states=torch.arange(48, 62, dtype=torch.float32)[:, None, None, None].expand(14, 3, 2, 2).clone(),
                actions=torch.arange(48, 62, dtype=torch.float32)/100,
                mask=torch.ones(1, 2, 2))


def prediction(value, *, fields=False):
    flow = Flow().eval()
    aeros = {k: Aero(k).eval() for k in (1, 4)}
    result = probe.predict_origin(
        flow, aeros, value, torch.zeros(4), torch.ones(4), return_flow_fields=fields
    )
    return flow, aeros, result


def test_shared_flow_and_exact_history_action_target_timing():
    value = sample()
    before = copy.deepcopy(value)
    flow, aeros, result = prediction(value)
    assert flow.calls == 10
    for k in (1, 4):
        assert len(aeros[k].inputs) == 20
        np.testing.assert_allclose(result[f'k{k}']['ar'][:, 0], np.arange(51, 61)+np.arange(52, 62)/100, rtol=1e-6)
    x = aeros[4].inputs[0]
    assert x[0, :12, 0, 0].tolist() == [48]*3+[49]*3+[50]*3+[51]*3
    torch.testing.assert_close(x[0, 13:, 0, 0], torch.arange(48, 53)/100)
    for key in value:
        assert torch.equal(value[key], before[key])


def test_future_truth_changes_h1_not_free_ar():
    original = sample()
    changed = copy.deepcopy(original)
    changed['states'][4:] += 100
    _, _, a = prediction(original)
    _, _, b = prediction(changed)
    for k in (1, 4):
        np.testing.assert_array_equal(a[f'k{k}']['ar'], b[f'k{k}']['ar'])
        assert not np.array_equal(a[f'k{k}']['h1'], b[f'k{k}']['h1'])


def test_flow_fields_use_same_ten_endpoints_and_separate_h1_ar():
    value = sample()
    flow, _, (forces, fields) = prediction(value, fields=True)
    assert flow.calls == 20
    assert set(forces) == {'k1', 'k4'}
    assert fields['h1'].shape == fields['ar'].shape == (10, 3, 2, 2)
    # H1 receives the true current state; AR recursively receives its own state.
    assert fields['h1'][:, 0, 0, 0].tolist() == list(range(52, 62))
    assert fields['ar'][:, 0, 0, 0].tolist() == list(range(52, 62))


def test_field_metrics_masked_normalized_and_shape_failclosed():
    truth = torch.ones(10, 3, 2, 2)
    prediction = truth.clone()
    prediction[:, 0, 0, 0] += 2
    mask = torch.tensor([[[[1., 0.], [0., 0.]]]])
    value = probe.field_metrics(prediction, truth, mask)
    assert value['normalized_channel_rmse'] == pytest.approx([2., 0., 0.])
    assert value['normalized_all_rmse'] == pytest.approx(2/np.sqrt(3))
    with pytest.raises(ValueError):
        probe.field_metrics(prediction[:9], truth, mask)


def test_summary_retains_same_origin_field_comparison_by_group():
    target = np.zeros((10, 4))
    force = probe.metrics(target, target, np.zeros((52, 4)), baseline())
    field = {
        arm: {
            'normalized_channel_rmse': [value, value, value],
            'normalized_channel_relative_l2': [value, value, value],
            'normalized_all_rmse': value,
        }
        for arm, value in [('parent_h1', 1.), ('parent_ar', 2.), ('p028_h1', 3.), ('p028_ar', 4.)]
    }
    rows = [dict(family='train8', phase='b02', metrics={'k1_h1': force},
                 normalized_field_metrics=field)]
    value = probe.summaries(rows)['family_phase:train8:b02']['normalized_field_metrics']
    assert value['aggregation'] == 'arithmetic_mean_case_metrics_not_pooled'
    assert value['arms']['parent_ar']['mean_case_normalized_all_rmse'] == 2
    assert value['arms']['p028_ar']['mean_case_normalized_channel_rmse'] == [4, 4, 4]


@pytest.mark.parametrize('bad', ['nonfinite', 'shape', 'training'])
def test_failclosed_inputs_and_mode(bad):
    value = sample()
    flow = Flow().eval()
    if bad == 'nonfinite':
        value['states'][0, 0, 0, 0] = float('nan')
    if bad == 'shape':
        value['states'] = value['states'][:-1]
    if bad == 'training':
        flow.train()
    with pytest.raises((ValueError, FloatingPointError)):
        probe.predict_origin(flow, {k: Aero(k).eval() for k in (1, 4)}, value, torch.zeros(4), torch.ones(4))


def baseline():
    return dict(total_drag=2., rear_cl_fluctuation_rms=1., source='synthetic fixture')


def test_ten_error_not_hidden_by_52_true_prefix():
    target = np.zeros((10, 4))
    pred = target.copy()
    pred[:, 3] = 1
    prefix = np.zeros((52, 4))
    result = probe.metrics(pred, target, prefix, baseline())
    assert result['ten_point']['rear_cl_signed_mean_error'] == 1
    assert result['mixed62']['mean_rear_cl']['error'] == pytest.approx(10/62)
    assert result['ten_point']['rear_cl_centered_residual_mse'] == 0
    assert result['ten_point']['absolute_rms_error'] == 0
    assert 'canonical_joint_gate_pass' not in str(result)
    assert np.count_nonzero(prefix) == 0


@pytest.mark.parametrize('shape', [(9,4), (11,4), (10,3)])
def test_metric_endpoint_shape_rejected(shape):
    with pytest.raises(ValueError):
        probe.metrics(np.zeros(shape), np.zeros((10,4)), np.zeros((52,4)), baseline())


def test_metric_nonfinite_rejected():
    pred = np.zeros((10,4))
    pred[0,0] = np.inf
    with pytest.raises(ValueError):
        probe.metrics(pred, np.zeros((10,4)), np.zeros((52,4)), baseline())


def test_invalid_recorded_slew_kept_and_cost_unavailable():
    result = probe.metrics(np.zeros((10,4)), np.zeros((10,4)), np.zeros((52,4)),
                           baseline(), omega=.5, delta_omega=.5)
    assert result['terminal_endpoint_cost']['available'] is False
    assert result['terminal_endpoint_cost']['delta_omega'] == .5
    assert 'violates' in result['terminal_endpoint_cost']['reason']


def test_cost_uses_actual_legal_action_and_no_admission_flag():
    result = probe.metrics(np.zeros((10,4)), np.zeros((10,4)), np.zeros((52,4)),
                           baseline(), omega=.3, delta_omega=.05)
    assert result['terminal_endpoint_cost']['available'] is True
    assert result['diagnostic_only'] is True


def test_invalid_cost_keeps_force_denominator():
    rows = []
    for delta in (.05, .5):
        value = probe.metrics(np.zeros((10,4)), np.zeros((10,4)), np.zeros((52,4)),
                              baseline(), omega=.3, delta_omega=delta)
        rows.append(dict(family='train16', phase='b00', metrics={'k1_ar': value}))
    summary = probe.summaries(rows)['overall44']['metrics']['k1_ar']['cost_subset']
    assert summary['valid_count'] == 1 and summary['invalid_count'] == 1
    assert summary['force_statistics_count'] == 2


def test_reader_shapes_exact_frames_normalization_and_times():
    visited = []
    class Reader:
        def __getitem__(self, index):
            visited.append(index)
            return dict(state=torch.full((3,2,2), float(index)), mask=torch.ones(1,2,2),
                omega=torch.tensor(float(index)/100), force=torch.full((4,),float(index)),
                time=torch.tensor(148.+index*.1)), {}
    dataset = SimpleNamespace(_reader=lambda _: Reader(), state_mean=torch.ones(3,1,1),
                              state_std=torch.full((3,1,1),2.), action_scale=.75)
    value = probe.read_origin(dataset, 0)
    assert visited == list(range(62))
    torch.testing.assert_close(value['states'][:,0,0,0], (torch.arange(48,62)-1)/2)
    assert value['forces'][:52,0].tolist() == list(range(52))
    assert value['forces'][52:,0].tolist() == list(range(52,62))
    assert value['physical_actions'][61] == float(torch.tensor(.61))


def test_execute_import_contract_uses_evaluator_config_loader(monkeypatch):
    sentinel = object()
    monkeypatch.setitem(sys.modules, 'train_tandem_fno', SimpleNamespace(build_model=sentinel))
    monkeypatch.setitem(sys.modules, 'evaluate_tandem_fno', SimpleNamespace(load_composed_config=sentinel))
    monkeypatch.setitem(sys.modules, 'fluid_control.tandem_datapipe', SimpleNamespace(TandemRolloutDataset=sentinel))
    values = probe.runtime_dependencies()
    assert values[0] is sentinel and values[1] is sentinel and values[3] is sentinel


@pytest.mark.parametrize(
    'status,reaches_runtime',
    [
        ('P028_MATCHED_H10_COMPARISON_EXECUTION_APPROVED', True),
        ('P028_MATCHED_H10_COMPARISON_PENDING_LEAD_APPROVAL', False),
        ('P027_OFFLINE_DIAGNOSTIC_EXECUTION_APPROVED', False),
    ],
)
def test_execute_status_matches_launcher_before_runtime_access(monkeypatch, tmp_path, status,
                                                               reaches_runtime):
    class RuntimeReached(RuntimeError):
        pass

    monkeypatch.setattr(
        probe,
        'runtime_dependencies',
        lambda: (_ for _ in ()).throw(RuntimeReached('runtime reached')),
    )
    with pytest.raises(RuntimeReached if reaches_runtime else ValueError):
        probe.execute({'status': status}, tmp_path / 'never-created.json')


def test_phase_is_authoritative_metadata_not_name():
    record = {'cases': {'misleading_b06': dict(split='train',phase='b00',env_index=0,
               expected_frames=129, control_dt=.1,episode=2,run_window=[148.,160.8])}}
    value = probe.train16_metadata('misleading_b06', record)
    assert value['phase'] == 'b00' and value['episode'] == 2


def test_pooled_lead_rmse_and_paired_summary():
    rows = []
    for error in (1., 3.):
        target = np.zeros((10,4))
        pred = target+error
        h1 = probe.metrics(target,target,np.zeros((52,4)),baseline())
        ar = probe.metrics(pred,target,np.zeros((52,4)),baseline())
        rows.append(dict(family='base',phase='b00',metrics={'k1_h1':h1,'k1_ar':ar}))
    result = probe.summaries(rows)
    metric = result['family_phase:base:b00']['metrics']['k1_ar']
    assert metric['per_lead_four_rmse'][0][0] == pytest.approx(np.sqrt(5))
    assert metric['four_rmse_mean_case'][0] == 2
    assert metric['per_lead_total_drag']['mae'][0] == 4
    assert result['overall44']['paired_ar_minus_h1']['k1']['four_mae'][0] == 2


def mapping_fixture():
    mapping = dict(status='FCP003C_FULL_TRAIN_SOURCE_PHASE_MAPPING_COMPLETE', trajectories=[],manifest_sha256={})
    data, predecl = {}, {'cases': {}}
    for family, (_, count) in probe.FAMILIES.items():
        label = 'base20' if family == 'base' else family
        data[family] = dict(train_files={},manifest_sha256=family)
        mapping['manifest_sha256'][label] = family
        for i in range(count):
            name = f'{family}_{i}.h5'
            data[family]['train_files'][name] = 'a'*64
            mapping['trajectories'].append(dict(family=label,file=name,canonical_physical_phase='b00',
                family_phase_label='b06',physical_start_time=148.,source_case=None))
            if family == 'train16':
                predecl['cases'][name[:-3]] = dict(split='train',phase='b00',env_index=0,
                    expected_frames=129,control_dt=.1,episode=i,run_window=[148.,160.8])
    return mapping, data, predecl


@pytest.mark.parametrize('damage', [None,'duplicate','missing','extra','manifest'])
def test_exact44_source_mapping_coverage(damage):
    mapping, data, predecl = mapping_fixture()
    if damage == 'duplicate':
        mapping['trajectories'][-1] = mapping['trajectories'][0]
    elif damage == 'missing':
        mapping['trajectories'].pop()
    elif damage == 'extra':
        row = dict(mapping['trajectories'][0], file='extra.h5')
        mapping['trajectories'].append(row)
    elif damage == 'manifest':
        mapping['manifest_sha256']['base20'] = 'wrong'
    if damage:
        with pytest.raises(ValueError):
            probe.validate_phase_mapping(mapping,data,predecl)
    else:
        result = probe.validate_phase_mapping(mapping,data,predecl)
        assert len(result) == 44
        assert result[('base','base_0.h5')]['phase'] == 'b00'
        assert result[('base','base_0.h5')]['family_phase_label'] == 'b06'
