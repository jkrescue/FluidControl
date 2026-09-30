# PhysicsNeMo MPC 与 OpenFOAM 回放

## 配置

- 动力学模型：扩展数据集、无 teacher forcing 的 PhysicsNeMo FNO Epoch 30
- 控制对象：后圆柱角速度
- 控制周期：`0.1 D/U∞`
- MPC：30 步预测窗、15 次 Adam 优化
- 约束：`omega∈[-0.75,0.75]`、`|Delta omega|≤0.04`
- 代价权重：升力 1.0、阻力 1.0、动作 0.05、动作变化率 0.03
- 动作输入：物理转速除以数据 manifest 的 `max_abs_omega=5` 后送入 FNO

## 代理模型结果

100 步评估覆盖 5 个不同初始流场。所有动作和状态均为有限数，约束零违反。

| 初始状态 | `Cl RMS` 变化 | `Cd mean` 变化 |
| --- | ---: | ---: |
| test_00，step 0 | -10.22% | +6.33% |
| test_00，step 325 | -15.44% | +9.69% |
| test_01，step 206 | -12.06% | +5.98% |
| test_02，step 263 | -4.14% | +5.18% |
| test_03，step 325 | -5.47% | +6.12% |
| 平均 | -9.47% | +6.66% |

将优化学习率从 0.08 提高到 0.4 后，动作长期饱和，`Cl RMS` 增加 12.79%，因此被拒绝。该负结果保存在 `surrogate/rejected_high_lr_result.json`。

## OpenFOAM 回放

将 test_00、step 0 的 100 个动作冻结后，从同一个 `t=80` 无控制流场重启 OpenFOAM。回放使用 2000 个 `pimpleFoam` 步，最大 Courant 数为 0.245，无非有限值。

| 统计窗口 | `Cl RMS` 变化 | `Cd mean` 变化 |
| --- | ---: | ---: |
| `t=80–90` | -10.49% | +3.78% |
| `t=82–90` | -15.62% | +5.23% |

CFD 与代理都显示升力波动下降和平均阻力上升，方向与量级一致。该试验是预先冻结动作的高保真回放，不是根据 CFD 新状态重新计算动作的闭环控制。

## 动作尺度审计

初次实现沿用了第一阶段 `max_abs_omega=1` 的假设，遗漏了扩展数据的动作尺度 5。代码审查发现后已修正并重新执行全部代理和 CFD 结果。旧动作序列及其 CFD 结果保存在 `legacy_unscaled/`，只用于审计，未纳入上表和最终结论。

## 文件

- `selected_actions.csv`：用于 CFD 回放的冻结动作；
- `cfd_replay_result.json`：CFD 数值健康和物理指标；
- `cfd_replay_timeseries.png`：CFD 与零转速基线对照；
- `surrogate_timeseries.png`、`surrogate_final_fields.png`：代理模型结果；
- `surrogate/*.json`：参数消融和独立初始状态结果。
