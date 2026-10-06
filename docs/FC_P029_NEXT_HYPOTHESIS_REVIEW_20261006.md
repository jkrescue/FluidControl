# FC-P029 下一假设评审（2026-10-06）

## 状态与边界

本文是 P029 原始正式评估仍在运行期间的只读研究判断，不是新实验批准、模型准入或阈值调整。现有官方 PhysicsNeMo FNO 架构、数据划分、归一化、正式指标和 PPO 阻断条件均保持不变；最终判断仍以完整 formal 终态为准。

## 已观察到的转移断裂

同一 train44、origin-51、H10 协议下，P029 相对 K1 的平均 case 指标为：场 AR RMSE `0.038909243208102205 -> 0.03793723525648767`，rear-Cl AR MAE `0.03869321600319711 -> 0.03335770925861487`，total-Cd AR MAE `0.01690930107777769 -> 0.0153860518200831`。这是训练内、短时域证据，不是泛化证据。来源为 `artifacts/fcp029_h10_comparison_20261006/result.json`（SHA-256 `aed040eb22766f8f47b1f6093f50bc1aa753dd748cd90b8d68ac9db1095b329d`）。

在 validation10 的同一正式 H100 协议下，P029 相对 K1 却变差：velocity relative-L2 `0.043591 -> 0.056598218710937856`，rear-Cl MAE `0.040186 -> 0.04511747685228956`，total-Cd MAE `0.010928 -> 0.018809126985484155`。P029 的 H100 action-minus-zero Cd MAE `0.020433813333511353` 仍通过既定 `0.023` 子门槛，不能抵消场和窗口质量问题。P029 文件为 `validation10/evaluation.json`（SHA-256 `f210350771b7be1ad6fbf1f97ca772a0d2c4a321ebb3470b7666c8f398b77a42`）和 `endpoint_gate.json`（SHA-256 `caddf56fecb492075480966ad84818737a7592b9a6a9c23d3b8a17881865e6e2`）。

## 机制核查

P029 的项目自定义目标仅展开 10 步。`scripts/p029_control_aware_flow_objective.py`（SHA-256 `904fc903b35b6a754faf1243217b3a7f4a5f552afe90e1e8e3a33654d4524516`）在 `range(10)` 内累计场与冻结 aerodynamic FNO 的受力损失；第 0 个受力项对 flow 没有梯度，只有后 9 个受力项经 aerodynamic 输入回传，而 `qhat_{s+10}` 只受场监督。`scripts/train_p029_control_aware_flow.py`（SHA-256 `e7dd8b9a261b4a47dbfcec6ce4be792dce0294454460b9924e81c37fb87c4aa0`）虽然由 H100 窗口加载数据，但每个样本只消费其前 10 步；固定顺序的 1368 个窗口以 8 个窗口累积为 171 次更新，LR `1e-5`，没有 H11--H100 损失。训练结果 SHA-256 为 `39b246d07ff673de5b3d5fdc2e65be5e46bcb466e1290f5549f2e75d5cb81336`。

三类解释的相对证据如下：

1. **优化暴露/时域不匹配——当前首选。** H10 内场和力可改善，但 H100 场、rear-Cl 与 total-Cd 同时退化，符合局部十步目标把流场推向短期有利、长期不稳定或有偏的轨迹。171 次更新全部受固定 clip 影响也可能放大这一现象，但现有证据不能把原因归结为 clip、LR 或权重，故不建议扫描它们。
2. **数据覆盖——仍可能，但不是首选。** train 静态数据已包含 `0, ±0.375, ±0.75` 的相同行为幅值，并覆盖 b00/b02/b04/b06；validation10 使用 b01/b05 的插值相位和相同静态动作。因此“动作幅值缺失”不成立，但 held-out 相位泛化仍可能解释 train H10 与 validation H100 的分歧。train8/train16 的动态动作不能消除 b01/b05 未参与拟合这一事实。
3. **冻结 force readout 失配——不是单独充分解释。** 冻结 aerodynamic FNO 的输入梯度可能诱导 flow 产生对该读出器有利、但不物理稳健的状态；然而 validation H100 的 velocity 也明显变差，而不仅是力通道。此前 P026 K1/K4、P009--P013 已分别检验历史、线性读出、decoder scope、梯度冲突和独立 aerodynamic FNO；继续扫 head、历史长度或损失权重会重复已拒绝分支。

## 排名第一的下一可证伪假设

**假设 H1：P029 的主要转移失败来自 flow 训练只暴露于 H10 目标；其短期改善在同一 train44 轨迹上于 H25--H100 发生反转，而非首先来自动作幅值覆盖不足或单独的冻结力读出误差。**

最小有意义的检验是一次**无优化器、train-only、matched H100 时间剖面诊断**：在原 44 条训练轨迹、原 1368 个 H100 窗口、原 recorded actions、原 normalization、相同窗口权重和相同官方推理精度下，对 K1 与 P029（可同时保留 P028 作已存在的 field-only 对照）报告 H1/H10/H25/H50/H100 的场误差、rear-Cl 和 total-Cd，并按 family、canonical phase、action profile 分组。两模型必须使用完全相同的起点；不读 validation/frozen，不拟合参数，不保存候选，不据结果选择 horizon 或权重。

预先解释规则：

- 若 P029 在同一 train44 上 H10 改善、但 H25--H100 交叉为更差，支持“训练时域不匹配”；之后才值得另行评审固定的长时域 flow 监督，而不是直接启动训练。
- 若 P029 在 train44 的 H100 仍改善、仅在 b01/b05 validation 退化，则该假设被削弱，证据转向 held-out 相位覆盖/泛化；这也不会自动授权新增 CFD。
- 若 H100 场保持或改善、只有受力退化，则该假设同样被削弱，冻结 aerodynamic readout 的分布失配才成为更强候选。

该诊断不改变架构、数据划分、门槛或正式评估，也不重复 P027/P028 的 origin-51 H10 检查。它只回答现有证据尚未回答的问题：P029 的误差反转是在**同一训练分布的长时域**内已经出现，还是只在 held-out phase 上出现。执行仍需独立资源与执行批准；完整 P029 formal 终态优先。

## 对短时域、全场反馈 MPC 的含义（条件性，不是替代准入）

若未来控制体系能在每个短控制段之后取得真实 CFD 的完整 `(u,v,p)` 场并重新初始化 FNO，则 receding-horizon MPC 理论上不必依赖一次连续 H100 开环预测；它只需在每次重置之间保持短 H 动力学和候选动作排序可靠。因此，H100 开环失败并不在逻辑上证明这种**另一种控制架构**必然失败。但当前仓库尚未实现这一前提：真实 CFD/HydroGym 在线路径提供的是 69D 稀疏观测，不是 FNO 所需的完整归一化场；现有 `src/fluid_control/cem_mpc.py` 只是与动力学无关的 CEM 优化器，既有调用器仍是从 HDF 完整场启动的 legacy surrogate-only screen。完整接口差距已记录在 `docs/FC_P026_CONTROL_STATE_INTERFACE_REVIEW_20261006.md`。

因此，不能用“将来可采用短 MPC”豁免当前 P029 的完整 formal、PPO/world-model H100 门槛，亦不能把现有 CEM screen 当作闭环证据。若 H1 的 train-only H100 时间剖面支持“误差主要在 H10 后累积”，它至多为另行预注册的短时域 MPC 路径提供动机；该路径还必须先完成：

- 真 CFD 完整场到 FNO reset 的明确、因果、无未来信息接口，同时保持 K1/K4 状态历史与 canonical 62 点受力历史语义；
- 在固定短 H 下，对同一真实状态的备选动作进行 control-oriented 排序验证，而不仅是沿 recorded action 的预测误差；
- 用成对真实 CFD 闭环运行检验原物理目标（total drag、rear-Cl 均值/波动和动作约束），且与零控制及既有 CFD-only 基线按相同预算比较。

这不是第二个当前建议实验。本文排名第一的下一步仍只有上述无优化器 H100 时间剖面；只有它和完整 P029 formal 的结果共同支持时，才可提交独立的 MPC bridge/动作排序方案。即使该未来方案成功，也不能追溯性地把 P029 标为通过，或把 FNO 插入一个未使用的控制路径后声称完成 surrogate-assisted control。

## 既有证据约束

- 设计合同：`docs/FC_P029_CONTROL_AWARE_FLOW_PLAN_20261006.md`（SHA-256 `d9201cf8985e759089ab9d7e5ca92c5d3812681b6c88580e592226683c4b805d`）。
- 历史 H100 训练 FC-E003/FC-E004/FC-E005 的 horizon、epoch、loss、parent 或数据组成均不同，不能冒充本假设的 matched P029 H10-vs-H100 因果对照。
- P026 K1/K4 已在 H100 窗口上训练 aerodynamic 模型但均未通过窗口门槛；这不等于 P029 的 flow 模型已经接受 H100 梯度。
- P012 没有支持强烈的 field/force 梯度方向冲突；因此本建议不是重新开启 loss-weight 或 gradient-scale sweep。
- 当前 CEM 是离线 legacy screen，69D CFD observation 也不能直接重建完整 FNO state；因此“缩短 horizon”本身不是已经可执行的闭环修复。
