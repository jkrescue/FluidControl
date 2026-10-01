# DGX Spark 可视化与 PPO 首轮审计（2026-10-02）

## 可视化交付

- 生成脚本：`scripts/build_tandem_visual_report.py`（仅 Python 标准库，离线 HTML）。
- DGX 页面：`artifacts/visualization/tandem_progress_20261002/index.html`，附 12 张独立测试轨迹的 PNG。
- Mac 副本：`/workspace/fluid_control/fluid_control_visual_report_20261002/index.html`；已在现有 Chrome 中打开。
- 页面覆盖四条测试轨迹各 1/10/50 步的真实 OpenFOAM u/v/p、PhysicsNeMo FNO 预测、逐点绝对误差，20 轮验证 MAE，以及 1/10/50 步独立测试误差。
- 源数据：`artifacts/tandem_fno_expanded_spark_20epoch/{training_history.json,heldout_evaluation*.json,control_readiness.json,heldout_figures/}` 和 `artifacts/hydrogym/tandem_ppo_pilot_20epoch_spark/audit.json`。所有流场图为真实 CFD 数据和模型预测的评估导出，不是手绘或合成样本。

重新生成：

```bash
python3 scripts/build_tandem_visual_report.py \
  --output artifacts/visualization/tandem_progress_20261002
```

## 冻结 FNO + HydroGym PPO 审计

PPO 以真实 OpenFOAM 轨迹的 t=80 起点训练，冻结 20 轮 PhysicsNeMo FNO 作为代理环境推进器；训练 512 个环境步，每回合评估 32 步，CPU 容器执行，未占用 GPU 训练显存。比较对象是同一代理环境中的零动作，不是未训练随机策略。所有评估起点来自 4 条独立验证轨迹和 4 条独立测试轨迹的第 0、100、400 帧。第 0 帧在各轨迹中共享同一 t=80 初始场，因此不能把 24 次评估描述为 24 个独立初始条件。

- 审计状态：`SURROGATE_RL_PILOT_EVALUATION_FAILED`。23/24 组完成；`expanded_test_00` 第 400 帧提前终止。
- 验证组 12/12 完成，平均奖励差（PPO − 零动作）为 −0.000871；7/12 组为正。
- 测试组 11/12 完成，已完成组平均奖励差为 −0.001288；6/11 组为正。失败组不纳入平均值，因此不能用该均值推断总体收益。
- 已完成组的平均 Cd 差分别为验证 +0.001789、测试 +0.002832；没有整体减阻证据。
- 当前只有 PPO 训练终点的策略快照。不存在 32/128/256 步的中间策略结果；页面只展示零动作与 512 步策略的真实对照，并明确标注此限制。
- 以上均为代理环境结果，不是闭环 OpenFOAM 控制结果。稳定性未过门槛，不能直接进入宣称控制有效的阶段。

下一步先复现并定位第 400 帧提前终止的原因，区分零动作和 PPO 轨迹，记录终止步、代理状态界限与动作；然后进行有检查点的 PPO 迭代审计。只有稳定性门槛通过后，才将策略作为候选进入分段 OpenFOAM 真实闭环回放，并保留零动作对照。

## 资源边界

DGX Spark 为 CPU/GPU 统一内存，不能把 `nvidia-smi` 的离散显存字段当作容量判据。页面生成和本次 PPO 使用 CPU；此前 PhysicsNeMo FNO 训练使用 GPU。审计时主机 `MemAvailable` 约 114 GiB、磁盘剩余约 664 GiB；后续 GPU 训练仍须持续保证至少 20 GiB 可用统一内存。OpenFOAM 中等网格求解仍在 CPU 上运行，尚不能报告网格收敛结论。
