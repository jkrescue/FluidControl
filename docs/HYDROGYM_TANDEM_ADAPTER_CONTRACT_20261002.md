# 串列双圆柱 PhysicsNeMo–HydroGym 适配契约（2026-10-02）

本文件记录早期薄适配层的软件接口与验收门槛；当时已通过单步验收，双圆柱 RL 和 OpenFOAM 闭环尚未完成。使用的官方组件是 PhysicsNeMo 2.2.2 FNO/检查点 API、HydroGym commit `4ab9854dea3d84e38a59c25e0f5835a00cf8225f` 的 `PDEBase`、`TransientSolver`、`FlowEnv`，以及 SB3 2.7.1。两套库已在派生容器 `fluid-control-physicsnemo-hydrogym:2.2.2-4ab9854` 同时导入；构建脚本为 `scripts/setup_physicsnemo_hydrogym_spark.sh`。不修改宿主 Python/CUDA 环境。

2026-10-02 更新：上文是**早期 67 维、末柱 Cd/Cl 奖励契约的历史表述**。该路径后来完成了 PPO 和 32 步真实 OpenFOAM 反馈，但物理目标变差，见 `docs/PPO_REAL_OPENFOAM_FEEDBACK_PILOT.md`。当前论文主线已锁定“两柱总阻力为主、两柱升力作保护”的 Stage-C 目标：新 FNO 输出四个受力系数，HydroGym 观测为 64 个探针速度 + 前后柱四个受力系数 + 转速，共 69 维；奖励使用 `src/fluid_control/stage_c_objective.py` 的完整脱落周期窗口。原 67 维代码与下文旧指标只用于历史复现，**不能作为当前总阻力控制结果**。Stage-C 进入 PPO 前仍须通过冻结的 v3 Gate B；现有两组 30 轮单步模型都未通过。

## 薄适配层的真实边界

HydroGym 官方 `FlowEnv` 按 `flow=PDEBase`、`solver=TransientSolver` 组合，`step()` 调求解器一步，再从 `flow.get_observations()` 和 `flow.evaluate_objective()` 返回 Gymnasium 结果。双圆柱环境应实现最小的项目本地 `PDEBase`/ `TransientSolver` 子类，保留官方 `FlowEnv`、Gymnasium 和 SB3 API；它是**代理动力学后端**，不是 HydroGym 官方 OpenFOAM 求解器，也不能被称为 CFD 真实环境。

代理状态为当前归一化的 `u/v/p` 全场、固定流体掩码、当前物理转速和后柱 `Cd/Cl`。每一步把 `[state, mask, omega_now/5, omega_next/5]` 送入官方 PhysicsNeMo FNO；输出前三通道是归一化状态增量，后两通道经流体掩码平均、训练集力统计反归一化得到下一时刻后柱 `Cd/Cl`。这必须与 `scripts/evaluate_tandem_fno.py` 的单步推理逐项一致。只使用实际 HDF5 初始帧；禁止随机噪声或人工合成场冒充 CFD 数据。

动作是后柱物理转速 `[-5,5]`，控制间隔 `0.1 D/U∞`。环境包装器还须验证动作变化率、有限值和动作支持范围；越界动作拒绝或显式裁剪并在 `info` 中报告。观测第一版使用现有 32 个尾流位置的 `u/v`（共64数）加后柱 `Cd/Cl` 和当前动作；可保留全场观测只作算法上界。探针从代理流场在固定物理坐标双线性插值；位置和插值误差先与真实 HDF5/原始 OpenFOAM 探针比较，不能凭网格索引猜测。

奖励按可审计物理分量给出：`-dt*(w_cd*Cd_rear + w_cl*Cl_rear**2 + w_u*omega**2 + w_rate*(omega-omega_prev)**2)`。`info` 单独记录每项、预测力、动作、状态有限性和模型版本；`sum(info.reward_*)` 必须等于返回的 reward。奖励可用于代理探索，但不是 CFD 控制效果。

## 数据隔离和最小验收

1. 训练回合仅从独立 v2 的 train 轨迹启动；validation 用于超参和冻结策略选择；test 只在最终留出评估读取。所有轨迹仍共享 Re、几何、网格和 `t=80` 重启场，此划分只隔离动作日程，不代表初始相位泛化。
2. 先在一个 validation 帧、真实观测下一步动作上比对适配层与既有 `evaluate_tandem_fno.py`，要求状态和力差在数值精度内；重置同一帧须完全可复现。反号动作应改变预测，且动作和力单位正确。
3. 用有限步随机/零动作做软件链路烟雾测试，检查官方 `FlowEnv` 返回的 observation/reward/terminated/truncated/info，约束零违反、所有值有限。长时域若离开训练分布或代理发散应显式终止，不允许悄悄继续累计虚假 reward。
4. 代理 PPO/SAC 至少多个种子，与零动作、开环和已有 MPC 作相同评价；只把代理结果称为候选策略。当前 5-epoch 代理的 1 步力 MAE 比保持力基线差，需先看 20-epoch 的独立留出结果，再决定长时域代理训练的可信窗口。
5. 冻结候选策略后，按同一观测、控制周期和动作约束接 OpenFOAM 状态反馈接口，跨多个涡脱落周期与独立初始相位比较后柱 `Cd_mean`、`Cl_rms`、能耗及约束。只有真实 CFD 回放通过，才声称双圆柱闭环控制收益；之后再做中网格检验。

## 已执行的软件和物理输入验收

`src/fluid_control/tandem_hydrogym.py` 实现了官方 HydroGym 核心类的最小子类与可审计 Gymnasium 包装器。使用真实验证集帧、5-epoch PhysicsNeMo 检查点执行 `scripts/validate_tandem_hydrogym_adapter.py`，适配层与原 FNO 评估计算的单步流场/受力差、奖励分量求和差、重置观测差均为 0；限速生效，范围外动作被拒绝。这是接口和数值等价烟雾测试，不能推断长时域代理可信或控制收益。

`scripts/validate_tandem_probe_mapping.py` 对比真实 OpenFOAM `wakeProbes` 与 Curator HDF5 插值，在 `expanded_validation_00` 和 `expanded_test_04` 的 `t=90/120/160` 共六组时刻上最大单分量误差分别为 `0.004206/0.004498`，均低于当前 `0.02` 的验收阈值。这验证观测位置和尺度；不是代理预测误差指标。结构化结果见 `docs/results/tandem_hydrogym_adapter_parity.json`、`tandem_probe_mapping_validation_00.json`、`tandem_probe_mapping_test_04.json`。
