# HydroGym–PhysicsNeMo 串列双圆柱 PPO 软件烟雾测试（2026-10-02）

## 可核验结论

在 DGX Spark 的隔离 CPU 容器内，官方 HydroGym `FlowEnv`、官方 PhysicsNeMo 2.2.2 FNO 检查点和 Stable-Baselines3 PPO 已共同完成一条真实 CFD 初态上的短回合强化学习软件链路。训练使用独立 v2 数据中的 `expanded_train_00` 第100帧，PPO 训练32个环境步；冻结后在 `expanded_validation_00` 第100帧分别回放零请求动作和策略16步。两个回放都数值有限、按预定步数结束，奖励分量与总奖励的最大不一致分别为 `2.78e-17` 和 `1.39e-17`。结果状态严格为 `surrogate_only_software_smoke_not_cfd_control`，**不证明 RL 控制收益**。

执行入口：`bash scripts/run_tandem_hydrogym_ppo_smoke_spark.sh`。实验使用 CPU-only 派生容器 `fluid-control-physicsnemo-hydrogym:2.2.2-4ab9854`，`--network none --memory 16g --cpus 4`；未占用正在训练20-epoch FNO的 GPU0。HydroGym 源码固定 commit `4ab9854dea3d84e38a59c25e0f5835a00cf8225f`。代理来自真实 OpenFOAM v2512 数据上训练的 PhysicsNeMo FNO epoch5；训练/验证环境从各自真实 Curator HDF5 帧重置，不能把代理生成的后续状态称为 OpenFOAM 数据。

## 16步 validation 代理回放

| 指标 | 零请求动作 | 冻结 PPO | 解释 |
| --- | ---: | ---: | --- |
| reward 总和 | -2.263332 | -2.263261 | PPO 高 `7.10e-5`，极小 |
| 后柱 Cd 平均 | 0.642674 | 0.642745 | PPO 高 `0.0111%`，方向不利 |
| 后柱 Cl RMS | 1.963517 | 1.963370 | PPO 低 `0.00749%`，极小 |
| 实际动作 RMS | 0.279508 | 0.279510 | 几乎相同 |
| 最大全场归一化幅值 | 6.941864 | 6.941436 | 均低于20的数值发散门槛 |
| 限速步数 | 2 | 3 | 生效且记录 |

初始 CFD 帧的后柱转速不为零。受每步最大 `|Δomega|=0.5` 约束，“零动作”基线的首步实际转速仍为 `1.0`，随后才降至零；因此表中零动作是**零请求动作加相同执行器限速**，并非从首步即物理转速零。PPO 第一步请求约 `-0.000488`，实际同样被限速到 `1.0`。这解释两个轨迹几乎重合；32步训练也远远不足以估计策略稳定性或统计控制效果。

## 产物与下一步

可核验逐步奖励/力/动作日志在 `docs/results/tandem_hydrogym_ppo_smoke_audit.json`；原始模型权重保留在远程 `artifacts/hydrogym/tandem_ppo_smoke_5epoch_spark_audited/ppo_smoke.zip`（SHA-256 `bd8aa20b76cf7e851772dc7b6ce47b79cff187bc3cd9bd16c66641c871383bc7`），不提交 GitLab。当前 FNO 的独立5-epoch 测试在1步力预测上比保持基线差；需要先读完20-epoch 独立留出结果，检查短期力误差与50步滚动误差，再决定长回合多种子 PPO/SAC 的可信环境窗和是否补充 CFD。只有冻结策略经过独立 OpenFOAM 状态反馈回放和网格检查，才能报告真实控制收益。
