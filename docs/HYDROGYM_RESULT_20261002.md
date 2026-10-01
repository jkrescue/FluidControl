# HydroGym 旋转圆柱强化学习验收结果（2026-10-02）

## 结论

DGX Spark 的隔离 ARM64/Firedrake 容器已完成官方 HydroGym `RotaryCylinder`、`Re=100`、medium 网格、50 压力探针上的 8,192 环境步 PPO 训练和留出相位评估。所用重启场来自 [HydroGym 官方公开数据](https://huggingface.co/datasets/dynamicslab/HydroGym-environments)，不是伪造流场。模型使用 Stable-Baselines3 官方 `MlpPolicy`；流体方程仍由 HydroGym/Firedrake 求解。

对 5 个训练中未用的重启 checkpoint，每个策略回放 600 步、舍弃前 100 步。PPO 相对零控制，5/5 组的升力 RMS 下降、5/5 组的含控制代价物理奖励改善；**平均阻力不是稳定改善**（2 组下降，3 组略升）。这证明当前 RL 软件链和升力抑制信号可复现，但不能宣称串列双圆柱闭环控制成功，也不能宣称阻力稳定降低。

| 留出 checkpoint 时刻标签 | 平均 Cd 相对零控制 | Cl RMS 相对零控制 | 每步平均奖励差 |
| ---: | ---: | ---: | ---: |
| 1080 | -0.6737% | -68.66% | +0.0001944 |
| 1110 | -0.1980% | -60.52% | +0.0001182 |
| 1140 | +0.0574% | -50.69% | +0.0000655 |
| 1170 | +0.1014% | -44.90% | +0.0000461 |
| 1199 | +0.0030% | -42.32% | +0.0000523 |
| 5 相位算术平均 | -0.1420% | -53.42% | +0.0000953 |

这里的“留出”仅指不同重启相位，不是不同雷诺数、几何形状或独立实验。5 个相位可能相关，不能据此给出总体统计显著性或跨工况泛化结论。早期 256 步软件冒烟模型在多数相位反而增大升力 RMS，因此本表仅代表 8,192 步正式模型，不能把冒烟结果混入正式性能。

## 审计与复现

- 奖励逐步定义：`-dt[Cd + 0.2 Cl² + 0.01 omega² + 0.001(omega-omega_prev)²]`；分别保存阻力、升力、执行动作和动作变化四项。10 条 600 步回放共 6,000 行，四项和与记录奖励的最大绝对差为 **0.0**，所有汇总量有限值。
- 训练模型、归一化统计、逐步回放和日志保存在 Spark 项目 `artifacts/hydrogym/rotary_physical_v1/`；小型汇总文件同步于 `docs/results/hydrogym_rotary_physical_v1_audit.json`。原始 checkpoint 缓存和 6,000 行回放不进入 GitLab。
- SHA-256：`model_final.zip` = `9ba82140f9c4c3ad626206d01689cae2af20fa200e853e9665e472ce19c68ce1`；`vecnorm_final.pkl` = `3216ed7206ce2281841418397d1ee154f3a46318d8667b46a19cbd7391793371`；`heldout_rollouts.csv` = `4c924d054b1106816cbf5804cb22be2d0a24f173b6c7a6268d9b240d677bdfb8`。
- 源码与数据版本见 `docs/HYDROGYM_SPARK_20261002.md`。复现实行 `bash scripts/run_hydrogym_rotary_spark.sh`，它拒绝覆盖完成模型，需换新输出目录来重训。

## 下一步判定

1. 在单圆柱基准做至少多个随机种子、相位更分散且更长的回放，并报告执行动作能耗；若目标是阻力下降，应调权后重新训练，不得拿当前策略冒充阻力控制成功。
2. 串列双圆柱仍以 OpenFOAM 真实数据和 PhysicsNeMo 官方 FNO 为主线；需先看完整 Curator 数据、FNO 测试集 1/10/50 步 rollout 与物理力系数。只有代理稳定、动作条件响应可信，才把它用于快速策略搜索，再让 OpenFOAM 闭环回放验收。
3. HydroGym 当前官方旋转圆柱与本项目 OpenFOAM 双圆柱不是同一环境。后续若对接，需要明确定义动作、观测、奖励、时间步及重启状态的适配层，并保留 CFD 终验；不能把两个环境指标直接比较。
