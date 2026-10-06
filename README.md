# PhysicsNeMo 串列双圆柱流动控制

## 项目概述

本项目面向串列双圆柱流动控制，使用 OpenFOAM 生成 CFD 数据，并基于 NVIDIA PhysicsNeMo 完成数据处理和 FNO 流场预测。

场景固定为 Re=100、L/D=5 的两个固定串列圆柱，仅通过后圆柱转动控制流动；目标是降低两柱总阻力，同时限制后柱升力波动与平均侧向力。

## 当前进展（2026-10-06）

| 工作 | 已核实的结果 | 尚未完成 |
| --- | --- | --- |
| 已验证核心链 | 真实 OpenFOAM CFD → 官方 Curator/DataPipe/FNO → HydroGym 接口 + SB3 PPO → 冻结 CPU PPO 的真实 OpenFOAM 反馈；模型、策略、数据、运行源和批准均有 SHA 绑定 | 尚未在全新输出上按复现指南重放后三阶段；原始 CFD 获取、Curator 转换和 K1 parent 训练没有单一总入口 |
| B 策略真实 CFD 闭环 | 同一冻结 B-32768 PPO 在 b00/b01/b07 各完成 800 个 paired 控制周期；主 60 D/U 总减阻分别约 3.895% / 3.928% / 3.903%，后柱升力 RMS ratio 均约 0.815，平均偏置主窗满足原标准 | 已观察相位不构成统计独立泛化；b00/b01 早期偏置失败仍保留；不宣称净能耗或硬件实时性 |
| PhysicsNeMo FNO 状态 | P064-B 作为冻结 surrogate 完成对应 fresh PPO 训练；真实闭环收益已独立复核 | B 完整预测 formal gate 仍 FAIL；H25 后续候选在同六工况快速评估中恶化并被拒绝，不进入新 PPO/CFD |
| 学习贡献归因 | 同 seed、b00 restart、投影/滤波及 identity VecNormalize 的初始权重对照已独审：主窗减阻 −0.007557%，训练后 B 为 +3.895284%；RMS ratio 1.000882→0.815623，平均偏置 1.650279%→1.137815%；paired-zero 全 16000 行逐值相同 | 初始策略两项升力标准通过、仅减阻失败；结果支持本次训练权重有实际贡献，不证明胜过所有简单控制器、去掉投影仍达标或跨 seed 泛化 |

当前 B-policy 是在冻结 PhysicsNeMo FNO surrogate 环境中训练的 SB3 PPO，因此不再沿用“只有 CFD-only PPO 有收益”的旧摘要。实际部署阶段仍是 **CPU PPO 策略 + 真实 OpenFOAM 观测/动作反馈**：不调用 online FNO，也不是 MPC。HydroGym 提供环境接口，OpenFOAM 才是真实数值求解器；“在线反馈”不是硬件实验。

最新证据与下一步依据：

- [当前闭环复现入口、版本和职责](docs/CURRENT_CLOSED_LOOP_REPRODUCTION_GUIDE_20261006.md)
- [B-policy b00 真实 CFD 独审](docs/P064_B_PROJECTED_PPO_LONG_CFD_TERMINAL_REVIEW_20261006.md)
- [B-policy b01 真实 CFD 独审](docs/P064_B_PROJECTED_PPO_B01_LONG_CFD_TERMINAL_REVIEW_20261006.md)
- [B-policy b07 真实 CFD 独审](docs/P064_B_PROJECTED_PPO_B07_LONG_CFD_TERMINAL_REVIEW_20261006.md)
- [相同投影/滤波下初始权重与训练后 B 的匹配 CFD 对照](docs/P064_INITIAL_POLICY_CFD_TERMINAL_REVIEW_20261006.md)
- [B surrogate 完整 formal 评估](docs/P064_B_FORMAL_TERMINAL_REVIEW_20261006.md)
- [当前权威状态](PROJECT_STATE.md)、[实验账本](EXPERIMENTS.md)与[结构化结果](experiments/results.csv)

旧 CFD-only PPO 仍是独立历史基线，不被新 B-policy 结果覆盖：

- [2026-10-04 真实 CFD-only PPO 结果及限制](docs/DIRECT_CFD_PPO_RESULTS_20261004.md)
- [当时的 FNO–HydroGym–OpenFOAM 接口审计](docs/FNO_HYDROGYM_PPO_OPENFOAM_INTEGRATION_AUDIT_20261004.md)
- [历史连续受力窗口检查](docs/FNO_FORCE_WINDOW_DIAGNOSTIC_20261004.md)

## 技术流程

```text
OpenFOAM CFD
  → VTK
  → PhysicsNeMo Curator
  → PhysicsNeMo DataPipe
  → FNO 训练与评估
  → HydroGym 接口中的冻结 FNO 环境
  → Stable-Baselines3 PPO 训练
  → 冻结 CPU PPO
  → 真实 OpenFOAM 配对反馈验证（无 online FNO / MPC）
```

## 代码结构

| 路径 | 内容 |
| --- | --- |
| `cfd/tandem_cylinders/` | CFD 工况生成与校验 |
| `conf/` | 训练和控制配置 |
| `scripts/` | 数据处理、训练、评估及 VTK 导出脚本 |
| `src/fluid_control/` | PhysicsNeMo DataPipe 和公共模块 |
| `docs/` | 补充资料与历史记录 |

## 使用指南

完整环境配置、执行命令和验收标准见以下文档：

- [不可静默变更的研究目标](docs/RESEARCH_OBJECTIVE.md)
- [训练操作手册](train_recipe.md)
- [CFD 工况说明](cfd/tandem_cylinders/CASE_SPEC.md)
- [论文复现对照](docs/PAPER_REPRODUCTION.md)
- [闭环控制方案](closed_loop_control_spec.md)
- [PhysicsNeMo–HydroGym 研究路线](docs/HYDROGYM_RESEARCH_ROADMAP.md)

已验运行使用项目隔离的 `.venv-curator-py312`、PPO overlay、固定 PhysicsNeMo b40 镜像和固定 OpenFOAM 镜像。版本、入口、systemd 资源合同与不可直接重跑的历史批准见[当前闭环复现指南](docs/CURRENT_CLOSED_LOOP_REPRODUCTION_GUIDE_20261006.md)。不要为复现实验无条件新建或覆盖 Python 环境。

普通 editable install 仅适合开发和 CPU 单元测试，不代表已验的训练/闭环运行时；开发者如需安装，应另建独立环境并遵循项目依赖声明，不能据此声称复现实验结果。

原始 CFD 数据、VTK、HDF5、模型文件和运行日志不纳入版本控制，可按操作手册重新生成。
