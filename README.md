# PhysicsNeMo 串列双圆柱流动控制

## 项目概述

本项目面向串列双圆柱流动控制，使用 OpenFOAM 生成 CFD 数据，并基于 NVIDIA PhysicsNeMo 完成数据处理和 FNO 流场预测。

场景固定为 Re=100、L/D=5 的两个固定串列圆柱，仅通过后圆柱转动控制流动；目标是降低两柱总阻力，同时限制后柱升力波动与平均侧向力。

## 当前进展（2026-10-04，阶段记录）

| 工作 | 已核实的结果 | 尚未完成 |
| --- | --- | --- |
| 真实 CFD 强化学习与在线仿真闭环 | SB3 PPO 经 HydroGym 接口与项目 OpenFOAM 适配器交互；固定策略在两个启动时刻总减阻 4.22% / 4.25%，均满足既定升力限制 | 两个启动时刻约相隔三个脱涡周期，不是统计独立样本；反馈优于固定动作序列、网格收敛及广泛泛化尚未证实 |
| PhysicsNeMo FNO 代理模型 | 官方 FNO、Curator、DataPipe 已形成真实数据训练流程；新增动态转速数据上的 H50/H100 训练正在进行 | 已完成父模型仍有控制动作效果判断错误，不能将 CFD-only PPO 收益归因于 FNO |
| 数据与可复现性 | 真实 OpenFOAM 数据、不可变训练归一化、训练/验证分离、模型和代码 SHA 记录 | 正在单独整理此前 PPO 训练交互的真实流场，不混入当前训练，也不使用验证轨迹训练 |

过去的 MPC 接口测试和短时回放不等于通过最终减阻验收。当前有物理收益的策略是 **CFD-only PPO**，不是 FNO 训练出的控制策略；在线指仿真中实时读取流动状态并反馈动作，不是硬件实验。

最新证据与下一步依据：

- [已完成的真实 CFD PPO 结果及限制](docs/DIRECT_CFD_PPO_RESULTS_20261004.md)
- [FNO 到 HydroGym 的接口与模型准入检查](docs/FNO_HYDROGYM_PPO_OPENFOAM_INTEGRATION_AUDIT_20261004.md)
- [控制相关的连续受力窗口检查](docs/FNO_FORCE_WINDOW_DIAGNOSTIC_20261004.md)

## 技术流程

```text
OpenFOAM CFD
  → VTK
  → PhysicsNeMo Curator
  → PhysicsNeMo DataPipe
  → FNO 训练与评估
  → 闭环控制验证
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

项目代码安装命令：

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .
```

原始 CFD 数据、VTK、HDF5、模型文件和运行日志不纳入版本控制，可按操作手册重新生成。
