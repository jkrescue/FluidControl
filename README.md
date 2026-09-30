# PhysicsNeMo 串列双圆柱流动控制

## 项目概述

本项目面向串列双圆柱流动控制，使用 OpenFOAM 生成 CFD 数据，并基于 NVIDIA PhysicsNeMo 完成数据处理和 FNO 流场预测。

第一阶段已完成 CFD 数据生成、Curator 转换、DataPipe 校验、FNO 单步与多步训练、独立测试及 ParaView 数据导出。闭环研究将在扩展高转速动作数据后开展。

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
