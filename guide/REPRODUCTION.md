# 复现指南

## 已验证环境

科学计算与大数据位于 DGX Spark 的 `/workspace/fluid_control`，不是 Mac。使用项目隔离 Python 环境、固定 PhysicsNeMo 和 OpenFOAM 镜像，不覆盖系统 Python 或其他项目环境。

GitHub 包含代码、配置、批准、报告及部分紧凑证据，**不包含全部 HDF5、CFD 重启状态和模型权重**。新 clone 后仅安装依赖不能复现实验；已有入口依赖固定的 Spark 路径、版本和数据。上游数据生成与从零训练尚无单一完整入口。

## 默认模型与策略

| 工件 | 路径或身份 |
|---|---|
| B 双 FNO | artifacts/fcp064_controlled_aero_arm_b_20261006/dual_model_manifest.json |
| PPO | artifacts/p064_b_symmetry_canonical_h5_32768_ppo_20261007/payload/ppo_final.zip |
| PPO SHA256 | 5c05699e0851787d85d40c407647f80c19d3aebeb7dff82e019336cde77c6c6e |
| 策略归一化 | 同目录 vecnormalize.pkl，必须与策略一起使用 |
| 数据依赖 | [CANONICAL_CHAIN_INPUT_INVENTORY](../reproducibility/CANONICAL_CHAIN_INPUT_INVENTORY_20261007.json) |

[模型索引](../research_records/MODEL_INDEX.md)列出默认工件；完整文件与 SHA 见上述机器清单，不以未采用候选替换默认工件。

## 默认只读预检

在**已有完整工件的 Spark 仓库**执行：

```sh
cd /workspace/fluid_control
.venv-curator-py312/bin/python scripts/reproduce_canonical_closed_loop.py
python3 scripts/check_canonical_chain_inventory.py --repo "$PWD" --inventory docs/CANONICAL_CHAIN_INPUT_INVENTORY_20261007.json
```

第一条默认只读，期望 PREFLIGHT_PASS_NOT_RUNNING；通过预检不等于新 CFD 结果。第二条核查依赖，附加 `--verify-payload-bytes` 才核验大文件字节。

重跑需新的批准文件、SHA、独立输出和显式执行参数，不重复执行历史批准。核查资源及归属；Spark CPU/GPU 共享物理内存，保留至少 20 GiB 可用统一内存，具体入口可能要求更高余量。

## GitHub 阅读版组织

主要方法文档在 `guide/`，结果与模型索引在 `research_records/`。被代码、配置或复现入口引用的旧文件在 `reproducibility/` 保持原字节，docs 符号链接指向该目录。

旧计划、重复报告与运行快照已从主分支移除，完整记录保存在 [research-history-20261009 分支](https://github.com/jkrescue/FluidControl/tree/research-history-20261009)。历史链接或旧 HTML 报告的完整重建应使用该分支；主分支只保留当前阅读材料及检索到的程序依赖。

Linux/Spark checkout 应保留符号链接；Windows 未启用 Git 符号链接支持时可能得到文本文件，应改用支持符号链接的环境运行。GitLab 科研工作目录未因 GitHub 文档整理而修改。

[数据与分段运行原始说明](../reproducibility/CURRENT_CLOSED_LOOP_REPRODUCTION_GUIDE_20261006.md) · [CFD 配置](../cfd/tandem_cylinders/CASE_SPEC.md) · [训练](TRAINING.md) · [控制](CONTROL.md)
