# 复现指南

## 代码与原运行环境

原科学计算在 DGX Spark 完成，使用隔离 Python 环境和固定 PhysicsNeMo / OpenFOAM 镜像。GitHub 不发布真实账户、地址或部署路径；`/workspace/fluid_control` 为占位路径，不是节点实际目录。

GitHub 包含脱敏代码、配置、记录和报告，**不包含全部 HDF5、CFD 重启状态和权重**。新 clone 后仅安装依赖不能复现实验；上游数据生成与从零训练尚无单一完整入口。

脱敏修改了机器路径及部分文件字节。保留的历史 SHA 用于追溯原科研工件，不能校验脱敏源码，也不能直接运行原哈希绑定的执行入口。新部署需要配置本机路径、补齐工件并重新审核运行配置；不将旧实验记录当作新运行授权。

## 默认模型与策略

| 工件 | 路径或身份 |
|---|---|
| B 双 FNO | artifacts/fcp064_controlled_aero_arm_b_20261006/dual_model_manifest.json |
| PPO | artifacts/p064_b_symmetry_canonical_h5_32768_ppo_20261007/payload/ppo_final.zip |
| PPO SHA256 | 5c05699e0851787d85d40c407647f80c19d3aebeb7dff82e019336cde77c6c6e |
| 策略归一化 | 同目录 vecnormalize.pkl，必须与策略一起使用 |
| 数据依赖 | [CANONICAL_CHAIN_INPUT_INVENTORY](../reproducibility/CANONICAL_CHAIN_INPUT_INVENTORY_20261007.json) |

[模型索引](../research_records/MODEL_INDEX.md)列出默认工件；完整文件与 SHA 见上述机器清单，不以未采用候选替换默认工件。

## 原环境的只读预检

以下命令记录原科研环境的预检方式，不是 GitHub 脱敏版本的可直接运行承诺：

```sh
cd /workspace/fluid_control
.venv-curator-py312/bin/python scripts/reproduce_canonical_closed_loop.py
python3 scripts/check_canonical_chain_inventory.py --repo "$PWD" --inventory docs/CANONICAL_CHAIN_INPUT_INVENTORY_20261007.json
```

第一条默认只读，期望 PREFLIGHT_PASS_NOT_RUNNING；通过预检不等于新 CFD 结果。第二条核查依赖，附加 `--verify-payload-bytes` 才核验大文件字节。

重跑需新的批准文件、SHA、独立输出和显式执行参数，不重复执行历史批准。核查资源及归属；Spark CPU/GPU 共享物理内存，保留至少 20 GiB 可用统一内存，具体入口可能要求更高余量。

## GitHub 阅读版组织

方法文档在 `guide/`，结果与模型索引在 `research_records/`。`reproducibility/` 仅保留默认 B 训练、canonical PPO、CFD 验证及已有 G 配置的必要依赖；docs 符号链接指向该目录。

旧计划、重复报告和快照从主分支移除；[历史分支](https://github.com/jkrescue/FluidControl/tree/research-history-20261009)保留脱敏的完整记录。旧探索脚本和报告重建所需的资料应在对应历史版本查找，不再保留在主分支。

Linux/Spark checkout 应保留符号链接；Windows 未启用 Git 符号链接支持时可能得到文本文件，应改用支持符号链接的环境运行。GitLab 科研工作目录未因 GitHub 文档整理而修改。

[CFD 配置](../cfd/tandem_cylinders/CASE_SPEC.md) · [训练](TRAINING.md) · [控制](CONTROL.md)
