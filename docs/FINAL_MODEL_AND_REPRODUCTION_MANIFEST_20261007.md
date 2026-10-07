# 最终模型与复现交付清单（2026-10-07）

审计快照：2026-10-07 09:23 UTC。主节点仓库为 `/workspace/fluid_control`；下文路径均相对此根。科学测试截止 **12:20:45 UTC（北京时间20:20:45）**，最终代码、模型索引及报告归档截止 **12:50:45 UTC**。本文不授权新实验，不复制大模型。

## 默认交付与边界

默认仍是冻结 **B surrogate + E082 canonical PPO policy + 配套VecNormalize**，不是最新训练候选。官方PhysicsNeMo提供FNO、checkpoint与Curator/Reader；项目代码提供数据/训练/控制适配，HydroGym提供环境接口，开源SB3提供PPO。PPO在冻结FNO环境中学习；已验证部署是 **CPU PPO推理→真实OpenFOAM反馈**，没有在线FNO或MPC参与该默认闭环。

- [E095真实800周期复现](CANONICAL_B01_REPRODUCTION_TERMINAL_REVIEW_20261007.md)：主窗减阻4.0090689485%，rear-Cl波动RMS比0.8171901322，均值偏置3.6366076181%。原2%/1.05/10%物理门限通过。
- [E109后续时段](P064_B_FUTURE_TIME_CFD_TERMINAL_REVIEW_20261007.md)与[E114延长段](P064_B_CONTINUATION_328_408_TERMINAL_REVIEW_20261007.md)：累计1600反馈周期，经历已核验恢复，不是一个无中断进程。最新800主窗减阻4.13258915%，波动降低17.9228711%，偏置1.235202589%；四个20 D/U块及规定joined窗均通过原门限。
- 完整代理预测精度仍未达标；同Re100/L/D5的新时间段不等于独立物理工况或统计独立样本。不声称泛化、净节能、硬件实时性或整体项目完成。
- G及absolute64各有单工况探索物理PASS，但预测选择FAIL、无全面优越性，均不替换B。I及多个辅助目标未通过原选择规则。
- 代表性256点候选本快照仍在运行：unit `fluid-control-p064-representative256-training-20261007.service`，inv `6bc6aca4e1bd48d5938b273d319fc716`。未读取其live checkpoint；未来路径 `artifacts/p064_representative256_training_20261007/candidate/dual_model_manifest.json` **不属于已交付模型**。必须终态工程审计、同精度六窗与固定开发评价、Lead决定，不能以训练误差或文件存在晋级。

## 完整B检查点与部署工件

以下10项已于本次在Spark只读逐字节流式SHA256核验，未反序列化模型、未推理。`.mdlus`模型文件和`.pt`训练状态均保留；只交一个权重文件不等于完整训练检查点。

| 路径 | 字节 | SHA256 |
|---|---:|---|
| `artifacts/fcp064_controlled_aero_arm_b_20261006/dual_model_manifest.json` | 9635 | `92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891` |
| `artifacts/fcp064_controlled_aero_arm_b_20261006/training_protocol.json` | 1528 | `7bb41ca973bfabfc01e73796fb744d628490291d268bdf584d2e6ef523b7621e` |
| `artifacts/fcp064_controlled_aero_arm_b_20261006/flow/FNO.0.0.mdlus` | 188903667 | `dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31` |
| `artifacts/fcp064_controlled_aero_arm_b_20261006/flow/checkpoint.0.0.pt` | 1773 | `4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e` |
| `artifacts/fcp064_controlled_aero_arm_b_20261006/aerodynamic/FNO.0.1.mdlus` | 188903667 | `57d4634df22ce96c1c4467a2ed52412be452375129af05b89f10a690e363356e` |
| `artifacts/fcp064_controlled_aero_arm_b_20261006/aerodynamic/checkpoint.0.1.pt` | 377807463 | `abc8eb89523d50a2c019d31b7e6a23dda384e9afc7d465d512abe32a80d1e216` |
| `artifacts/p064_b_symmetry_canonical_h5_32768_ppo_20261007/payload/ppo_final.zip` | 251295 | `5c05699e0851787d85d40c407647f80c19d3aebeb7dff82e019336cde77c6c6e` |
| `artifacts/p064_b_symmetry_canonical_h5_32768_ppo_20261007/payload/vecnormalize.pkl` | 5149 | `1d25005144b6436c3e2641ee89d1585e3c9f8b9fdb1f26b9cd39c7d83610c145` |
| `artifacts/b00_controlled_train_dataset_view_20261006/normalization.json` | 1248 | `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1` |
| `artifacts/fcp027_diagnostic_source_20261006_immutable/training_config.yaml` | 1764 | `07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9` |

B的直接父本是 `artifacts/fcp026_history_training_k1_20261005/candidate/dual_model_manifest.json`，SHA `7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7`。flow沿P009/K1固定，B只微调气动力分支28个张量（不是仅最后线性层），两lifting bias冻结。官方FNO仍6输入/7输出，mask-pool后四力。配置只是基础模型/数据配置；有效训练协议还必须带上training_protocol、原192+64采样顺序与批准文件。

训练数据/划分依赖见 [机器inventory](CANONICAL_CHAIN_INPUT_INVENTORY_20261007.json)：原train-only三根 `artifacts/p064_host_train_only_views_20261006/{base,train8,train16}` 加b00 view，归一化固定不重算。复现起点是**已存在真实curated数据和pretrained K1**，不是data-only从零训练。原始OpenFOAM→VTK→官方Curator/HDF的上游职责及分段入口见[复现指南](CURRENT_CLOSED_LOOP_REPRODUCTION_GUIDE_20261006.md)；旧projected策略不等于本文E082 canonical策略，部署政策以本表5c056/1d250为准。

## 推荐入口（默认只读）

在Spark仓库根运行：

```sh
cd /workspace/fluid_control
.venv-curator-py312/bin/python scripts/reproduce_canonical_closed_loop.py
python3 scripts/check_canonical_chain_inventory.py --repo "$PWD" --inventory docs/CANONICAL_CHAIN_INPUT_INVENTORY_20261007.json
```

第一条期望 `PREFLIGHT_PASS_NOT_RUNNING`，不启动CFD、不加载策略；第二条为依赖inventory检查，附加 `--verify-payload-bytes` 才流式核大payload，不解码。本文只记录推荐命令，没有再跑科学计算。Mac既有SSH转发页面为 `http://localhost:8766/`，曲线/场图有各自历史标签，不把回放显示成实时运行。

安全入口默认B；`--profile g`仅是已核验单工况探索复现。实际新复现必须另有新批准JSON、SHA、独占unit/output和显式`--execute`；历史approval是一轮证据，不可直接再次执行。canonical B物理driver由 `docs/P064_B_SYMMETRY_CANONICAL_B01_CFD_APPROVAL_20261007.json` SHA `ee010bbe1932e0b48b2f2e90a1b9dd77d463f86dad6367e0c0dd49a46f0b45f1` 固定；它绑定E082 policy/Vec、canonical symmetry adapter `a55b569986b6e62fd23d1c46dbe4f117659795aef3953cac81359506d1ac38ae`、单动作filter、restart树与OpenFOAM镜像。

Git保存审查过的代码、配置、索引、批准和报告；大HDF/CFD/模型payload在Spark原路径，未承诺包含在Git clone中。下载/迁移应按本表及inventory保留全部相关文件再复核SHA，不复制到Mac充当科研数据仓库。

## 截止与自动启动器核查

09:22 UTC只读检查用户/system timers、用户running units、相关进程与用户crontab：唯一running科学unit为上述rep256，另有dashboard与只读资源监控。未发现其他活跃科学launcher；用户crontab为空，`atq`未安装，因此不将该检查泛化为全机绝对不存在其他调度途径。

发现 `fluid-control-training-evaluation-watchdog.timer` 每分钟触发同名 `.service`，其第二ExecStart为 `scripts/reconcile_training_evaluation_state.py`。源码虽然旧docstring称allowlist为空，实际允许启动 `fluid-control-train16-posteval-main-resume-r1-20261004.service`；本次latest decision仅LEAD_ACTION_QUEUED，但不能据此保证未来无自动启动。

Lead批准后09:23:06 UTC再次 `systemctl --user cat` 核定义，实际执行：

```sh
systemctl --user disable --now fluid-control-training-evaluation-watchdog.timer
systemctl --user stop fluid-control-training-evaluation-watchdog.service
```

timer前为active/waiting/enabled，后为inactive/dead/disabled；oneshot前后均inactive/dead/static。只移除该timer启用软链并停止启动器，未删除科研数据；本轮不得恢复旧自动启动。**未停止** `fluid-control-dashboard-20261002.service`、`fluid-control-dual-node-watchdog-20261003.service`、训练或其他项目服务。

rep256实际 `KillMode=control-group`、`TimeoutStopSec=20`、`RuntimeMaxSec=2430`，自然上限早于全局12:20:45。后续每个新科学unit必须由Lead登记精确unit/inv及最迟终止时间不晚于截止。截止时若某已登记科学unit仍运行，先核inv/ExecStart归属，再仅对该完整unit执行`systemctl --user stop`并确认子进程退出；不要宽泛pkill、通配停止所有fluid-control或停止dashboard/归档。此处是精确停止方案，不是授权新任务或已创建的自动截止服务。
