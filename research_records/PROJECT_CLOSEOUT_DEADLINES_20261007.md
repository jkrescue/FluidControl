# 项目限时收尾合同（2026-10-07）

本文件记录 Lead 在 `2026-10-07T09:20:45Z`（北京时间 `2026-10-07 17:20:45`）下达的硬截止线。它优先于此前“持续迭代/继续探测”的开放式安排，但不修改任何科学阈值、已登记结果或历史批准。

## 绝对截止时间

- **停止全部科学测试：`2026-10-07T12:20:45Z`（北京时间 `20:20:45`）。** 到点后不得启动新的训练、推理评估、PPO、CFD、数据转换、超参数扫描或诊断。截止前只允许完成Lead明确授权的科学执行；已明确授权的只读后处理、独立核验、文档和归档可按其范围完成，并如实保存成功或失败结果。
- **完成归档与同步：`2026-10-07T12:50:45Z`（北京时间 `20:50:45`）。** 在此之前完成代码、配置、审批、结果索引、模型SHA清单、论文素材报告、GitLab同步和最终状态说明。大模型、HDF、CFD场和容器镜像留在Spark主节点，以路径、大小和SHA清单引用，不直接提交Git。

## 收尾规则

1. 不降低或重解释既有物理门限：减阻至少`2%`、rear-Cl波动RMS比不高于`1.05`、均值偏置不高于`10%`。
2. 不把真实PPO闭环通过改写成FNO完整预测通过；也不因预测FAIL否定已经独立核验的E114基本真实闭环。
3. 当前Representative256训练的终态与fixed-six结果在生成后填入最终报告；若截止前不能形成独立证据，则明确写“未完成/未知”，不得估算。
4. 历史批准只证明历史执行身份，不授权重跑。截止后默认仅允许只读复现预检、查看看板和读取已归档证据。
5. 任何未完成目标进入“限制与后续研究”清单，不再为补齐叙事而临时新增实验。
6. `2026-10-07T09:23:06Z`已将`training-evaluation-watchdog.timer`从`active/waiting/enabled`精确停为`inactive/dead/disabled`；同名oneshot保持`inactive/dead/static`。该操作未停止dashboard或当时已批准的Representative256训练。

## 交付文件

- 中文最终报告：[PROJECT_FINAL_REPORT_20261007.md](PROJECT_FINAL_REPORT_20261007.md)
- 当前状态：[`PROJECT_STATE.md`](../PROJECT_STATE.md)
- 实验与决策台账：[`EXPERIMENTS.md`](../EXPERIMENTS.md)、[`DECISIONS.md`](../DECISIONS.md)、[`results.csv`](../experiments/results.csv)
- 基本闭环复现：[CANONICAL_CLOSED_LOOP_QUICKSTART.md](CANONICAL_CLOSED_LOOP_QUICKSTART.md)、[CANONICAL_MULTI_STAGE_RUNBOOK_20261007.md](CANONICAL_MULTI_STAGE_RUNBOOK_20261007.md)
- 大文件/模型清单：[FINAL_MODEL_AND_REPRODUCTION_MANIFEST_20261007.md](FINAL_MODEL_AND_REPRODUCTION_MANIFEST_20261007.md)（Git只保存清单，大文件留主节点）。
