# 当前交付摘要

已完成 Re100、L/D=5 串列双圆柱的真实反馈闭环：PPO在FNO代理环境学习，部署时由 **CPU PPO＋真实OpenFOAM** 接收观测、输出旋转动作；无在线FNO/MPC。

以下为已核原始受力的主窗结果。RMS降低指后柱升力去均值波动降低；偏置为绝对均值/相同初始状态、无旋转对照的升力波动RMS。

| 已完成验证 | 减阻 | RMS降低 | 偏置 |
|---|---:|---:|---:|
| [E095：默认B，b01](CANONICAL_B01_REPRODUCTION_TERMINAL_REVIEW_20261007.md) | 4.0091% | 18.2810% | 3.6366% |
| [E109：B后续时段](P064_B_FUTURE_TIME_CFD_TERMINAL_REVIEW_20261007.md) | 3.9949% | 18.3412% | 2.8533% |
| [E114：B同分支连续延长至160 D/U](P064_B_CONTINUATION_328_408_TERMINAL_REVIEW_20261007.md) | 4.1326% | 17.9229% | 1.2352% |
| [E111：Absolute64探索，b01](P064_ABSOLUTE64_B01_CFD_TERMINAL_REVIEW_20261007.md) | 4.1362% | 18.9804% | 1.6817% |

各项均按原减阻≥2%、RMS比≤1.05、偏置≤10%核验；E114还预注册并通过四个连续20 D/U块及joined160/140 D/U窗口。E109/E114不是同b01比较，E114是同一E109分支的延长，不是统计独立物理工况。Absolute64部分早窗/全窗偏置及动作平方代价较B差，不称全面优越或净节能，B仍默认。

**尚不能声称完整目标完成：**原预测验收仍FAIL；物理通过不等于代理高精度或普遍泛化。B04 late-state coverage目前仅完成train-only来源、801点动作表及旧normalization绑定的准备审查，未执行OpenFOAM、Curator转换或训练；后续各阶段须另行批准。

官方PhysicsNeMo提供FNO，Curator及HDF5Reader用于数据处理/读取，HydroGym环境接口与开源SB3承接环境/PPO；时钟、双网络拼接、奖励、数据适配和安全处理是项目代码，不冒称官方内置功能。看板已有真实场图/预测对比均保留原候选标签，不代表Absolute64或E112的新场图。Spark训练保留至少20GiB统一内存。

## 怎么复现

起点是Spark已有真实curated数据、预训练K1与运行环境，**不是从零训练**。在Spark仓库执行只读预检：

```sh
cd /workspace/fluid_control
python3 scripts/reproduce_canonical_closed_loop.py
```

见[安全入口](CANONICAL_CLOSED_LOOP_QUICKSTART.md)与[多段链路](CANONICAL_MULTI_STAGE_RUNBOOK_20261007.md)。旧批准仅为历史证据；新执行须新审批、输出及unit。Git不含全部大数据、模型、镜像和环境。

Mac已配置转发时打开 <http://localhost:8766/> 查看实际状态、800点动作/Cd/Cl曲线；打不开不等于科研任务停止。
