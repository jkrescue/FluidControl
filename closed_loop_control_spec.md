# 闭环控制说明

默认流程为固定 PPO 读取真实 CFD 观测、输出受限转速、推进 OpenFOAM，再取得下一观测。每次反馈推进 0.1 D/U∞；部署期间不运行 FNO，不继续训练 PPO。

观测、动作、奖励与 MPC 实现见[控制说明](guide/CONTROL.md)，验收指标与实测数据见[评估结果](research_records/RESULTS.md)。旧探索方案保存在[历史控制规范](https://github.com/jkrescue/FluidControl/blob/research-history-20261009/closed_loop_control_spec.md)。
