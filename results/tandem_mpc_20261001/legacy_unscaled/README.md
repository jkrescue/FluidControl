# 动作尺度审计记录

首次 MPC 实现没有读取扩展数据 manifest 中的 `max_abs_omega=5`，直接把物理转速送入 FNO。`selected_actions.csv` 和 `cfd_replay_result.json` 保留该次运行的原始证据，仅用于追踪问题，不参与控制器选择、结果表或论文结论。
