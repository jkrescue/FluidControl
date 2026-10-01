# OpenFOAM → HydroGym 67 维观测只读接口（2026-10-02）

`src/fluid_control/openfoam_observation.py` 为后续真实 CFD 分段反馈提供只读接口 `observation_at(case, target_time, applied_omega)`。它从原始 OpenFOAM `postProcessing/wakeProbes/*/U` 与 `forceRear/*/coefficient.dat` 精确取同一时刻样本，按固定顺序输出 32×(u,v)、后柱 Cd/Cl、已施加转速，共 67 个 float32 值和来源文件清单。该模块不是 CFD 求解器、不是 PhysicsNeMo 模型，也不生成仿真数据。

接口逐文件核对 32 个探针编号及 (x=17, y=6..9, z=0.05) 坐标、Cd/Cl 列顺序、时间、有限值与 `|omega|<=5`；找不到目标时间立即失败。若 OpenFOAM 分段重启产生多个同一时间样本，必须在 1e-7 绝对误差内一致，否则拒绝返回，避免静默混用旧轨迹。单元测试使用明确标注为解析器夹具的简短文本，不作为科研数据；`python3 -m unittest discover -s tests -v` 共 7 个测试通过。

在 DGX Spark 已对真实 `expanded_validation_00` 与 `expanded_test_04` 的 t=90/120/160 六个时刻回归：与 Curator/HydroGym 观测布局的最大绝对差依次为 0.0042061、0.0034006、0.0020997、0.0021356、0.0044976、0.0010068，均低于既定 0.02 探针映射阈值。随后在固定 PhysicsNeMo 2.2.2 + HydroGym 容器中，官方 `FlowEnv` 第 100 帧的实际重置观测与此接口对齐：探针最大差 0.0042062，后柱受力和转速差为零，模型一步状态/受力/奖励的原有等价检验仍通过。结构化证据为 `docs/results/tandem_openfoam_observation_bridge_parity.json`；完整六帧原始观测布局校验见 `HYDROGYM_67D_OBSERVATION_PARITY_20261002.md`。

这解决的是在线闭环所需的观测读取接口；**尚未执行真实 OpenFOAM 分段反馈，也不能推导任何 RL 控制收益**。下一步必须在新的、不可覆盖的 CFD case 中，以同一初态分别运行冻结候选策略与零请求动作基线，逐步记录实际施加动作和数值健康，再比较真实 Cd、Cl RMS 与控制代价。
