# 串列双圆柱扩展 CFD 数据生成验收（2026-10-02）

DGX Spark 上的 OpenFOAM v2512 ARM64 隔离容器已完成 `tandem_cylinder_dynamic_rotation_expanded_v1` 的 **32/32 条真实受控 CFD 轨迹**，不是插值或伪造数据。场景为 Re=100、圆柱中心距 L/D=5、后柱转动 `|omega|<=5`；每条从同一已验证的无控制场 `t=80` 重启至 `t=160`，`dt=0.005`，共 16,000 个求解步和 801 个全场快照。动作日程按代码事先固定，训练/验证/测试为 24/4/4 条。

完整求解器配置、网格、边界条件、动作日程和生成命令见 `docs/CFD_DATA_GENERATION.md` 与 `cfd/tandem_cylinders/`。该阶段使用 19,290 单元粗网格；后续模型或控制收益不能在未做网格收敛与独立物理验证前宣称高精度。

主 QC 清单保存在 Spark `artifacts/tandem_cylinders/expanded_v1_cfd_qc.json`，同一小型清单已同步到 `docs/results/expanded_v1_cfd_qc.json`（SHA-256 `cd57df304df3a844a119e1689a653544f72e5fcc2fd89bde82dc11e3e7cb6972`）。32 条均通过数值检查：最小快照数 801，最大 Courant 数 0.476685879，最大单步全局连续性残差 7.32518077e-12；每条求解器正常结束且动作未越界。检查器还验证了尾时刻、探针和前后柱力系数样本数，以及验证/测试动作变化率未超出训练覆盖范围。

原始 OpenFOAM 场数据留在 `cfd/tandem_cylinders/cases/`，当前约 69 GiB，不进入 GitLab。Curator 正在用官方 VTKSource/PhysicsNeMo Mesh 流程导出和整理全场到 HDF5；首条 801 帧 HDF5 已通过逐帧有效值/掩码质检。**全量 Curator、PhysicsNeMo DataPipe、FNO 训练和留出测试尚未完成**；下一阶段流水线会等待全量质检标记，按顺序执行，不会把本 CFD 数值 QC 误写成代理模型性能。

进度快照（2026-10-02 02:50 CST）：Curator 已形成 **17/32 条**完整 HDF5。新增 `expanded_train_09..12` 四条均有 801 帧，通过逐帧有限值/掩码检查，且转速、前后柱 Cd/Cl 与原始 OpenFOAM 文件逐点对齐；该批最大转速误差 `2.29e-7`、最大力系数误差 `2.38e-7`（HDF5 float32 舍入量级）。其余数据仍在整理，FNO 训练未启动。

物理响应粗筛（17 条已完成 HDF5，2026-10-02 03:00 CST）：全部从同一无控制 `t=80` 场重启，后柱初始 `Cd/Cl` 与原始无控制时间序列的绝对差最大仅 `0.000506/0.007536`；在 `t>=120` 的后半窗，逐时刻平均绝对差的最小值为 `Cd 0.415`、`Cl 1.289`，且各条动作 RMS 为 `1.887..3.365`。可复查脚本为 `scripts/audit_curated_control_response.py`，完整 32 条会在训练前再次验收。该差异只排查动作未生效或复制数据，不是阻力/升力改善，更不是闭环控制收益。

进度快照（2026-10-02 03:15 CST）：Curator 已形成 **21/32 条**完整 HDF5。新增 `expanded_train_13..16` 四条各 801 帧，逐帧场与掩码、原始动作和前后柱 Cd/Cl 标签均通过质检；该批最大动作误差 `2.37e-7`、最大力系数误差 `2.37e-7`。累计 21 条也通过无控制共同起点与后半窗物理响应粗筛（最小后半窗平均绝对差仍为 `Cd 0.415`、`Cl 1.289`）。Curator 正处理 `expanded_train_17..20`；全量数据与 FNO 训练仍未完成。
