# DGX Spark 执行进展（2026-10-01）

迁移只改变运行环境；目标仍为串列双圆柱（Re=100、L/D=5）、后柱旋转 |omega|<=5、PhysicsNeMo 代理与 HydroGym 强化学习的闭环控制。旧 HDF5 和 FNO checkpoint 未恢复，因此不沿用旧模型性能数字。

## 当前已验证

- OpenFOAM v2512 ARM64 容器从无控制基线 t=80 重启，expanded_train_00 完成至 t=160 的 16,000 步求解，数值与 801 个全场快照检查通过。24/4/4 划分在生成前固定；其余轨迹正在独立求解。
- Curator 官方源码固定于 86533e581b3550326d89e97cb4d4126e7061b416，在项目 Python 3.12 虚拟环境源码构建成功，Source、Filter、Sink、VTKSource、run_pipeline 导入通过。README 标注 x86_64；ARM64 是本机实测，不代表官方支持承诺。
- PhysicsNeMo 2.2.2 隔离容器可识别 GB10；官方 HDF5Reader、Mesh API 与训练入口导入通过。完整 DataPipe/训练要等真实 HDF5 生成。
- HydroGym 官方源码固定于 4ab9854dea3d84e38a59c25e0f5835a00cf8225f。项目独立虚拟环境中，官方 JAX 谱方法 Kolmogorov 环境与 PPO 示例依赖安装、CPU import 通过；训练正在运行。该基准不等于串列双圆柱控制收益。

CFD 批量脚本为 cfd/tandem_cylinders/run_expanded_on_spark.sh；拒绝覆盖已有输出，默认 3 个 4 CPU/8 GiB OpenFOAM 容器任务，启动前要求至少 150 GiB 磁盘可用。完成后自动写 artifacts/tandem_cylinders/expanded_v1_cfd_qc.json。大型场数据和模型输出不进入 GitLab，代码、配置与摘要持续同步。

GB10 为 CPU/GPU 共享内存，nvidia-smi 显存列为 N/A。GPU 训练前需设置分配上限，并同时监控 CUDA free 与系统 MemAvailable，保证至少 20 GiB 余量。当前 JAX PPO 仅使用 CPU。

资料：[PhysicsNeMo](https://github.com/NVIDIA/physicsnemo)、[Curator](https://github.com/NVIDIA/physicsnemo-curator)、[HydroGym](https://github.com/dynamicslab/hydrogym)、[DGX Spark 已知问题](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)。
