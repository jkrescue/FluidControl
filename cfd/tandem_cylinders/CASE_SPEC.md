# 串列双圆柱流动控制 CFD 工况与数据说明

更新日期：2026-09-29

## 1. 文档范围

本文档说明串列双圆柱流动控制项目的物理工况、数值方法、数据组成、质量验证及第一阶段 PhysicsNeMo 模型结果。环境安装和逐步执行命令见项目根目录的 [`train_recipe.md`](../../train_recipe.md)。

第一阶段采用 OpenFOAM 独立生成 CFD 数据。参考研究使用自研 GPU LBM，因此本项目结果属于独立物理复现，不代表原作者代码复跑，也不等同于复现论文报告的控制效果。

第一阶段状态如下：

| 环节 | 状态 |
| --- | --- |
| 无控制及旋转代表工况网格验证 | 通过 |
| 恒定转速 CFD 轨迹 | 5/5 通过 |
| 动态转速 CFD 轨迹 | 16/16 通过 |
| VTK 导出 | 21/21 通过 |
| Curator HDF5 数据集 | 21/21 通过 |
| PhysicsNeMo DataPipe | 通过 |
| PhysicsNeMo FNO 训练 | 50/50 Epoch 完成 |
| 独立测试轨迹 rollout | 1、10、50 步均稳定 |

## 2. 研究对象与阶段目标

均匀来流依次经过两根等直径圆柱。两根圆柱的中心位置固定，前圆柱保持静止，后圆柱绕自身轴线转动。

```text
均匀来流 U∞ →

    前圆柱                         后圆柱
   固定且不转       L/D = 5       固定中心并自转
       ○  ───────────────────────────  ↻ ○  ───→ 尾流
   (10D, 7.5D)                    (15D, 7.5D)
```

第一阶段建立动作条件流场代理模型。模型以当前流场和后圆柱转速为输入，预测下一时刻的速度场、压力场及后圆柱受力，并在未见动作轨迹上进行多步滚动预测。

长期目标是形成在线闭环主动流动控制系统，根据来流和尾流观测实时调整后圆柱转速。候选控制指标包括平均阻力、升力脉动、尾流模态能量及控制代价。结构减振目标还需要结构动力学或流固耦合数据支持。

现阶段 FNO 使用完整流场作为状态输入，仅承担流动物理代理功能。工程闭环还需要稀疏传感器、状态估计、控制器以及独立 CFD 或实验回放验证。现有 32 个尾流速度探针可用于后续稀疏观测研究。

## 3. 物理工况

| 项目 | 设置 |
| --- | --- |
| 流动模型 | 二维、不可压缩、层流 |
| 圆柱直径 | `D=1` |
| 来流速度 | `U∞=1` |
| 密度 | `rho=1` |
| 运动黏度 | `nu=0.01` |
| 雷诺数 | `Re=U∞D/nu=100` |
| 计算域 | `30D × 15D`，挤出厚度 `0.1D` |
| 前圆柱中心 | `(10D, 7.5D)` |
| 后圆柱中心 | `(15D, 7.5D)` |
| 圆心距 | `L/D=5` |
| 入口 | `U=(1,0,0)`；压力零梯度 |
| 出口 | 速度零梯度；运动学压力 `p=0` |
| 上下边界 | `slip` |
| 前后表面 | `empty` |
| 前圆柱 | 固定无滑移壁面 |
| 后圆柱 | `rotatingWallVelocity`；角速度 `omega(t)` |

后圆柱表面速度比定义为：

```text
q = omega D / (2 U∞) = omega / 2
```

第一阶段动作范围 `omega∈[-1,1]`，对应 `q∈[-0.5,0.5]`。

## 4. 求解器与数值设置

| 项目 | 设置 |
| --- | --- |
| 软件 | OpenCFD OpenFOAM v2512 |
| 求解器 | `pimpleFoam` |
| 容器镜像 | `opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae6698c489293ba30c991380fe3f899c622f319` |
| 容器限制 | 禁止网络、只读根文件系统、非 root、4 CPU、8 GiB 内存、无 GPU |
| 时间格式 | 二阶 `backward` |
| 时间步 | `Δt=0.005D/U∞` |
| 对流项 | `Gauss linearUpwind grad(U)` |
| 压力速度耦合 | PIMPLE |
| 全场输出间隔 | `0.1D/U∞` |
| 力与探针输出间隔 | 每个 CFD 时间步 |

`run_openfoam.sh` 固定容器镜像和资源限制。每个算例保留边界条件、`controlDict`、`fvSchemes`、`fvSolution`、求解日志及 `case_config.json`。

### 4.1 网格

| 网格 | 单元数 | 每个圆柱壁面面元数 | 用途 |
| --- | ---: | ---: | --- |
| 粗网格 | 19,290 | 96 | 第一阶段数据生成 |
| 中等网格 | 77,160 | 192 | 代表工况空间敏感性验证 |

两套网格均通过 `checkMesh`，结果为 `Mesh OK`。粗网格用于低雷诺数二维代理模型数据生成，不作为工程高保真 DNS 网格。

## 5. 原始数据组成

| 数据 | 原始位置 | 采样间隔 | 用途 |
| --- | --- | --- | --- |
| 速度场 `U=(u,v,0)` | `<time>/U` | `0.1` | 模型状态与标签 |
| 运动学压力 `p` | `<time>/p` | `0.1` | 模型状态与标签 |
| 前圆柱 `Cd/Cl` | `postProcessing/forceFront/` | `0.005` | 物理校验 |
| 后圆柱 `Cd/Cl` | `postProcessing/forceRear/` | `0.005` | 主要受力标签 |
| 32 点尾流速度 | `postProcessing/wakeProbes/` | `0.005` | 尾流校验与稀疏观测 |
| 动作 `omega(t)` | `case_config.json` 和速度边界 | 完整动作表 | 条件输入与追溯 |
| 求解诊断 | `log.pimpleFoam` | 每步 | Courant 数、连续性与结束状态 |

力系数参考面积为 `D × 0.1D=0.1D²`。前、后圆柱分别积分受力。

## 6. 数值质量验证

### 6.1 轨迹完整性检查

所有轨迹执行以下检查：

- 求解日志以 `End` 正常结束；
- 求解步数、场快照数、力记录数和探针记录数符合配置；
- 最大 Courant 数小于 1；
- 每步全局连续性误差绝对值小于 `1e-5`；
- 速度、压力、力和探针数据不含 NaN 或 Inf；
- 动作表覆盖完整时间窗且未超出声明范围。

这些检查用于确认数据完整性和数值健康状态，不构成与参考论文 LBM 结果的一致性证明。

### 6.2 无控制网格验证

粗网格与中等网格采用相同时间格式、时间步和 `t=80..160` 统计窗。

| 相对差异 | 前圆柱 | 后圆柱 |
| --- | ---: | ---: |
| 平均阻力 `Cd_mean` | 0.4635% | 0.0476% |
| 升力脉动 `Cl_rms` | 1.5567% | 0.5012% |
| 主频 `St` | 0.8055% | 0.8001% |

32 个探针的最大绝对差如下：

| 探针统计 | `u` | `v` |
| --- | ---: | ---: |
| 时间均值 | 0.01720 | 0.00807 |
| RMS | 0.01542 | 0.01220 |

主要受力统计差异均低于 3% 验收门槛。

### 6.3 旋转工况网格验证

`omega=+1` 代表工况在粗网格和中等网格上各使用 16,001 个统计样本，统计窗为 `t=80..160`。

| 相对差异 | 前圆柱 | 后圆柱 |
| --- | ---: | ---: |
| 平均阻力 `Cd_mean` | 0.4982% | 0.1560% |
| 升力脉动 `Cl_rms` | 2.4643% | 1.0616% |
| 主频 `St` | 0.7945% | 0.7928% |

探针最大绝对差为：时间均值 `u=0.02451、v=0.00673`，RMS `u=0.02399、v=0.01367`。主要受力统计差异均低于 3%。完整结果保存在 `artifacts/tandem_cylinders/rotation_grid_comparison.json`。

## 7. 数据集设计

### 7.1 恒定转速轨迹

| 算例 | `omega` | 划分 | 原始时间范围 | 使用范围 |
| --- | ---: | --- | --- | --- |
| `control_small_m100` | -1.0 | train | `0..160` | `80..160` |
| `control_small_z000` | 0.0 | train | `0..160` | `80..160` |
| `control_small_p100` | +1.0 | train | `0..160` | `80..160` |
| `control_small_m050` | -0.5 | validation | `0..160` | `80..160` |
| `control_small_p050` | +0.5 | test | `0..160` | `80..160` |

每条轨迹包含 32,000 个求解步、1,600 个速度/压力快照、前后圆柱各 32,000 条力记录及 32 个探针的 32,000 个时刻，单条数据约 4 GiB。5 条轨迹均通过质量检查。

### 7.2 动态转速轨迹

动态轨迹从无控制粗网格解 `t=80` 重启，仅改变后圆柱转速。

| 划分 | 数量 | 动作设计 |
| --- | ---: | --- |
| train | 8 | 每 4 个时间单位换档的随机线性 ramp，档位为 `-1,-0.5,0,0.5,1` |
| train | 4 | 不同幅值、周期和相位的 multisine |
| validation | 1 | 四分之一档位 ramp：`-0.75,-0.25,0.25,0.75` |
| validation | 1 | 独立 multisine |
| test | 1 | 独立四分之一档位 ramp |
| test | 1 | 幅值和频率随时间变化的 chirp |

动作采用 OpenFOAM `Function1 table` 线性插值，范围为 `[-1,1]`。每条轨迹覆盖 `t=80..160`，包含 16,000 个求解步、801 个场快照和 800 个相邻帧对。

16 条动态轨迹汇总如下：

| 项目 | 结果 |
| --- | ---: |
| train / validation / test | 12 / 2 / 2 条 |
| 总求解步数 | 256,000 |
| 总场快照数 | 12,816 |
| 总探针时刻数 | 256,000 |
| 最大 Courant 数 | 0.251260868 |
| 最大逐步全局连续性误差绝对值 | `1.93218635e-12` |
| 质量检查 | 16/16 通过 |
| 原始动态数据 | 约 32 GiB |

数据清单位于 `artifacts/tandem_cylinders/dynamic_dataset_manifest.json`。

### 7.3 数据划分

21 条轨迹统一使用 `t=80..160`。划分以完整轨迹为单位，避免相邻帧跨集合泄漏。

| 划分 | 轨迹数 | 每轨迹帧数 | 每轨迹相邻帧对 | 一步样本数 |
| --- | ---: | ---: | ---: | ---: |
| train | 15 | 801 | 800 | 12,000 |
| validation | 3 | 801 | 800 | 2,400 |
| test | 3 | 801 | 800 | 2,400 |
| 合计 | 21 | 16,821 | 16,800 | 16,800 |

## 8. PhysicsNeMo 数据与训练链路

| 阶段 | NVIDIA 组件 | 项目实现 |
| --- | --- | --- |
| VTK 读取与采样 | Curator `VTKSource`、PhysicsNeMo `Mesh.sample_data_at_points` | `scripts/curate_tandem_cfd.py` |
| 流水线执行 | Curator `Source`、`Filter`、`Sink`、`run_pipeline` | 轨迹 HDF5、manifest、训练集归一化统计 |
| 数据读取 | `HDF5Reader`、`DatasetBase`、`TensorDict`、PhysicsNeMo `DataLoader` | `src/fluid_control/tandem_datapipe.py` |
| 配置管理 | Hydra、OmegaConf | `conf/tandem_fno.yaml` |
| 模型 | `physicsnemo.models.fno.FNO` | `scripts/train_tandem_fno.py` |
| 分布式运行 | `DistributedManager` | 单 GPU及 `torchrun` 双 GPU |
| 训练与验证 | `StaticCaptureTraining`、`StaticCaptureEvaluateNoGrad` | FP32 训练与无梯度验证 |
| 日志与 checkpoint | `LaunchLogger`、`PythonLogger`、`save_checkpoint`、`load_checkpoint` | `.mdlus`、训练状态、配置和历史指标 |
| 独立测试 | PhysicsNeMo FNO、`load_checkpoint` | `scripts/evaluate_tandem_fno.py` |

PhysicsNeMo 2.2.2 的 FNO metadata 不支持 GPU AMP，训练采用 FP32。项目适配代码负责轨迹分组、动作与受力时间对齐、质量检查及结果输出，不替代 NVIDIA 提供的核心数据与模型接口。

## 9. 执行与验证记录

### 9.1 VTK 与 Curator

| 项目 | 结果 |
| --- | --- |
| 单轨迹 VTK 冒烟 | `dynamic_train_00` 导出 801 帧，约 2.2 GiB，日志正常结束 |
| 全量 VTK | 21/21 条轨迹，每条 801 帧，总量约 45 GiB |
| Curator 版本 | NVIDIA `physicsnemo-curator` 提交 `86533e581b3550326d89e97cb4d4126e7061b416` |
| Curator 运行环境 | Python 3.12.3，Curator 0.1.0，PhysicsNeMo 2.2.2 |
| Curator 主要依赖 | Rust/Cargo 1.98.1，NumPy 2.5.3，h5py 3.16.0，PyVista 0.49.0，VTK 9.7.1 |
| 训练运行环境 | NVIDIA 驱动 596.72，PyTorch 2.14.0+cu130，CUDA 13.0，NCCL 2.30.7 |
| 单轨迹 Curator 冒烟 | 801/801 帧，墙钟时间 4 分 21.06 秒，峰值内存约 2.22 GiB |
| 全量 Curator | 21 个 HDF5，共 5.5 GiB，划分 15/3/3 |
| 全量审计 | `CURATED_DATASET_OK`，退出码 0 |

官方单帧 API 验证读取到 39,336 个点、115,740 个四面体单元及 `p/U/TimeValue`。在 `64×32` 规则网格上，有效采样点为 2,024/2,048，覆盖率为 0.98828125。

HDF5 单轨迹结构如下：

| 数据集 | 形状 | 类型 |
| --- | --- | --- |
| `state` | `(801,3,128,256)` | float32 |
| `mask` | `(801,1,128,256)` | uint8 |
| `omega` | `(801,1)` | float32 |
| `force` | `(801,4)` | float32 |
| `time` | `(801,1)` | float32 |

时间轴从 80.0 到 160.0 严格递增；mask 有效覆盖率为 0.9869384765625；有效区域逐帧压力均值的最大绝对值为 `1.6456821227265347e-09`；无效区域状态为 0。

### 9.2 环境兼容性记录

| 问题 | 原因 | 处置 |
| --- | --- | --- |
| Curator 在 Python 3.11 导入失败 | 源码使用 PEP 695 泛型语法 | Curator 使用独立 Python 3.12 环境，训练环境保持 Python 3.11 |
| `nvidia-physicsnemo==2.2.2` 解析冲突 | Curator 的 uv source 指向 PhysicsNeMo GitHub HEAD | 安装时使用 `--no-sources-package nvidia-physicsnemo` |
| `sklearn==0.0.post12` 构建被拒绝 | 传递依赖使用弃用占位包 | 安装期间设置 `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True` |
| WSL 双 GPU NCCL 报错 `Cuda failure 999` | GPU P2P 不可用，默认传输初始化失败 | 关闭 P2P、IB 和 cuMem device/host allocation，保留 SHM 通信 |

Curator 安装与验证日志位于：

- `artifacts/tandem_cylinders/curator_install_py312.log`
- `artifacts/tandem_cylinders/curator_import_py312.log`
- `artifacts/tandem_cylinders/curator_mesh_extra_install.log`
- `artifacts/tandem_cylinders/curator_mesh_extra_install_retry2.log`
- `artifacts/tandem_cylinders/curator_mesh_extra_install_retry3.log`
- `artifacts/tandem_cylinders/curator_official_vtk_api_check.log`
- `artifacts/tandem_cylinders/curator_smoke.log`
- `artifacts/tandem_cylinders/curator_full.log`
- `artifacts/tandem_cylinders/curated_validation.log`

NCCL 最小 all-reduce 在 SHM 配置下通过，两个 rank 的结果均为 3.0。Socket 传输作为故障回退保留。相关日志位于 `artifacts/tandem_cylinders/nccl_shm_check.log` 和 `nccl_socket_check.log`。

### 9.3 DataPipe 与训练

DataPipe 审计得到 12,000/2,400/2,400 个 train/validation/test 窗口。输入张量形状为 `(6,128,256)`，场增量标签为 `(3,128,256)`，受力标签为 `(2,)`，连续 batch 均为有限值。

双 GPU 冒烟测试完成 1 Epoch，训练 loss 为 `0.2025793`，验证场 MAE 为 `0.00336758`，RMSE 为 `0.00516719`，归一化后柱受力 MAE 为 `0.746278`。单卡吞吐约 114.8 samples/s；双卡 SHM 全局吞吐约 180.0 samples/s，并行效率约 78.4%。

正式训练完成 50 Epoch，退出码为 0。Epoch 50 为验证场 MAE 最佳 checkpoint：

| 指标 | 结果 |
| --- | ---: |
| 场 MAE | `5.3888804e-04` |
| 场 RMSE | `7.9748279e-04` |
| 归一化后柱受力 MAE | `8.5806989e-03` |
| 后柱受力 MAE 单项最低值 | Epoch 42：`8.4354119e-03` |

正式训练从 checkpoint 恢复后采用每 rank batch size 64、全局 batch size 128，并保留 optimizer 和 scheduler 状态。

### 9.4 独立测试轨迹 rollout

冻结 Epoch 50 checkpoint 在 3 条独立测试轨迹上完成 1、10、50 步 rollout。所有片段均保持数值稳定，无失败片段。

| 步数 | 片段数 | 场 MAE | 相对 persistence 改善 | 后柱受力 MAE | 相对 persistence 改善 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,400 | 0.0006102 | 93.03% | 0.0052400 | 92.11% |
| 10 | 240 | 0.0047165 | 94.30% | 0.0131261 | 97.90% |
| 50 | 48 | 0.0160685 | 84.52% | 0.0489211 | 93.97% |

评估生成 27 张 `u/v/p` Ground Truth、Prediction 和 Absolute Error 对比图。动态测试轨迹的高频误差随 rollout 累积；难度最高的 `dynamic_test_01` 在 50 步时，场 MAE 为 0.0218156，后柱受力 MAE 为 0.0760545。

上述结果表明第一阶段动作条件代理模型通过既定测试，不代表闭环控制效果已经得到验证。

## 10. 适用范围与后续扩展

现有数据适用于：

- 单一 `Re=100`、单一 `L/D=5`、二维层流条件下的动作条件短时流场预测；
- OpenFOAM、Curator、HDF5、PhysicsNeMo DataPipe 和 FNO 的完整链路验证；
- 未见动作轨迹上的 1、10、50 步滚动预测评估；
- 同一物理工况下的代理模型 MPC 原型研究。

现有数据不覆盖跨雷诺数、跨圆心距、跨几何、三维湍流、电机真实功耗、结构疲劳寿命及工程安全认证。跨来流速度研究需要增加多雷诺数 CFD 轨迹，并将 `U∞/Re` 作为模型条件。闭环控制结论需要通过独立 OpenFOAM 回放或实验验证，并同时报告控制收益、动作约束、控制代价和鲁棒性。

## 11. 产物与证据索引

| 产物或日志 | 内容 |
| --- | --- |
| `artifacts/tandem_cylinders/dynamic_dataset_manifest.json` | 16 条动态轨迹的质量检查清单 |
| `artifacts/tandem_cylinders/rotation_grid_comparison.json` | 旋转代表工况的粗、中网格比较 |
| `artifacts/tandem_cylinders/vtk_export.log` | 全量 VTK 导出日志 |
| `artifacts/tandem_cylinders/vtk_counts.txt` | 21 条轨迹的 VTK 帧数检查 |
| `artifacts/tandem_cylinders/curator_official_vtk_api_check.log` | 官方 VTKSource 与 Mesh API 检查，标记 `OFFICIAL_VTK_MESH_API_OK` |
| `artifacts/tandem_cylinders/curator_smoke_hdf5_check.log` | 单轨迹 HDF5 结构与数值检查，标记 `SMOKE_HDF5_OK` |
| `data/curated/tandem_cylinders/manifest.json` | Curator 数据集文件及划分清单 |
| `data/curated/tandem_cylinders/normalization.json` | 仅基于训练集计算的归一化统计 |
| `artifacts/tandem_cylinders/curated_validation.json` | 全量 HDF5 独立审计，标记 `CURATED_DATASET_OK` |
| `artifacts/tandem_cylinders/datapipe_validation.log` | DataPipe 形状、样本数和有限值检查，标记 `PHYSICSNEMO_DATAPIPE_OK` |
| `artifacts/tandem_cylinders/gpu_diagnostic.log` | GPU、CUDA、NCCL 和 P2P 状态记录 |
| `artifacts/tandem_cylinders/nccl_shm_check.log` | 双 GPU SHM all-reduce 检查，标记 `NCCL_ALL_REDUCE_OK` |
| `artifacts/tandem_fno/environment.log` | 正式训练环境记录 |
| `artifacts/tandem_fno/best/` | Epoch 50 最佳模型与训练状态 |
| `artifacts/tandem_fno/training_history.json` | 逐 Epoch 训练与验证指标 |
| `artifacts/tandem_fno/evaluation.json` | 独立测试轨迹 rollout 结果 |
| `artifacts/tandem_fno/rollout_visualizations/` | 27 张流场对比图 |
| `artifacts/tandem_fno/stage1_release/` | 第一阶段冻结清单与 SHA-256 校验 |

## 12. 源代码索引

| 文件或目录 | 内容 |
| --- | --- |
| `cases/<case>/` | OpenFOAM 配置、场、受力、探针和日志 |
| `make_baselines.py` | 基础几何、网格和静止基线生成 |
| `make_small_control_dataset.py` | 5 条恒定转速算例生成 |
| `make_dynamic_control_dataset.py` | 16 条动态动作算例生成 |
| `make_rotation_grid_check.py` | `omega=+1` 中等网格算例生成 |
| `run_openfoam.sh` | 固定容器环境下的 OpenFOAM 执行入口 |
| `validate_small_dataset.py` | 恒定转速轨迹校验 |
| `validate_dynamic_dataset.py` | 动态轨迹校验与 manifest 生成 |
| `compare_convergence.py` | 粗、中网格统计比较 |
| `../../scripts/export_tandem_vtk.sh` | VTK 导出 |
| `../../scripts/curate_tandem_cfd.py` | Curator 数据处理 |
| `../../src/fluid_control/tandem_datapipe.py` | PhysicsNeMo DataPipe |
| `../../scripts/train_tandem_fno.py` | FNO 训练 |
| `../../scripts/evaluate_tandem_fno.py` | 独立测试与可视化 |
| `../../scripts/control_tandem_mpc.py` | 代理模型 MPC 原型 |
| `../../conf/tandem_fno.yaml` | FNO 配置 |
| `../../conf/tandem_mpc.yaml` | MPC 配置 |
| `../../train_recipe.md` | 环境配置与完整执行命令 |
| `../../closed_loop_control_spec.md` | 第二阶段闭环控制规范 |
