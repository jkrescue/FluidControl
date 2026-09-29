# 串列双圆柱流动控制 CFD 工况与数据说明

更新日期：2026-09-29

## 1. 文档范围

本文档说明串列双圆柱流动控制项目的物理工况、数值方法、数据组成、质量验证及第一阶段 PhysicsNeMo 模型结果。环境安装和逐步执行命令见项目根目录的 [`train_recipe.md`](../../train_recipe.md)。

第一阶段采用 OpenFOAM 独立生成 CFD 数据。参考研究使用自研 GPU LBM，因此本项目结果属于独立物理复现，不代表原作者代码复跑，也不等同于复现论文报告的控制效果。

第一阶段状态如下：

| 环节 | 状态 |
| --- | --- |
| 无控制及旋转代表工况网格验证 | 通过 |
| 高转速 `q=±2、±2.5` 数值验证 | 通过；`q=±2.5` 为网格门槛边缘 |
| 恒定转速 CFD 轨迹 | 5/5 通过 |
| 动态转速 CFD 轨迹 | 16/16 通过 |
| VTK 导出 | 21/21 通过 |
| Curator HDF5 数据集 | 21/21 通过 |
| PhysicsNeMo DataPipe | 通过 |
| PhysicsNeMo FNO 训练 | 50/50 Epoch 完成 |
| 10 步 rollout 微调 | 20/20 Epoch 完成 |
| 独立测试轨迹 rollout | 1、10、50、100 步均稳定 |

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
| 容器镜像 | `opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319` |
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

### 6.4 参考论文无控制工况

Zhao 等（*Ocean Engineering*, 2024, 118138）在 `Re=100、L/D=5` 下报告后圆柱平均 `Cd=0.91`、平均 `|Cl|=1.07`、最大 `|Cl|=1.69`。OpenFOAM 无控制工况在 `t=80..160` 的结果如下。

| 后圆柱统计量 | 参考论文 | OpenFOAM | 相对差异 |
| --- | ---: | ---: | ---: |
| 平均 `Cd` | 0.91 | 0.909245 | 0.083% |
| 平均 `|Cl|` | 1.07 | 1.069066 | 0.087% |
| 最大 `|Cl|` | 1.69 | 1.647281 | 2.528% |

该结果支持代表性无控制流动的一致性。论文使用 CUDA 加速 D2Q9 MRT 多块 LBM，本项目使用 OpenFOAM 有限体积法，因此属于独立数值复现。完整对照与控制阶段边界见[论文复现对照](../../docs/PAPER_REPRODUCTION.md)。

### 6.5 恒定旋转响应

`t=80..160` 的恒定转速轨迹表明，后圆柱旋转已经对流场和受力产生稳定响应。

| `omega` | 表面速度比 `q` | 后柱平均 `Cd` | 后柱平均 `Cl` | 后柱 `Cl RMS` |
| ---: | ---: | ---: | ---: | ---: |
| -1.0 | -0.50 | 0.814664 | +0.994808 | 1.516810 |
| -0.5 | -0.25 | 0.883916 | +0.497500 | 1.272839 |
| 0 | 0 | 0.909391 | -0.000734 | 1.179131 |
| +0.5 | +0.25 | 0.885694 | -0.495786 | 1.271655 |
| +1.0 | +0.50 | 0.813922 | -1.003043 | 1.516647 |

正负转速产生近似反对称平均升力。恒定旋转降低平均阻力，但在当前范围内提高 `Cl RMS`。该结果验证了旋转边界条件，没有构成论文闭环升力抑制结果；闭环策略需要根据流动相位动态调整转速。

### 6.6 高转速数值验证

为扩展论文相关动作域，从静止基线 `t=80` 重启了 `q=±2、±2.5`（`omega=±4、±5`）恒定旋转算例。全部短试验均正常到达 `t=100`，日志包含 `End`，受力有限，粗网格 `Δt=0.005` 的最大 Courant 数分别约为 0.42 和 0.49。

`q=±2.5` 的时间步检查采用粗网格，并比较 `Δt=0.005` 与 `0.0025`。在 `t=90..100` 内，减半时间步后两侧后柱 `Cl RMS` 分别变化 0.45% 和 0.33%，最大 Courant 数降至约 0.245。因此粗网格批量数据继续采用 `Δt=0.005`。

空间检查将粗、中网格均以 `Δt=0.0025` 延长至 `t=160`，统计窗为 `t=120..160`：

| `q` | 粗网格 `Cl RMS` | 中网格 `Cl RMS` | 相对差异 | 粗/中网格主频 |
| ---: | ---: | ---: | ---: | ---: |
| -2.5 | 6.973804 | 7.186560 | 3.051% | 0.174989 / 0.174989 |
| +2.5 | 6.953958 | 7.143287 | 2.723% | 0.174989 / 0.174989 |

两侧 `Cl RMS` 网格差的平均值为 2.887%，最大值为 3.051%。中等网格下正负转速的 `Cl RMS` 幅值差为 0.604%，表明统计窗已得到良好的符号对称性。平均阻力接近零，其相对差异对分母敏感；粗、中网格的阻力标准差差异分别为 3.418% 和 2.637%。

该结果处于 3% 门槛边缘。扩展数据采用以下多保真策略：粗网格 `Δt=0.005` 用于批量训练轨迹；中等网格 `Δt=0.0025` 作为高转速独立数值验证及最终控制策略 CFD 回放环境。不同网格样本不得在缺少网格标识的情况下混入同一训练数据集。

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

动作覆盖审计显示，训练集最大 `|domega/dt|=0.5`；`dynamic_test_01` chirp 的最大值为 0.7624，高出 52.5%。全部 test 动作间隔中有 5% 超出训练变化率上限，该轨迹用于检验变化率外推，不作为训练数据。

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
| WSL 启动时报 `getpwuid` 与 I/O 错误 | 承载 `ext4.vhdx` 的宿主 D 盘空间耗尽 | 无损迁移 Windows 缓存目录并保留目录联接；训练产物迁至 C 盘，CFD/HDF5 仍保留在 ext4 |

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

10 步自回归微调从 Epoch 50 模型初始化，teacher forcing 在前 10 Epoch 由 0.5 降至 0，共训练 20 Epoch。最佳 checkpoint 为 Epoch 20；验证集 10 步平均流场 MAE 为 `0.00225494`，末端流场 MAE 为 `0.00366929`，平均/末端后柱受力 MAE 为 `0.00582447/0.00749916`。GPU0/1 峰值显存为 52,002/52,178 MiB，训练退出码为 0。

### 9.4 单步基线初始 rollout

冻结 Epoch 50 checkpoint 在 3 条独立测试轨迹上完成初始 1、10、50 步 rollout。所有片段均保持数值稳定，无失败片段。该结果保留为第一版验收记录；最终模型对照见下一段固定步长的严格评估。

| 步数 | 片段数 | 场 MAE | 相对 persistence 改善 | 后柱受力 MAE | 相对 persistence 改善 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,400 | 0.0006102 | 93.03% | 0.0052400 | 92.11% |
| 10 | 240 | 0.0047165 | 94.30% | 0.0131261 | 97.90% |
| 50 | 48 | 0.0160685 | 84.52% | 0.0489211 | 93.97% |

初始评估生成 27 张 `u/v/p` Ground Truth、Prediction 和 Absolute Error 对比图。动态测试轨迹的高频误差随 rollout 累积；难度最高的 `dynamic_test_01` 在 50 步时，场 MAE 为 0.0218156，后柱受力 MAE 为 0.0760545。

上述结果表明第一阶段动作条件代理模型通过既定测试，不代表闭环控制效果已经得到验证。

采用固定 5 帧起点间隔的严格对照进一步检查了 1/10/50/100 步，共 480/477/453/423 个窗口。多步模型相对单步基线的结果如下。

| 步数 | 流场 MAE | 流场变化 | 后柱受力 MAE | 受力变化 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00057276 | -6.17% | 0.00546587 | +0.81% |
| 10 | 0.00426863 | -9.47% | 0.00807204 | -38.37% |
| 50 | 0.01327908 | -16.80% | 0.03627153 | -29.35% |
| 100 | 0.02922517 | -24.97% | 0.07109247 | -26.68% |

所有严格测试窗口保持稳定。恒定 `omega=+0.5` 工况的 100 步受力误差增加 17.34%，两条动态测试轨迹分别改善 30.56% 和 32.45%。评估生成 36 张物理单位流场真值、预测和绝对误差对比图。

### 9.5 ParaView 流场导出

OpenFOAM 原生流场已按 7 个工况、5 个物理时刻打包为 35 个 `.vtu`，恒定转速覆盖 `omega=-1、-0.5、0、0.5、1`，并包含两条动态测试轨迹。多步模型在 3 条独立 test 轨迹上从 frame 100 自回归 100 步，按 step 1 和每 10 步输出 33 个 `.vtr`。

预测 VTK 在同一规则网格内保存真值、预测和绝对误差的 `U/p`，并保存掩码、转速、时间及后柱真实/预测 `Cd/Cl`。全部 68 个 VTK 数据文件已由 PyVista 回读，结果为 `PARAVIEW_VTK_EXPORT_OK`。生成及检查命令见 [`train_recipe.md`](../../train_recipe.md) 第 11 节。

Mac 端另保留完整连续序列：`dynamic_train_00.vtm.series` 包含 801 个训练 CFD 时刻，三条预测 `.pvd` 各包含 100 个连续 rollout VTU。上述四个入口已使用 ParaView 6.1.0 直接打开，时间轴和字段检查通过。

## 10. 适用范围与后续扩展

现有数据适用于：

- 单一 `Re=100`、单一 `L/D=5`、二维层流条件下的动作条件短时流场预测；
- OpenFOAM、Curator、HDF5、PhysicsNeMo DataPipe 和 FNO 的完整链路验证；
- 未见动作轨迹上的 1、10、50 步滚动预测评估；
- 同一物理工况下的代理模型 MPC 原型研究。

现有数据不覆盖跨雷诺数、跨圆心距、跨几何、三维湍流、电机真实功耗、结构疲劳寿命及工程安全认证。跨来流速度研究需要增加多雷诺数 CFD 轨迹，并将 `U∞/Re` 作为模型条件。闭环控制结论需要通过独立 OpenFOAM 回放或实验验证，并同时报告控制收益、动作约束、控制代价和鲁棒性。

参考论文允许的无量纲表面速度比为 `[-6,6]`，收敛策略的代表范围约为 `[-2.21,2.07]`；当前已训练代理仍只覆盖 `[-0.5,0.5]`。高转速 CFD 数值门槛已验证到 `q=±2.5`，但这些试验尚未加入代理训练集。因此现有代理不能用于论文动作域内的闭环结论。

后续按以下顺序执行：

1. 在已通过数值验证的 `q∈[-2.5,2.5]` 范围生成动态 train/validation/test 轨迹，并保证训练集覆盖测试动作变化率；
2. 使用中等网格 `Δt=0.0025` 保留高转速独立验证轨迹，不与粗网格训练样本直接混合；
3. 重新执行 Curator、DataPipe、FNO 单步训练、多步微调及分动作区间评估；
4. 再实现论文的 32 探针观测、控制代价和 PPO，并在独立 OpenFOAM 中闭环回放；
5. 使用升力抑制率、阻力、动作代价和稳定性与论文进行同指标比较。

## 11. 产物与证据索引

| 产物或日志 | 内容 |
| --- | --- |
| `artifacts/tandem_cylinders/dynamic_dataset_manifest.json` | 16 条动态轨迹的质量检查清单 |
| `artifacts/tandem_cylinders/action_coverage.json` | 各划分动作幅值与变化率覆盖审计 |
| `artifacts/tandem_cylinders/control_small_z000_stats.json` | 无控制后柱受力统计及论文数值对照依据 |
| `artifacts/tandem_cylinders/rotation_grid_comparison.json` | 旋转代表工况的粗、中网格比较 |
| `artifacts/tandem_cylinders/high_rotation_pilots_dt005.json` | `q=±2、±2.5` 粗网格短时稳定性检查 |
| `artifacts/tandem_cylinders/high_rotation_pilots_dt0025.json` | `q=±2.5` 时间步减半检查 |
| `artifacts/tandem_cylinders/high_rotation_grid_comparison.json` | `q=±2.5` 长时间窗粗、中网格比较 |
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
| `artifacts/tandem_fno_rollout/best/` | 10 步微调最佳 PhysicsNeMo checkpoint |
| `artifacts/tandem_fno_rollout/evaluation_dense.json` | 多步模型严格 test rollout 结果 |
| `artifacts/tandem_fno_rollout/baseline_comparison.json` | 单步基线与多步模型对照 |
| `artifacts/tandem_fno_rollout/rollout_visualizations/` | 36 张多步模型流场对比图 |
| `artifacts/tandem_fno_rollout/EVIDENCE_SHA256SUMS` | 最佳模型、配置、指标和 36 张图片的 SHA-256 清单 |
| `artifacts/tandem_paraview/` | OpenFOAM 原生场和 PhysicsNeMo 预测的 ParaView 时间序列 |
| `artifacts/tandem_paraview/validation.json` | 68 个 VTK 数据文件的回读检查结果 |
| `artifacts/tandem_fno/stage1_release/` | 第一阶段冻结清单与 SHA-256 校验 |

## 12. 源代码索引

| 文件或目录 | 内容 |
| --- | --- |
| `cases/<case>/` | OpenFOAM 配置、场、受力、探针和日志 |
| `make_baselines.py` | 基础几何、网格和静止基线生成 |
| `make_small_control_dataset.py` | 5 条恒定转速算例生成 |
| `make_dynamic_control_dataset.py` | 16 条动态动作算例生成 |
| `make_rotation_grid_check.py` | `omega=+1` 中等网格算例生成 |
| `make_high_rotation_pilots.py` | 高转速粗/中网格及时间步试验生成 |
| `run_high_rotation_pilot.sh` | 防覆盖的高转速试验执行入口 |
| `extend_high_rotation_pilot.sh` | `q=±2.5` 算例长时间窗续算入口 |
| `analyze_high_rotation_pilots.py` | 稳定性、Courant 数、受力和主频统计 |
| `compare_high_rotation_pilots.py` | 高转速粗、中网格统计比较 |
| `run_openfoam.sh` | 固定容器环境下的 OpenFOAM 执行入口 |
| `validate_small_dataset.py` | 恒定转速轨迹校验 |
| `validate_dynamic_dataset.py` | 动态轨迹校验与 manifest 生成 |
| `compare_convergence.py` | 粗、中网格统计比较 |
| `../../scripts/export_tandem_vtk.sh` | VTK 导出 |
| `../../scripts/curate_tandem_cfd.py` | Curator 数据处理 |
| `../../src/fluid_control/tandem_datapipe.py` | PhysicsNeMo DataPipe |
| `../../scripts/train_tandem_fno.py` | FNO 训练 |
| `../../scripts/evaluate_tandem_fno.py` | 独立测试与可视化 |
| `../../scripts/train_tandem_fno_rollout.py` | 10 步自回归微调 |
| `../../scripts/compare_rollout_evaluations.py` | 单步基线与多步模型误差对照 |
| `../../scripts/audit_tandem_actions.py` | 动作幅值与变化率覆盖审计 |
| `../../scripts/package_tandem_cfd_vtk.py` | 代表性 OpenFOAM VTK 时间序列打包 |
| `../../scripts/export_tandem_prediction_vtk.py` | PhysicsNeMo 预测场 VTK 导出 |
| `../../scripts/validate_tandem_vtk_export.py` | CFD 与预测 VTK 回读检查 |
| `../../scripts/control_tandem_mpc.py` | 代理模型 MPC 原型 |
| `../../conf/tandem_fno.yaml` | FNO 配置 |
| `../../conf/tandem_fno_rollout.yaml` | 多步微调配置 |
| `../../conf/tandem_mpc.yaml` | MPC 配置 |
| `../../train_recipe.md` | 环境配置与完整执行命令 |
| `../../closed_loop_control_spec.md` | 第二阶段闭环控制规范 |
