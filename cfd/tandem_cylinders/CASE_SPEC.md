# 串列双圆柱流动控制 CFD 工况与数据说明

更新日期：2026-09-29

## 阅读摘要

本算例研究两根等直径圆柱串列布置时，后圆柱自转如何改变尾流和后圆柱受力。工况固定为二维层流、`Re=100`、圆心距 `L/D=5`。CFD 使用 OpenFOAM 独立计算，目标是为第一阶段 PhysicsNeMo 动作条件流场模型生成可追溯的数据。

当前结论如下：

1. 无控制粗网格和中等网格的主要受力统计差异均小于预设的 3%，粗网格可用于第一阶段数据生成。
2. 5 条恒定转速轨迹已经完成并通过基本数值质量检查。
3. 16 条动态转速轨迹已经全部完成，并通过逐文件只读检查。
4. `omega=+1` 的中等网格算例已完成 32,000 步并正常写出 `End`；粗／中网格主要受力统计差异均小于 3%。
5. CFD、VTK、Curator、Datapipe、PhysicsNeMo FNO 50 Epoch 正式训练和独立 test rollout 均已完成，第一阶段已冻结。

这些结果属于使用 OpenFOAM 的独立物理复现。论文作者使用自研 GPU LBM，因此本项目不把现有结果称为作者代码复跑，也不宣称已经复现论文报告的控制效果。

## 1 研究问题

均匀来流从左向右依次经过前圆柱和后圆柱。两根圆柱的中心位置固定，只有后圆柱绕自身轴线转动。

```text
均匀来流 U∞ →

    前圆柱                         后圆柱
   固定且不转       L/D = 5       固定中心并自转
       ○  ───────────────────────────  ↻ ○  ───→ 尾流
   (10D, 7.5D)                    (15D, 7.5D)

主要输入：后圆柱角速度 omega(t)
主要响应：流场 U、p 和后圆柱 Cd、Cl
辅助观测：后圆柱下游 32 个速度探针
```

第一阶段模型要回答：给定当前流场和后圆柱转速，能否预测下一时刻的流场变化及后圆柱受力，并能否在未见过的动作轨迹上保持短期滚动预测稳定。

### 1.1 长期目标：不同来流下的实时在线闭环控制

项目的长期目标是建立实时闭环主动流动控制系统：系统根据当前来流和尾流状态，在线调整后圆柱转速，以优化阻力、升力脉动、尾流稳定性或结构振动相关指标。这个方向具有明确研究价值和潜在工程价值，可用于主动减阻、抑制涡脱落、降低周期载荷，以及为模型预测控制和强化学习控制提供快速环境。

目标闭环可概括为：

```text
来流和尾流观测
  -> 状态估计
  -> 控制器选择后圆柱转速 omega(t)
  -> 流动物理系统响应
  -> 新的流场、Cd、Cl 或结构响应
  -> 下一控制周期
```

当前 PhysicsNeMo FNO 只实现其中的“流动物理系统快速代理”：输入当前全流场和给定转速，预测下一时刻流场及后圆柱受力。它不会自行选择转速，因此当前成果是控制系统的预测模型，而不是闭环控制策略。

控制目标尚未固定，后续应从可测量且可验证的指标中选择，可考虑：

| 候选目标 | 可用指标 | 当前数据是否直接支持 |
| --- | --- | --- |
| 降低平均阻力 | 后圆柱或两柱的时间平均 `Cd` | 后圆柱标签直接支持；两柱联合目标需在训练数据中保留并使用前柱力 |
| 抑制升力波动 | `Cl_rms`、峰值或主频幅值 | 后圆柱标签直接支持 |
| 抑制尾流涡脱落 | 探针/流场频谱、涡量或 POD 模态能量 | 当前全场与探针可支持离线定义 |
| 降低控制代价 | `omega²`、转速变化率或功率模型 | `omega` 已有；真实电机功率需要额外模型或测量 |
| 减小结构振动 | 位移、速度、加速度、疲劳载荷 | 当前固定圆柱 CFD 不直接支持；`Cl_rms` 只能作为载荷代理，正式减振需要结构动力学或流固耦合 |

不同来流速度意味着在 `D` 和 `nu` 固定时雷诺数同时变化。当前数据只有 `U∞=1、Re=100、L/D=5`，不能支持跨来流速度控制。后续必须把 `U∞/Re` 加入模型条件，并在目标速度范围内补充多雷诺数、动态转速、完整轨迹级 train/validation/test CFD 数据。

真实在线系统通常不能直接获得完整 `u/v/p` 场，而当前 FNO 需要全场输入。工程闭环还需要使用压力传感器、尾流探针或其他稀疏观测，并增加状态估计器，或者直接训练基于稀疏观测的控制策略。现有 32 个尾流速度探针可作为第一版传感器观测候选。

建议的后续阶段为：

1. 完成并验证当前单一 `Re=100` 动作条件代理模型，包括独立 test 多步 rollout。
2. 在同一 `Re` 下选定一个明确目标，先做离线动作优化或模型预测控制，并加入转速幅值、变化率和控制能耗约束。
3. 将控制器接回独立 OpenFOAM 仿真逐控制周期回放，与无控制、恒定转速和预设周期动作比较，避免只在代理模型内自我验证。
4. 补充多来流速度数据，把 `U∞/Re` 作为条件变量，验证训练范围内插值和范围边界性能。
5. 将全场输入替换或重建为稀疏传感器观测，测量端到端推理延迟，形成实时数字闭环。
6. 如最终目标是结构减振，再加入结构自由度、载荷响应和流固耦合验证。

这一长期目标合理，但价值成立的条件是最终控制效果必须在未参与训练的 CFD 或真实系统中验证，并同时报告流动收益、控制能耗、约束违反和鲁棒性。当前阶段不宣称已经实现闭环控制或跨来流泛化。

## 2 固定物理工况

| 项目 | 设置 |
| --- | --- |
| 流动 | 二维、不可压缩、层流 |
| 圆柱直径 | `D=1` |
| 来流速度 | `U∞=1` |
| 密度 | `rho=1` |
| 运动黏度 | `nu=0.01` |
| 雷诺数 | `Re=U∞D/nu=100` |
| 计算域 | `30D × 15D`，二维挤出厚度 `0.1D` |
| 前圆柱中心 | `(10D, 7.5D)` |
| 后圆柱中心 | `(15D, 7.5D)` |
| 圆心距 | `L/D=5` |
| 入口 | `U=(1,0,0)`；压力零梯度 |
| 出口 | 速度零梯度；运动学压力 `p=0` |
| 上下边界 | `slip` |
| 前后表面 | `empty`，形成单层二维网格 |
| 前圆柱 | 固定无滑移壁面 |
| 后圆柱 | `rotatingWallVelocity`；角速度为 `omega(t)` |

后圆柱表面速度比定义为

```text
q = omega D / (2 U∞) = omega / 2
```

因此第一阶段的 `omega∈[-1,1]` 对应 `q∈[-0.5,0.5]`。

## 3 求解器和数值设置

### 3.1 求解工具

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
| 全场输出间隔 | 训练轨迹为 `0.1D/U∞` |
| 力和探针输出 | 每个 CFD 时间步 |

`run_openfoam.sh` 固定容器镜像和资源限制。所有算例都保留 `controlDict`、`fvSchemes`、`fvSolution`、边界条件、求解日志和单独的 `case_config.json`，便于核查实际设置。

### 3.2 网格

| 网格 | 单元数 | 每个圆柱的壁面面元 | 用途 |
| --- | ---: | ---: | --- |
| 粗网格 | 19,290 | 96 | 第一阶段全部训练轨迹 |
| 中等网格 | 77,160 | 192 | 代表工况空间敏感性检查 |

两套网格均通过 `checkMesh`，结果为 `Mesh OK`。粗网格不是高保真工程 DNS；它是通过当前统计门槛后用于第一阶段代理模型的受控计算成本方案。

## 4 每条轨迹实际保存什么

| 数据 | 原始位置 | 采样频率 | 用途 |
| --- | --- | --- | --- |
| 速度场 `U=(u,v,0)` | `<time>/U` | 每 `0.1` | 模型状态和流场标签 |
| 运动学压力 `p` | `<time>/p` | 每 `0.1` | 模型状态和流场标签 |
| 前圆柱 `Cd/Cl` | `postProcessing/forceFront/` | 每 `0.005` | 物理核查 |
| 后圆柱 `Cd/Cl` | `postProcessing/forceRear/` | 每 `0.005` | 主要受力标签 |
| 32 点尾流速度 | `postProcessing/wakeProbes/` | 每 `0.005` | 论文观测形式和流场核查 |
| 动作 `omega(t)` | `case_config.json` 和 `U` 边界 | 完整动作表 | 条件输入和追溯 |
| 求解器诊断 | `log.pimpleFoam` | 每步 | Courant 数、残差、连续性和正常结束检查 |

力系数参考面积为 `D × 0.1D=0.1D²`，与二维挤出厚度一致。前后圆柱的力分别积分，避免把总力误当成后圆柱受力。

## 5 数值验证证据

### 5.1 所有轨迹必须满足的基本检查

只读校验器逐条检查：

- 日志以 `End` 正常结束；
- 求解步数、场快照数、力系数行数和探针行数符合配置；
- 最大 Courant 数小于 1；
- 每步全局连续性误差绝对值小于 `1e-5`；
- `U`、`p`、力和探针中没有 NaN 或 Inf；
- 动作表覆盖完整时间窗，且动作不超出声明范围。

这些检查证明数据完整且求解过程数值健康。它们本身不证明与论文 LBM 结果完全一致。

### 5.2 无控制粗网格与中等网格比较

比较使用相同的二阶时间格式、`Δt=0.005` 和固定统计窗 `t=80..160`。

| 统计量的相对差异 | 前圆柱 | 后圆柱 |
| --- | ---: | ---: |
| 平均阻力 `Cd_mean` | 0.4635% | 0.0476% |
| 升力脉动 `Cl_rms` | 1.5567% | 0.5012% |
| 主频 `St` | 0.8055% | 0.8001% |

32 个探针的粗／中网格最大绝对差为：

| 探针统计 | `u` | `v` |
| --- | ---: | ---: |
| 时间均值 | 0.01720 | 0.00807 |
| RMS | 0.01542 | 0.01220 |

主要受力统计均低于预先设定的 3% 门槛。因此，无控制工况支持在第一阶段使用粗网格。

### 5.3 旋转壁面代表工况

`omega=+1` 的中等网格轨迹已完成，并与粗网格 `control_small_p100` 在 `t=80..160` 比较。每套网格都有 16,001 个统计样本，包含两个时间窗端点。

| 统计量的相对差异 | 前圆柱 | 后圆柱 |
| --- | ---: | ---: |
| 平均阻力 `Cd_mean` | 0.4982% | 0.1560% |
| 升力脉动 `Cl_rms` | 2.4643% | 1.0616% |
| 主频 `St` | 0.7945% | 0.7928% |

探针最大绝对差为：时间均值 `u=0.02451、v=0.00673`，RMS `u=0.02399、v=0.01367`。主要受力统计全部低于 3%，旋转代表工况通过第一阶段空间网格门槛。

完整结果保存到：

```text
artifacts/tandem_cylinders/rotation_grid_comparison.json
```

该门槛已经通过，可以进入 VTK 格式转换和 Curator 数据整理。

## 6 第一阶段数据集设计

### 6.1 恒定转速轨迹

已完成 5 条粗网格轨迹：

| 算例 | `omega` | 数据划分 | 原始范围 | 用于训练的范围 |
| --- | ---: | --- | --- | --- |
| `control_small_m100` | -1.0 | train | `t=0..160` | `t=80..160` |
| `control_small_z000` | 0.0 | train | `t=0..160` | `t=80..160` |
| `control_small_p100` | +1.0 | train | `t=0..160` | `t=80..160` |
| `control_small_m050` | -0.5 | validation | `t=0..160` | `t=80..160` |
| `control_small_p050` | +0.5 | test | `t=0..160` | `t=80..160` |

每条原始轨迹有 32,000 个求解步、1,600 个 `U/p` 快照、前后圆柱各 32,000 条力记录和 32 个探针的 32,000 个时刻，磁盘占用约 4 GiB。5 条均已通过基本数值质量检查。

### 6.2 动态转速轨迹

动态轨迹从已验证的无控制粗网格解 `t=80` 重启，只改变后圆柱转速：

| 数据划分 | 数量 | 动作设计 |
| --- | ---: | --- |
| train | 8 | 每 4 个时间单位换档的随机线性 ramp，档位为 `-1,-0.5,0,0.5,1` |
| train | 4 | 不同幅值、周期和相位的 multisine |
| validation | 1 | 训练档位中未出现的 `-0.75,-0.25,0.25,0.75` ramp |
| validation | 1 | 独立 multisine |
| test | 1 | 独立四分之一档位 ramp |
| test | 1 | 幅值和频率随时间变化的 chirp |

每条动态轨迹覆盖 `t=80..160`，有 16,000 个求解步、801 个场快照和 800 个相邻帧对。动作通过 OpenFOAM `Function1 table` 线性插值，范围不超过 `[-1,1]`。

首条 `dynamic_train_00` 已通过完整只读检查：

| 检查项 | 结果 |
| --- | ---: |
| 求解步数 | 16,000 |
| `U/p` 场快照 | 801 |
| 探针时刻 | 16,000 |
| 前圆柱力记录 | 16,000 |
| 后圆柱力记录 | 16,000 |
| 最大 Courant 数 | 0.245889861 |
| 最大逐步全局连续性误差绝对值 | `1.17497028e-12` |
| 求解器结尾 | `End` |

16 条轨迹已经全部完成。正式校验结果为：

| 汇总项 | 结果 |
| --- | ---: |
| train / validation / test | 12 / 2 / 2 条 |
| 总求解步数 | 256,000 |
| 总场快照数 | 12,816 |
| 总探针时刻数 | 256,000 |
| 16 条轨迹最大 Courant 数 | 0.251260868 |
| 16 条轨迹最大逐步全局连续性误差绝对值 | `1.93218635e-12` |
| 基本数值 QC | 16 / 16 通过 |
| 原始动态数据占用 | 约 32 GiB |

正式清单位于 `artifacts/tandem_cylinders/dynamic_dataset_manifest.json`。

### 6.3 最终划分和样本量

5 条恒定转速轨迹和 16 条动态轨迹都统一截取 `t=80..160`：

| 划分 | 轨迹数 | 每轨迹帧数 | 每轨迹相邻帧对 | 一步样本数 |
| --- | ---: | ---: | ---: | ---: |
| train | 15 | 801 | 800 | 12,000 |
| validation | 3 | 801 | 800 | 2,400 |
| test | 3 | 801 | 800 | 2,400 |
| 合计 | 21 | 16,821 | 16,800 | 16,800 |

数据按完整轨迹划分，不把同一轨迹的相邻帧随机分到不同集合。这样可以用验证集和测试集检查未见动作序列，而不是只检查已见轨迹的时间插值。

## 7 这些数据能支持什么

通过全部 CFD 门控后，这套数据足以用于第一阶段目标：

- 验证 OpenFOAM → VTK → Curator → HDF5 → PhysicsNeMo Datapipe 的完整链路；
- 训练单一 `Re`、单一几何下的动作条件短时流场代理模型；
- 预测下一时刻 `u/v/p` 场增量和后圆柱 `Cd/Cl`；
- 在未见动作轨迹上评估 1、10、50 步滚动预测，并与 persistence 基线比较。

当前数据不能支持以下结论：

- 跨雷诺数、跨圆心距或跨几何泛化；
- 三维湍流和真实海洋环境预测；
- 电机功耗、结构疲劳寿命或工程安全认证；
- 仅凭代理模型误差证明闭环控制有效；
- 已复现论文报告的约 98% 升力抑制。

闭环控制策略仍需回到独立 CFD 中回放，并与同一物理工况的无控制基线比较。

## 8 文件和脚本位置

| 文件或目录 | 作用 |
| --- | --- |
| `cases/<case>/` | 原始 OpenFOAM 配置、场、力、探针和日志 |
| `make_baselines.py` | 生成基础几何、网格和静止基线 |
| `make_small_control_dataset.py` | 生成 5 条恒定转速算例 |
| `make_dynamic_control_dataset.py` | 生成 16 条动态动作算例及元数据 |
| `make_rotation_grid_check.py` | 生成 `omega=+1` 中等网格算例 |
| `run_openfoam.sh` | 在固定镜像和资源限制下运行 OpenFOAM |
| `validate_small_dataset.py` | 校验 5 条恒定转速轨迹 |
| `validate_dynamic_dataset.py` | 校验 16 条动态轨迹并写 manifest |
| `compare_convergence.py` | 比较粗／中网格的力和探针统计 |
| `../../scripts/export_tandem_vtk.sh` | 把 `U/p` 导出为 Curator 可读 VTK |
| `../../conf/tandem_fno.yaml` | Hydra 模型、数据和训练配置 |
| `../../train_recipe.md` | Curator、Datapipe、FNO 训练和测试的完整操作手册 |
| `../../closed_loop_control_spec.md` | 第二阶段闭环目标、约束、MPC 和 CFD 回放验收规范 |
| `../../conf/tandem_mpc.yaml` | 第一版 surrogate-only MPC 配置 |
| `../../scripts/control_tandem_mpc.py` | 使用冻结 PhysicsNeMo FNO 的代理内 MPC 冒烟入口 |

## 9 查看当前计算和原始证据

进入目录：

```bash
ssh <SSH_HOST>
wsl.exe -d Ubuntu-24.04
cd ~/workspace/fluid_control/cfd/tandem_cylinders
```

查看真实求解进程：

```bash
ps -eo pid,lstart,etimes,%cpu,%mem,cmd \
  | grep pimpleFoam | grep -v grep
docker ps --format 'table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Command}}'
```

持续查看某条轨迹的原始日志：

```bash
tail -f cases/dynamic_train_00/log.pimpleFoam
```

检查已经完成的首条动态轨迹：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import json
from validate_dynamic_dataset import validate_case
print(json.dumps(validate_case("dynamic_train_00"), indent=2))
PY
```

全部动态轨迹结束后生成正式清单：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 validate_dynamic_dataset.py \
  --workers 4 \
  --write ../../artifacts/tandem_cylinders/dynamic_dataset_manifest.json
```

## 10 进入 PhysicsNeMo 前的最终门槛

- [x] 16 条动态轨迹全部正常结束并通过 `validate_dynamic_dataset.py`
- [x] `omega=+1` 中等网格算例正常结束
- [x] 旋转工况粗／中网格的主要受力统计差异不超过 3%
- [ ] 21 条轨迹的 `t=80..160` 均有 801 个场快照
- [x] train、validation、test 的轨迹清单固定为 15、3、3
- [x] 原始 CFD、校验 JSON、日志和动作配置完整保留

全部打勾后，再按 `train_recipe.md` 执行 VTK 导出和 PhysicsNeMo Curator。训练由项目负责人手动启动。

## 11 分步学习与执行命令

以下步骤按顺序执行。每次只执行当前一步，把完整终端输出发回，再根据结果决定是否继续。不要跳过失败的门槛。

### NVIDIA PhysicsNeMo 组件对应关系

| 链路阶段 | 必须实际调用的 NVIDIA 组件 | 本项目实现与证据 |
| --- | --- | --- |
| 数据整理 | Curator `VTKSource`、`Source → Filter → Sink`、`run_pipeline`；PhysicsNeMo `Mesh.sample_data_at_points` | `scripts/curate_tandem_cfd.py`；官方 API 负责 VTK 读取、网格表示、采样和流水线执行，项目适配器只处理轨迹/标签；输出 HDF5、manifest、train-only normalization |
| 数据读取 | `HDF5Reader`、`DatasetBase`、`TensorDict`、PhysicsNeMo `DataLoader` | `src/fluid_control/tandem_datapipe.py`；懒读取同一轨迹相邻帧，并由官方 DataLoader 预取和批处理 |
| 配置 | Hydra 和 OmegaConf | `conf/tandem_fno.yaml`；每次运行写出 `resolved_config.yaml` |
| 模型 | `physicsnemo.models.fno.FNO` | `scripts/train_tandem_fno.py` 中的 `build_model` |
| 设备与分布式 | `physicsnemo.distributed.DistributedManager` | 单 GPU Python 和多 GPU `torchrun` 共用训练入口 |
| 训练与验证 | `StaticCaptureTraining`、`StaticCaptureEvaluateNoGrad` | FP32 梯度更新、裁剪和无梯度验证；PhysicsNeMo 2.2.2 的 FNO metadata 不支持 GPU AMP |
| 日志 | `LaunchLogger`、`PythonLogger` | minibatch/epoch/validation 指标和 `physicsnemo.log` |
| 保存恢复 | `save_checkpoint`、`load_checkpoint` | `.mdlus` 模型及 `checkpoint.*.pt` 训练状态；支持续训 |
| 测试 | `DistributedManager`、`load_checkpoint`、同一 PhysicsNeMo FNO | 独立 test 轨迹的 1/10/50 步滚动误差和 persistence 基线 |

上述组件必须在代码、日志或产物中能直接核验。普通 PyTorch DataLoader、优化器和 DDP 作为 PhysicsNeMo 官方示例中的底层运行组件保留，但不能替代上述 PhysicsNeMo 接口。

### 第一步 等待旋转中等网格完成

**目的：**确认最后一个 CFD 验证算例仍在正常推进。

```bash
ssh <SSH_HOST>
wsl.exe -d Ubuntu-24.04
cd ~/workspace/fluid_control/cfd/tandem_cylinders

watch -n 3 '
printf "latest: "
grep "^Time = " cases/control_grid_p100_medium/log.pimpleFoam | tail -1
printf "process: "
pgrep -af "pimpleFoam.*control_grid_p100_medium" || true
printf "last diagnostics:\n"
tail -n 12 cases/control_grid_p100_medium/log.pimpleFoam
'
```

按 `Ctrl+C` 退出。正常现象是 `Time` 持续增加、进程存在、残差和连续性误差保持有限。到 `Time = 160` 后执行：

```bash
tail -n 20 cases/control_grid_p100_medium/log.pimpleFoam
grep -c '^Time = ' cases/control_grid_p100_medium/log.pimpleFoam
```

预期最后出现独立一行 `End`，求解步数为 32,000。把这两条命令的完整输出发回。

### 第二步 比较旋转工况粗网格和中等网格

只有第一步正常完成后再执行：

```bash
mkdir -p ../../artifacts/tandem_cylinders
python3 compare_convergence.py \
  control_small_p100 control_grid_p100_medium \
  | tee ../../artifacts/tandem_cylinders/rotation_grid_comparison.json
```

把完整 JSON 发回。重点检查前后圆柱的 `Cd_mean`、`Cl_rms`、`St` 相对差异。主要指标原则上都应不超过 0.03；接近零的量需要结合绝对值判断。通过后，CFD 数据才正式开放给 Curator。

### 第三步 导出一条 VTK 做格式冒烟测试

**执行结果：已通过。**`dynamic_train_00` 成功导出 801 个 `internal.vtu`，覆盖 `t=80..160`，每帧包含 `U` 和 `p`；目录约 2.2 GiB，`foamToVTK` 用时 36.36 秒、峰值内存约 154,724 kB，日志正常以 `End` 结束，未检出 `fatal` 或 `error`。

```bash
cd ~/workspace/fluid_control
bash scripts/export_tandem_vtk.sh dynamic_train_00

find cfd/tandem_cylinders/cases/dynamic_train_00/VTK_curator \
  -name internal.vtu -type f | wc -l
tail -n 20 \
  cfd/tandem_cylinders/cases/dynamic_train_00/log.foamToVTK_curator
```

预期得到 801 个 `internal.vtu`，转换日志没有 `FOAM FATAL ERROR`。把计数和日志末尾发回。

### 第四步 导出全部 21 条轨迹

**执行结果：已通过。**剩余 20 条轨迹全部成功导出；加上冒烟测试轨迹，共 21/21 条，每条均为 801 个 `internal.vtu`。自动检查退出码为 0，总 VTK 数据约 45 GiB。`vtk_counts.txt` 和批量转换日志保存在 `artifacts/tandem_cylinders/`。

第三步通过后，先创建清单。已经导出的 `dynamic_train_00` 不应重复执行：

```bash
cat > /tmp/tandem_cases_remaining.txt <<'EOF'
dynamic_train_01
dynamic_train_02
dynamic_train_03
dynamic_train_04
dynamic_train_05
dynamic_train_06
dynamic_train_07
dynamic_train_08
dynamic_train_09
dynamic_train_10
dynamic_train_11
dynamic_validation_00
dynamic_validation_01
dynamic_test_00
dynamic_test_01
control_small_m100
control_small_z000
control_small_p100
control_small_m050
control_small_p050
EOF

xargs -a /tmp/tandem_cases_remaining.txt -n1 -P2 \
  bash scripts/export_tandem_vtk.sh \
  |& tee artifacts/tandem_cylinders/vtk_export.log
```

检查所有轨迹：

```bash
for d in cfd/tandem_cylinders/cases/*/VTK_curator; do
  case_name=$(basename "$(dirname "$d")")
  count=$(find "$d" -name internal.vtu -type f | wc -l)
  printf '%-30s %s\n' "$case_name" "$count"
done | sort | tee artifacts/tandem_cylinders/vtk_counts.txt

awk '$2 != 801 {print "BAD", $0; bad=1} END {exit bad}' \
  artifacts/tandem_cylinders/vtk_counts.txt
```

预期列出 21 条轨迹，每条均为 801；最后一条 `awk` 命令不应输出 `BAD`。

### 第五步 安装独立 Curator 环境

Curator 当前是 beta 包，并包含 Rust 扩展。使用独立环境，不修改训练 `.venv`：

**环境预检结果：已通过。**远程机有 Git 2.43.0、uv 0.12.5；系统起初没有 Rust、Curator 源码或 `.venv-curator`。官方仓库 HEAD 为固定提交 `86533e581b3550326d89e97cb4d4126e7061b416`，项目盘剩余约 397 GiB。首次按包元数据使用 Python 3.11 时，安装成功但导入在 `class Source[T](ABC)` 处失败；该提交实际使用 Python 3.12 PEP 695 语法。保持 NVIDIA 源码原样，Curator 改用独立 Python 3.12 环境；训练 `.venv` 仍为 Python 3.11。实测 `/usr/bin/python3.12` 可用，版本为 Python 3.12.3。失败环境已保留为 `.venv-curator-py311-failed`（Python 3.11.16），新的 `.venv-curator` 已使用 Python 3.12.3 创建。采用项目内 `.tools/rustup`、`.tools/cargo` 和 `.venv-curator`，无需 sudo。

**基础安装结果：已通过。**源码远端为 NVIDIA 官方仓库，工作树干净，HEAD 为固定提交；项目内 Rust/Cargo 均为 1.98.1。Curator 0.1.0 基础包已成功安装到 Python 3.12.3 环境，安装退出码为 0；实测依赖包括 NumPy 2.5.3、h5py 3.16.0、PyVista 0.49.0 和 VTK 9.7.1。日志位于 `artifacts/tandem_cylinders/curator_install_py312.log`。使用官方 `VTKSource` 还需按仓库定义安装 `mesh` extra，并把其中的 PhysicsNeMo 固定为与训练环境一致的 2.2.2。

**导入验证：已通过。**官方 `Source`、`Filter`、`Sink` 与 `run_pipeline` 接口均成功加载，输出 `CURATOR_IMPORT_OK`，退出码为 0；日志位于 `artifacts/tandem_cylinders/curator_import_py312.log`。

```bash
cd ~/workspace/fluid_control
mkdir -p .tools
export RUSTUP_HOME="$PWD/.tools/rustup"
export CARGO_HOME="$PWD/.tools/cargo"

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --no-modify-path --profile minimal
export PATH="$CARGO_HOME/bin:$PATH"

uv venv .venv-curator --python 3.12
git clone https://github.com/NVIDIA/physicsnemo-curator.git \
  .tools/physicsnemo-curator
git -C .tools/physicsnemo-curator checkout \
  86533e581b3550326d89e97cb4d4126e7061b416

uv pip install --python .venv-curator/bin/python \
  --no-sources-package nvidia-physicsnemo \
  "./.tools/physicsnemo-curator[mesh]" \
  "nvidia-physicsnemo==2.2.2" h5py numpy
```

第一次补装 `mesh` extra 时，Curator 的 `[tool.uv.sources]` 把 `nvidia-physicsnemo` 指向 GitHub HEAD，与固定的 2.2.2 冲突，解析以退出码 1结束且没有安装包。训练环境已确认使用软件包索引发布的 `nvidia-physicsnemo 2.2.2`。加入 `--no-sources-package nvidia-physicsnemo` 后只取消这一项 Git 源覆盖，不修改官方源码；失败日志保留在 `artifacts/tandem_cylinders/curator_mesh_extra_install.log`。

使用 `--no-sources` 后，解析进入官方 `mesh` extra 的传递依赖，但其中仍声明 `sklearn==0.0.post12`；该弃用占位包默认拒绝构建，退出码为 1。按包自身错误提示，仅在安装期间设置 `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True`，不修改任何源码。该次日志保存在 `artifacts/tandem_cylinders/curator_mesh_extra_install_retry2.log`。

**Mesh extra 安装结果：已通过。**第三次安装解析 134 个包并安装 97 个包，退出码为 0。核心版本为 `nvidia-physicsnemo 2.2.2`、Torch 2.14.0、TensorDict 0.14.2、PyArrow 25.0.1 和 Warp 1.17.0；日志位于 `artifacts/tandem_cylinders/curator_mesh_extra_install_retry3.log`。

验证：

```bash
git -C .tools/physicsnemo-curator rev-parse HEAD
.venv-curator/bin/python - <<'PY'
import h5py, numpy, pyvista
import physicsnemo_curator
from physicsnemo.mesh import Mesh
from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource
from physicsnemo_curator.run import run_pipeline
print("curator", physicsnemo_curator.__file__)
print("h5py", h5py.__version__)
print("numpy", numpy.__version__)
print("pyvista", pyvista.__version__)
print("Mesh/VTKSource", Mesh, VTKSource)
PY
```

### 第六步 Curator 单轨迹冒烟测试

本步骤使用的主链为官方 Curator `VTKSource` → 官方 PhysicsNeMo `Mesh.sample_data_at_points` → Curator `Filter/Sink/run_pipeline`。项目适配代码仅聚合 801 个时间帧，并加入动作、力和数据划分。实测原始 `internal.vtu` 同时含 point/cell `U/p` 与 `TimeValue`；规则网格使用 point data 插值。

**官方单帧 API 验证：已通过。**`VTKSource` 发现 801 帧；首帧官方 Mesh 为 39,336 个点、115,740 个四面体单元，完整包含 point/cell `p/U` 和 global `TimeValue=80.0`。官方 `Mesh.sample_data_at_points` 在 `64×32` 网格上得到 2,024/2,048 个有效点，覆盖率 0.98828125；输出 `OFFICIAL_VTK_MESH_API_OK`，退出码为 0。日志位于 `artifacts/tandem_cylinders/curator_official_vtk_api_check.log`。

```bash
rm -rf data/curated/tandem_smoke
.venv-curator/bin/python scripts/curate_tandem_cfd.py \
  --cases-root cfd/tandem_cylinders/cases \
  --output data/curated/tandem_smoke \
  --nx 256 --ny 128 --limit 1 \
  |& tee artifacts/tandem_cylinders/curator_smoke.log
```

**单轨迹执行：已通过。**官方 API 链路完成 `dynamic_test_00` 的 801/801 帧并写出 1 条轨迹，退出状态为 0；墙钟时间 4 分 21.06 秒，CPU 利用率 1919%，峰值常驻内存约 2.22 GiB，swap 为 0。Curator 以整条轨迹为工作项，因此处理期间显示 `0/1`，Sink 完成后显示 `1/1 (100%)`；逐帧日志证明内部从 1 推进到 801。日志位于 `artifacts/tandem_cylinders/curator_smoke.log`。

检查 HDF5：

```bash
.venv-curator/bin/python - <<'PY'
from pathlib import Path
import h5py
p = next(Path("data/curated/tandem_smoke").rglob("*.h5"))
with h5py.File(p, "r") as f:
    print("file", p)
    for key in f:
        print(key, f[key].shape, f[key].dtype)
PY
```

**HDF5 质量检查：已通过。**形状、dtype、chunk/compression 和 80.0–160.0 严格递增时间轴均正确。mask 仅含 0/1，有效覆盖率 0.9869384765625；动作范围 `[-0.75,0.75]`，力数据全部有限；逐帧最大有效区域压力均值绝对值为 `1.6456821227265347e-09`，无效区域状态最大绝对值为 0。检查输出 `SMOKE_HDF5_OK`，退出码为 0；日志位于 `artifacts/tandem_cylinders/curator_smoke_hdf5_check.log`。

预期 `state=(801,3,128,256)`、`mask=(801,1,128,256)`、`omega=(801,1)`、`force=(801,4)`、`time=(801,1)`。

### 第七步 生成完整 Curator HDF5 数据集

```bash
rm -rf data/curated/tandem_cylinders
.venv-curator/bin/python scripts/curate_tandem_cfd.py \
  --cases-root cfd/tandem_cylinders/cases \
  --output data/curated/tandem_cylinders \
  --nx 256 --ny 128 \
  |& tee artifacts/tandem_cylinders/curator_full.log

.venv-curator/bin/python -m json.tool \
  data/curated/tandem_cylinders/manifest.json
.venv-curator/bin/python -m json.tool \
  data/curated/tandem_cylinders/normalization.json
```

预期轨迹计数为 train 15、validation 3、test 3。归一化统计必须标明只来自 train。

**完整执行与独立审计：已通过。**Curator 21/21 完成，退出码 0，墙钟时间 1:18:48，峰值常驻内存约 2.50 GiB，swap 为 0；生成 21 个 HDF5、共 5.5 GiB，划分 15/3/3。`scripts/validate_tandem_curated.py` 独立逐文件审计并重新计算训练集统计，输出 `CURATED_DATASET_OK`，退出码 0。证据位于 `artifacts/tandem_cylinders/curator_full.log`、`curated_validation.log` 和 `curated_validation.json`。

### 第八步 验证 PhysicsNeMo Datapipe

```bash
uv pip install --python .venv/bin/python -e .

.venv/bin/python - <<'PY'
from fluid_control.tandem_datapipe import TandemWindowDataset
for split, expected in (("train", 12000), ("validation", 2400), ("test", 2400)):
    ds = TandemWindowDataset("data/curated/tandem_cylinders", split)
    sample, metadata = ds[0]
    print(split, len(ds), {k: tuple(v.shape) for k, v in sample.items()})
    print("metadata", metadata)
    assert len(ds) == expected
    assert sample["x"].shape == (6, 128, 256)
    assert sample["delta"].shape == (3, 128, 256)
    assert sample["force"].shape == (2,)
    assert all(v.isfinite().all() for v in sample.values())
    ds.close()
PY
```

**DataPipe 验证：已通过。**官方 `DatasetBase/HDF5Reader/DataLoader` 得到 12,000/2,400/2,400 个窗口；连续三个 batch 的所有张量形状正确且数值有限。输出 `PHYSICSNEMO_DATAPIPE_OK`，退出码 0；日志位于 `artifacts/tandem_cylinders/datapipe_validation.log`。

同时核对完整 PhysicsNeMo 运行组件和 Hydra 配置：

```bash
.venv/bin/python - <<'PY'
from omegaconf import OmegaConf
from physicsnemo.datapipes import DataLoader, DatasetBase, HDF5Reader
from physicsnemo.distributed import DistributedManager
from physicsnemo.models.fno import FNO
from physicsnemo.utils import (
    StaticCaptureEvaluateNoGrad,
    StaticCaptureTraining,
    load_checkpoint,
    save_checkpoint,
)
from physicsnemo.utils.logging import LaunchLogger, PythonLogger

cfg = OmegaConf.load("conf/tandem_fno.yaml")
print(OmegaConf.to_yaml(cfg))
print("PhysicsNeMo Datapipes and runtime imports: OK")
PY
```

### 第九步 运行 FNO 训练冒烟测试

```bash
bash scripts/run_tandem_fno_smoke.sh
```

该脚本使用 GPU 0/1、每进程显存比例上限 0.75，并每 5 秒写出实际 GPU 指标。首次通过断开式 SSH 后台启动时，`torchrun` 在有效训练前收到 SIGHUP并退出1；GPU无残留占用，问题属于启动方式。失败目录归档后在用户 tmux 前台重跑。

用户前台重跑后，两个 rank 在 PhysicsNeMo `DistributedManager.initialize()` 创建 NCCL process group 时均遇到 NCCL 2.30.7 `Cuda failure 999`，仍未进入模型训练；退出码为 1。后续清理阶段的 `Process group cannot be None` 是次生异常。下一步通过版本化脚本检查两张卡的独立 CUDA 运算与 peer access，再选择 NCCL transport 设置。

**GPU 诊断：单卡 CUDA 正常，P2P 不可用。**驱动 596.72、PyTorch 2.14.0+cu130、CUDA 13.0、NCCL 2.30.7；GPU 0 和 1 分别通过矩阵乘法，但 WSL 无法给出拓扑矩阵，0→1 和 1→0 的 peer access 均为 false。诊断输出 `CUDA_DEVICE_CHECK_OK`、退出码 0；日志为 `artifacts/tandem_cylinders/gpu_diagnostic.log`。下一步禁用 NCCL P2P 后仅测试最小 all-reduce。

**NCCL 传输修复：已通过。**仅禁用 P2P 仍失败；禁用 P2P、SHM、IB 和 cuMem 后，两个 rank 先通过内置 `NET/Socket` 完成 all-reduce，证明通信可回退。随后逐项恢复 SHM：保持 P2P、IB 和 cuMem device/host allocation 关闭，把 `NCCL_SHM_DISABLE` 设为 `0`。两个 rank 成功初始化，4 个通信通道均记录为 `via SHM/direct`，all-reduce 结果均为 3.0，输出 `NCCL_ALL_REDUCE_OK`，退出码 0。冒烟和正式训练脚本现固定使用 SHM；Socket 仅作为已验证的故障回退。日志位于 `artifacts/tandem_cylinders/nccl_shm_check.log`、`nccl_socket_check.log` 和 `nccl_socket.*.log`。

**双卡训练冒烟：已通过。**修复后的 PhysicsNeMo FNO 完成 1 个 epoch，训练 loss `0.2025793`，验证场 MAE `0.00336758`、RMSE `0.00516719`，归一化力系数 MAE `0.746278`，退出码 0。`.mdlus`、训练状态、best checkpoint、配置和 history 全部生成并通过独立产物检查。checkpoint 已改为仅 rank 0 保存，消除了双 rank 同名文件并发写入。GPU 0/1 最大显存占用为 1,726/1,858 MiB，最低剩余 70,714/70,582 MiB。

单卡对照为 `69.71 ms/iter`、约 `114.8 samples/s`；双卡 SHM 为 `88.90 ms/iter`，但全局 batch 翻倍后约 `180.0 samples/s`，吞吐提升约 1.57 倍，并行效率约 78.4%。因此正式训练采用 GPU 0/1 与 SHM 配置。该数据不能证明与原生 Linux P2P 性能一致，只证明当前 WSL 修复配置对本任务比单卡更快。

预期完成 1 个 epoch、所有 loss 和 metric 为有限数，并生成：

- `resolved_config.yaml`：Hydra 完整解析配置；
- `physicsnemo.log`：PhysicsNeMo PythonLogger 文件日志；
- `checkpoints/*.mdlus`：PhysicsNeMo 模型；
- `checkpoints/checkpoint.*.pt`：优化器、scheduler、epoch、metadata 和 capture 状态；
- `best/`：当前最佳验证 checkpoint；
- `training_history.json`：便于审阅的指标副本。

检查命令：

```bash
find artifacts/tandem_fno_smoke -maxdepth 2 -type f -printf '%P\n' | sort
python3 -m json.tool artifacts/tandem_fno_smoke/training_history.json
```

冒烟测试已经通过并确定使用双卡、每 rank batch size 8；正式 epochs 保持配置中的 50。完整执行命令与恢复方法见根目录 `train_recipe.md`。

### 第十步 正式训练和测试

正式训练使用 GPU 0/1、SHM 通信和 50 epochs。初始阶段采用每 rank batch size 8；根据项目负责人对训练速度和最终精度的要求，后续从 checkpoint 恢复时采用每 rank batch size 64、全局 batch size 128，并将每进程显存比例上限设为 0.85，以理论保留约 10.8 GiB/卡。恢复时保留 optimizer 和 scheduler 状态。`torchrun` 双 GPU命令、checkpoint 恢复及独立测试命令已记录在根目录 `train_recipe.md`。

**正式训练已经完成。**50/50 epochs 正常结束，退出码 0。Epoch 50 为验证场 MAE 最佳 checkpoint：场 MAE `5.3888804e-04`、场 RMSE `7.9748279e-04`、归一化后柱力 MAE `8.5806989e-03`；后柱力 MAE 的单项最低值为 Epoch 42 的 `8.4354119e-03`。下一门槛是使用冻结的 Epoch 50 PhysicsNeMo checkpoint 对 3 条独立 test 轨迹执行 FP32 的 1/10/50 步 rollout，同时评估流场、后柱 `Cd/Cl`、persistence 基线和数值稳定性。每条轨迹和每个 horizon 生成三个代表起点的 `u/v/p` Ground Truth、Prediction、Absolute Error 对比图，预计共 27 张 PNG。

**独立 test rollout 已通过。**评估退出码 0，1/10/50 步分别覆盖 2,400/240/48 个片段，全部 `stable=true` 且无失败片段。场 MAE 分别为 0.0006102、0.0047165、0.0160685，相对 persistence 降低 93.03%、94.30%、84.52%；后柱力 MAE 分别为 0.0052400、0.0131261、0.0489211，相对 persistence 降低 92.11%、97.90%、93.97%。27/27 张对比图生成成功。恒定转速轨迹的 50 步预测最稳；动态轨迹保留主要尾流结构，但高频误差随 rollout 累积，最难的 `dynamic_test_01` 在 50 步的场/后柱力 MAE 为 0.0218156/0.0760545。第一阶段动作条件流场代理模型因此通过，但该结果不等于闭环控制验证。
