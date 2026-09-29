# PhysicsNeMo 串列双圆柱流场预测与闭环控制

更新日期：2026-09-29。本仓库包含从 OpenFOAM 数据生成、NVIDIA PhysicsNeMo Curator、PhysicsNeMo Datapipe、FNO 训练、独立 test rollout 到第二阶段闭环控制设计的完整代码和操作记录。

第一阶段已经完成：5 条恒定转速和 16 条动态转速 CFD 轨迹通过质量检查，21/21 条轨迹完成 VTK 与 Curator HDF5 转换，PhysicsNeMo FNO 完成 50 epochs 训练。Epoch 50 在 3 条独立 test 轨迹上稳定完成 1/10/50 步 rollout；50 步场 MAE 为 `0.01607`，相对 persistence 降低 `84.52%`，并生成 27 张 Ground Truth/Prediction/Error 对比图。定量过程见 [`train_recipe.md`](train_recipe.md)，物理和数据边界见 [`CASE_SPEC.md`](cfd/tandem_cylinders/CASE_SPEC.md)。

第二阶段目标是在同一 `Re=100` 工况上建立在线闭环原型，优先抑制后圆柱升力波动，同时监控阻力和动作代价。目标函数、动作约束、MPC 范围和 OpenFOAM 回放验收规则见 [`closed_loop_control_spec.md`](closed_loop_control_spec.md)。

> 仓库保存可复现代码、配置和文档。原始 OpenFOAM 时间目录、45 GB VTK、5.5 GB HDF5、模型 checkpoint、运行日志和生成图片由 `.gitignore` 排除，需要按操作手册在计算节点生成。

## 1. 项目介绍与问题分析

### 1.1 选定的论文与研究边界

第一篇复现对象是港理工唐辉团队参与的 Zhao 等人论文：[《Mitigating the lift of a circular cylinder in wake flow using deep reinforcement learning guided self-rotation》](https://doi.org/10.1016/j.oceaneng.2024.118138)，发表于 *Ocean Engineering*（2024）。港理工公开了[接收稿全文](https://ira.lib.polyu.edu.hk/bitstream/10397/107766/1/Zhao_Mitigating_Lift_Circular.pdf)。它使用 GPU 加速 CFD 和深度强化学习，研究**两根等直径圆柱串列排列时，通过后圆柱自转降低后圆柱的升力波动**。

这是与港理工工作直接相关、物理条件和评价量相对明确的第一阶段数值案例。它**不是**此前讨论的“一个主圆柱加两个较小后置控制圆柱”；后者与唐辉指导的[叶伟健博士论文](https://theses.lib.polyu.edu.hk/handle/200/14670)更接近，主要研究尾流均匀性与水动力特征抑制，应作为后续独立课题。两种几何、优化目标和数据不得混用。本项目也不把非港理工团队的 Fan 等人实验称作港理工成果。

### 1.2 场景：谁受到不利作用，谁在转动

流体从左向右依次经过前、后两根固定中心位置的圆柱。前圆柱分离出的剪切层和涡向下游输运，可能直接作用在后圆柱上；后圆柱又形成自己的剪切层和尾流。论文关注的不是“前圆柱的阻力是否下降”，而是**处在前圆柱尾流中的后圆柱所受横向力为何大幅波动，以及能否通过控制其自转减弱该波动**。圆柱本体不作平移或振动；执行器是后圆柱绕轴线的随时间变化的自转。[论文问题与装置](https://ira.lib.polyu.edu.hk/bitstream/10397/107766/1/Zhao_Mitigating_Lift_Circular.pdf)

```text
                     均匀来流 U∞ →

          前圆柱（固定、不转）             后圆柱（固定中心、可自转）
                 ○   ─── 前柱尾流/涡 ───>       ↻ ○   ───> 下游尾流与速度传感器
                 <────────── 圆心距 L ──────────>

                     主要观测：后圆柱横向力 Fᵧ,rear 的波动
                     控制动作：后圆柱角速度 ω(t)
```

图为概念示意，不是论文流场图或计算结果。来流、圆柱直径 `D`、运动黏度 `ν` 定义雷诺数 `Re=U∞D/ν`；圆心距用 `L/D` 表示。论文用 `Re=100` 的二维不可压缩流，重点展示 `L/D=5`：此时前柱脱落的涡与后柱相互作用较强，后柱升力波动显著。不能把这一低雷诺数二维结论直接外推到真实海洋湍流、三维管束或所有间距。[论文第 2–3 节](https://ira.lib.polyu.edu.hk/bitstream/10397/107766/1/Zhao_Mitigating_Lift_Circular.pdf)

### 1.3 为什么要控制升力，而不是只看阻力

阻力沿来流方向，升力横向于来流。即使长期平均升力接近零，交替脱涡仍可导致很大的**瞬时升力和升力脉动**；这种周期性载荷可能使下游圆柱状结构承受不利的振动与疲劳风险。串列构件、换热管束、海上管线和缆索是相关的工程背景，但论文没有对这些具体设施做寿命或安全认证。论文报告：未控制时，某些圆心距下后圆柱的平均阻力低于孤立单圆柱，却可能具有更高的升力波动。因此，“减小阻力”和“抑制后柱升力波动”是两个不同问题，本案例以后者为主。[论文引言、图 3 与结论](https://ira.lib.polyu.edu.hk/bitstream/10397/107766/1/Zhao_Mitigating_Lift_Circular.pdf)

我们必须区分以下物理对照：

| 状态 | 用途 | 不能据此推出什么 |
| --- | --- | --- |
| 单根孤立静止圆柱 | 检查 CFD 求解器是否能复现基本绕流、受力与脱涡频率 | 不代表串列双圆柱的无控制基线。 |
| 两根圆柱串列、均不旋转 | **本论文的控制基线**：测后圆柱升力时间序列及波动 | 与孤立单圆柱的差异包含“增加第二根圆柱”的几何效应。 |
| 同样的串列几何、后圆柱按 `ω(t)` 自转 | 隔离旋转控制对后柱升力和尾流的影响 | 升力改善不自动表示总阻力或净能耗改善。 |

在二维计算中，后圆柱升力系数按单位展向长度受力定义为 `CL,rear = Fy,rear / (0.5ρU∞²D)`。主评价量应包括其时间序列、平均绝对值和脉动量（例如标准差或 RMS），并记录尾流结构和转速时序。比较必须使用相同的几何、`Re`、统计窗口和受力定义。论文使用的代价函数主要惩罚后圆柱升力绝对值，对转速平方加入较小权重；**这一惩罚项不等同于已测得的电机实际功率**。[论文第 2.3 节](https://ira.lib.polyu.edu.hk/bitstream/10397/107766/1/Zhao_Mitigating_Lift_Circular.pdf)

### 1.4 论文如何控制，以及我们要回答的问题

论文用后圆柱下游一条竖线上的 32 个速度传感器，观测流向、横向速度；PPO 智能体据此给出后圆柱的角速度动作。作者使用自研 GPU 加速的二维 D2Q9 格子玻尔兹曼求解器（LBM）作为 CFD 环境，而不是以照片、示意图或随机数生成流场。控制使后柱附近剪切层与前柱尾流的相互作用发生变化。论文在重点工况报告约 98% 的后柱升力波动抑制；这是**作者的结果，不是本项目已经复现的成绩**。[论文方法与结果](https://ira.lib.polyu.edu.hk/bitstream/10397/107766/1/Zhao_Mitigating_Lift_Circular.pdf)

本项目要逐层回答，而非直接把神经网络训练成功当作物理控制成功：

1. **物理层**：我们的 CFD 在“不转”状态能否产生与论文相符的后柱升力波动、主要脱涡频率和尾流形态？在预设自转下，流场与后柱受力是否呈合理响应？
2. **数据层**：能否从经过验证的 CFD 中保存连续、可追溯的速度场、两根圆柱各自的受力和后柱动作，而不是只有渲染图片或孤立样本？
3. **代理模型层**：PhysicsNeMo 模型能否从当前流场或稀疏传感器及动作，预测后续流场与后柱受力？它对未见过的动作序列是否优于简单基线，滚动预测是否稳定？
4. **控制层（后续阶段）**：仅在代理模型与 CFD 验证合格后，再研究优化转速或接入 HydroGym；是否复现 PPO 论文结果、是否引入额外控制框架，到那时另行确认。

阶段完成的标准是**可对照、可重复的物理量**，不是只出现一张类似论文的涡量图。若采用与原文不同的求解器，我们只能称为“独立物理复现”，不能称为作者代码复跑；论文的自研 LBM、网格处理和控制实现差异应记录。当前不预设能达到论文的 98%，更不把“后柱升力下降”改写成“前柱减阻”。

### 1.5 第一阶段的具体边界

第一阶段固定为论文重点工况 `Re=100、L/D=5`。计算域为 `30D × 15D`，前圆柱中心位于 `(10D,7.5D)`，后圆柱中心位于 `(15D,7.5D)`；入口为均匀来流，出口定运动学压力，上下自由滑移，前圆柱固定，后圆柱自转。当前目标是训练可感知动作的短时流场与后柱受力代理模型，随后用未见动作轨迹评估一步和滚动预测。闭环控制器、PPO 复现和三维工程外推不属于本阶段完成条件。

## 2. CFD 数据及当前状态

### 2.1 工具与数值设置

CFD 使用 OpenCFD OpenFOAM v2512 的 `pimpleFoam`，容器镜像固定为 `opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae6698c489293ba30c991380fe3f899c622f319`。容器运行时禁网、只读根文件系统、非 root 用户、每任务最多 4 CPU 和 8 GiB 内存。求解采用二维层流不可压缩 Navier–Stokes、二阶 `backward` 时间格式、`Δt=0.005D/U∞`；`D=U∞=ρ=1`，`ν=0.01`。

训练轨迹使用粗网格 19,290 个单元、每个圆柱 96 个壁面面元。中等网格有 77,160 个单元、每柱 192 个壁面面元。两者 `checkMesh` 均为 `Mesh OK`。无控制情况下，统计窗 `t=80..160` 的粗／中网格差异为：前柱 `Cd_mean 0.4635%`、`Cl_rms 1.5567%`、`St 0.8055%`；后柱分别为 `0.0476%`、`0.5012%`、`0.8001%`。`omega=+1` 旋转工况对应差异为：前柱 `0.4982%/2.4643%/0.7945%`，后柱 `0.1560%/1.0616%/0.7928%`。无控制和旋转工况的主要受力指标均通过预设 3% 门槛。

### 2.2 已有和正在生成的数据

5 条恒定转速轨迹已经完成：`omega=-1,-0.5,0,+0.5,+1`。每条从 `t=0` 到 `160` 共 32,000 个求解步，保存 1,600 个间隔为 `0.1` 的 `U/p` 场快照、每步前后圆柱 `Cd/Cl` 和每步 32 个尾流探针的 `u/v`；单条约 4 GiB。只读校验器已确认五条均正常结束、步数和字段齐全、未发现文本形式的 NaN/Inf。

为使动作条件模型看到瞬态响应，补充了 16 条 `t=80..160` 的动态动作轨迹：12 train、2 validation、2 test。每条有 16,000 个求解步、801 个场快照和 800 个相邻帧对。动作包含随机分段线性变化、多正弦、四分之一档位和 chirp，范围统一为 `[-1,1]`。16 条已全部完成并通过只读检查：总计 256,000 个求解步、12,816 个 `U/p` 场快照和 256,000 个探针时刻；最大 Courant 数 `0.251260868`，逐步全局连续性误差最大绝对值 `1.93218635e-12`，原始动态数据约 32 GiB。

### 2.3 第一阶段最终划分

| 划分 | 动态轨迹 | 恒定动作轨迹 | 轨迹总数 | 一步样本数 |
| --- | ---: | ---: | ---: | ---: |
| train | 12 | 3 | 15 | 12,000 |
| validation | 2 | 1 | 3 | 2,400 |
| test | 2 | 1 | 3 | 2,400 |

所有轨迹只进入一个划分，避免相邻帧泄漏。训练使用 `t=80..160`，5 条恒定轨迹也只截取同一时间窗。原始字段为 `U、p、mask、omega、front/rear Cd/Cl、time`。Curator 将场采样到 `x=[8,25]、y=[4,11]` 的 `256×128` 规则网格，每帧去掉有效流体区压力均值，并且只用 train 划分计算归一化统计。

这套数据足以启动第一阶段的 Curator、Datapipe、FNO 训练与未见动作轨迹评估。它仍只覆盖单一 `Re`、单一间距、二维层流和有限动作幅值，因此不能支持跨雷诺数、跨几何或真实工程闭环结论，也不能单凭训练损失宣称已复现论文约 98% 的控制效果。

旧的 Zenodo 三圆柱 fluidic pinball 数据和项目内谱方法开发数据属于不同几何与指标，不进入本串列双圆柱数据集。其历史仍保留在[公开 PIV 记录](docs/PUBLIC_PIV_CASE.md)、[旧 CFD 数据生成记录](docs/CFD_DATA_GENERATION.md)和[历史结果](docs/第一阶段结果.md)。

## 3. PhysicsNeMo 完整链路

### 3.1 Curator

使用 NVIDIA 官方 PhysicsNeMo Curator 提交 `86533e581b3550326d89e97cb4d4126e7061b416`。项目脚本以官方 `VTKSource` 读取 OpenFOAM 导出的 VTK，并用官方 `physicsnemo.mesh.Mesh.sample_data_at_points` 采样规则网格；项目适配代码只负责轨迹分组、动作/力时间对齐和物理质量检查。随后通过官方 `run_pipeline` 执行 `Source → Filter → Sink`，输出每轨迹一个压缩 HDF5、`manifest.json` 和 train-only `normalization.json`。Curator 当前为 beta，使用独立 `.venv-curator` 环境，避免改变已存在的 PhysicsNeMo 训练环境。

### 3.2 Datapipe

`src/fluid_control/tandem_datapipe.py` 继承官方 `physicsnemo.datapipes.DatasetBase`，使用 `HDF5Reader` 懒读取同一轨迹内相邻帧，返回 `TensorDict`，并交给官方 PhysicsNeMo `DataLoader` 做采样、批处理和预取。输入 6 通道为标准化 `u/v/压力`、有效掩码、当前动作和下一动作；标签为三通道场增量及下一时刻后圆柱 `Cd/Cl`。配置使用 Hydra/OmegaConf，并在每次运行时保存解析后的完整配置。

### 3.3 模型和评价

`scripts/train_tandem_fno.py` 使用 PhysicsNeMo 二维 FNO：6 输入通道、5 输出通道、4 个 Fourier 层、32 个隐通道、`24×24` 模态。训练运行时实际调用 PhysicsNeMo `DistributedManager`、`StaticCaptureTraining`、`StaticCaptureEvaluateNoGrad`、`LaunchLogger`、`PythonLogger`、`save_checkpoint` 和 `load_checkpoint`；支持单 GPU及 `torchrun` 多 GPU。优化器为 AdamW，训练 50 epochs、初始学习率 `2e-4`，用 validation 场 MAE 选择 PhysicsNeMo `.mdlus` checkpoint。PhysicsNeMo 2.2.2 的 FNO metadata 不支持 GPU AMP，因此本项目显式使用 FP32。

`scripts/evaluate_tandem_fno.py` 通过 PhysicsNeMo checkpoint 接口加载模型，在完整 test 轨迹上计算 1、10、50 步滚动流场和后圆柱 `Cd/Cl` 误差，与 persistence 基线比较，并生成 `u/v/p` 的 Ground Truth、Prediction 和 Absolute Error 图。`scripts/control_tandem_mpc.py` 使用冻结 FNO 做 surrogate-only 的可微 MPC 冒烟；该结果必须再由独立 OpenFOAM 闭环回放验证。

## 4. 操作入口

完整安装、CFD 校验、VTK、Curator、Datapipe、训练、恢复、独立 test 和可视化命令见 [`train_recipe.md`](train_recipe.md)。核心顺序为：

```text
OpenFOAM cases
  -> foamToVTK
  -> PhysicsNeMo Curator VTKSource / Mesh / Source-Filter-Sink
  -> HDF5 + train-only normalization
  -> PhysicsNeMo HDF5Reader / DatasetBase / DataLoader
  -> PhysicsNeMo FNO / DistributedManager / StaticCapture
  -> checkpoint + independent test rollout
  -> surrogate MPC smoke
  -> independent OpenFOAM closed-loop replay
```

主要入口：

| 任务 | 文件 |
| --- | --- |
| CFD 基线和网格生成 | `cfd/tandem_cylinders/make_baselines.py` |
| 恒定转速数据生成 | `cfd/tandem_cylinders/make_small_control_dataset.py` |
| 动态转速数据生成 | `cfd/tandem_cylinders/make_dynamic_control_dataset.py` |
| OpenFOAM 固定容器执行 | `cfd/tandem_cylinders/run_openfoam.sh` |
| VTK 导出 | `scripts/export_tandem_vtk.sh` |
| Curator 数据整理 | `scripts/curate_tandem_cfd.py` |
| PhysicsNeMo Datapipe | `src/fluid_control/tandem_datapipe.py` |
| FNO 训练 | `scripts/train_tandem_fno.py` |
| 独立 test 与可视化 | `scripts/evaluate_tandem_fno.py` |
| 第二阶段 MPC 冒烟 | `scripts/control_tandem_mpc.py` |

## 5. 仓库目录

```text
cfd/tandem_cylinders/   OpenFOAM 工况生成、执行和质量检查
conf/                   FNO 与 MPC 配置
scripts/                Curator、训练、评估和控制入口
src/fluid_control/      PhysicsNeMo Datapipe 与公共模块
docs/                   数据来源和历史实验记录
train_recipe.md         第一阶段逐步操作与实测结果
closed_loop_control_spec.md  第二阶段闭环控制规范
```

安装项目代码：

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .
```

PhysicsNeMo Curator 使用独立 Python 3.12 环境，固定提交和安装方法记录在 `train_recipe.md`，避免与训练环境混装。
