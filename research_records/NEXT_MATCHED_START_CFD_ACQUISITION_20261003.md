# 下一轮 matched-start OpenFOAM 数据采集设计

日期：2026-10-03
状态：**预先设计；尚未启动求解、整理或训练**

## 1. 结论先行

下一步不应先扩大 FNO，也不应让当前代理模型做主动选点。当前 v3/v4 在独立低幅动作验证上的 H100 pooled 总阻力 NRMSE 分别为
`16.2544%` 和 `17.8589%`，都比 persistence 的绝对误差更差；严格同初态的正负动作比较又只有 `start=0` 一对。它们能说明“已有数据缺少可辨识的低幅动作因果覆盖”，不能可靠地告诉 OpenFOAM 下一条轨迹该采哪里。

推荐先建立一个规模受控、按**独立起始相位分组**的 matched-start 数据矩阵：8 个相位，每个相位从 byte-identical 的 `U/p` restart 同时分出 `omega={-0.75,-0.375,0,+0.375,+0.75}` 五条分支，共 40 条。先跑其中 3 个训练相位的 `{-0.75,0,+0.75}`，即 9 条 commissioning 分支；它只验证数据管线，不足以训练或作物理结论。完整 40 条才是具备 train/validation/frozen-test 隔离的最小实验。

这批数据服务于 matched-start 因果辨识和 surrogate validation，**不是**正在进行的 long-dwell 物理筛查的延续，也不得根据该筛查的好坏重新挑相位或动作。

## 2. 科学边界不变

物理问题仍以 [`RESEARCH_OBJECTIVE.md`](RESEARCH_OBJECTIVE.md) 为准：固定串列双圆柱、`Re=100`、`L/D=5`，仅后圆柱有界旋转；主要目标是相对同相位零动作降低两圆柱系统的时间平均总阻力，且不显著恶化升力。

Zhao、Zhou、Ren、Tang、Wang 的 2024 年港理工工作提供了最接近的物理动机：用 PPO 引导尾流中圆柱自旋，主要抑制升力波动。官方 PolyU 页面和出版元数据支持这个高层描述，但当前项目的来源核查没有得到可逐页公开核验的全文，因此这里**不**把动作幅值、更新频率、传感器位置或 reward 数值说成该论文的精确复现。本项目额外要求系统总阻力、后圆柱平均升力和动作代价，目标没有改变。

已有低幅验证的限制必须保留：

- v3/v4 严格 `start=0` 正负动作 H100 瞬时总阻力差值误差约为 `0.09295/0.11136`，虽然这一对的符号正确，但误差都大于 2% 阻力目标对应的约 `0.046 Cd` 尺度；
- all-start 的 `68.97%` 符号排序只是在相同 elapsed time 比较已经分叉的状态，不是反事实 matched-start 证据；
- H100 终点瞬时力不等于 `[114,174]` 一类长窗平均力，不能据此声称控制成功。

## 3. 预先声明的采集矩阵

### 3.1 相位定义和 split

先在不受控、已统计稳定的 baseline 后圆柱 `Cl` 上用同一固定算法估计主 shedding phase；测得周期约为 `6.154 D/U`。从已有 baseline 的完整 `U/p` restart 中，按最近相位挑选八个独立起点：

```text
phi_k = k*pi/4,  k=0,...,7
```

选择规则、允许的相位偏差、候选时间范围和最终 restart SHA256 必须在任何受控分支求解前写进 manifest。不得在看到控制结果后换相位。八个相位按组隔离：

| split | 相位 bin | 分支数 | 用途 |
|---|---:|---:|---|
| train | `0, pi/2, pi, 3pi/2` | `4 x 5 = 20` | 训练及训练内诊断 |
| validation | `pi/4, 5pi/4` | `2 x 5 = 10` | 模型/超参数选择 |
| frozen test | `3pi/4, 7pi/4` | `2 x 5 = 10` | 配置冻结后一次性评估 |

split 的单位是**起始相位/restart 组**，不是 frame 或滑动 segment。一个 restart 的所有五种动作只能属于同一 split；同一 baseline 周期的近重复状态也不能跨 split。Frozen-test HDF5 可以生成并校验 hash，但模型选择完成前，评估脚本不得加载其场或力数据。

这里的八个 phase bins 都来自同一个不受控基准极限环，**不等于八个统计独立的物理工况**。这种跨 phase split 主要检验未见 shedding phase 的插值/外推，不能代表 Reynolds 数、几何、入口扰动、测量噪声或长期非平稳性的鲁棒性。选 restart 时要在预先固定的多个候选 shedding cycles 上计算近重复度和力信号自相关，先锁定候选周期、绝对 restart 时间和排除规则，再生成任何控制标签；不得把时间相近或高度自相关的 near-duplicate 分到不同 split。报告中同时给出“8 个 phase groups”和实际有效独立性诊断，不能把 frame 数包装成独立样本数。

### 3.2 动作和分支同一性

每个相位建立五条 OpenFOAM 分支：

```text
omega_target in {-0.75, -0.375, 0, +0.375, +0.75}
```

非零动作从 `omega=0` 以 `|domega/dt|<=1` 的同一确定性 ramp 达到 target 后保持；零动作使用相同 solver、输出和统计设置。每个相位的五条分支必须满足：

1. action onset 前 `U`、`p`、mesh、transport/turbulence、boundary 条件 byte-identical；
2. manifest 记录 source restart 时间、`U/p` SHA256、case config SHA256、OpenFOAM image/digest；
3. `dt=0.005`、19,290-cell 网格和四个 force channels 不变；
4. action schedule 由机器可读文件生成并逐步核对幅值、速率和时间戳；
5. solver failure、提前结束或不利相位不得静默丢弃。

`0.375` 与 `0.75` 提供低幅斜率和幅值非线性；正负号提供符号/对称性检查；零动作提供真正同相位基准。此次不增加更多幅值，否则 2 个验证相位不足以支撑相应复杂度。

五种保持动作只是最小 matched-start seed basis，不是完整控制数据集。真正闭环会产生切换、回零、不同 dwell、rate-limit 饱和以及状态依赖的动态 action primitives；只有这 40 条固定 primitive 达到下述辨识 gate 后，才另行预声明一批动态 primitive acquisition。不得用固定 plateau 数据覆盖不足，却声称已经覆盖了在线闭环的动作分布。

### 3.3 时长和输出

每条分支运行 `80 D/U`：前 `20 D/U` 作为受控 transient，固定分析窗为 `[t0+20,t0+80]`，包含约 9.75 个不受控 shedding periods。`dt=0.005` 对应每条 16,000 个 solver steps；场输出继续使用当前项目已验证的 cadence（目标 `0.1 D/U`），力与动作保留原始高频记录。

这与当前 long-dwell screen 的区别是：这里的 8x5 矩阵、split 和指标在求解前锁定，目的首先是因果覆盖和独立模型验证；不读取 long-dwell screen 的结果来选取动作或相位，也不把两者混成同一 cohort。即便某条恒定动作达到物理门槛，它仍只是 open-loop 分支，不是闭环控制结果。

## 4. 预先声明的指标

### 4.1 严格 matched-start 短时指标

只有每条分支共同的 onset state 是严格反事实起点。对 horizons `1/10/50/100` 分别记录：

- `front/rear Cd`、`front/rear Cl`、`Cd_total` 的 CFD 真值和代理预测；
- 每个非零动作相对同相位零动作的 `Delta Cd_total(a,0)`；
- 同幅正负动作的 `Delta Cd_total(+A,-A)`；
- endpoint 和从 onset 到 horizon 的 window-mean 两套差值；
- pairwise MAE、符号/排序准确率、按相位的 worst case；
- `A=0.375` 到 `0.75` 的局部幅值斜率及正负不对称性。

不得把同一动作轨迹内后续相同 frame index 的窗口当作 matched-state 对；这些窗口可以扩充动力学训练，但必须另标为 divergent-state samples。

### 4.2 长窗物理指标

在固定 `[t0+20,t0+80]` 窗口，对每个动作相对**同一相位零动作**报告：

- `mean(Cd_front)`、`mean(Cd_rear)`、`mean(Cd_total)` 及总阻力变化率；
- front/rear `Cl` 的 mean、fluctuation RMS、mean-absolute value；
- `rear Cl'_rms(control) / rear Cl'_rms(zero)`；
- `abs(mean rear Cl(control)) / rear Cl'_rms(zero)`；
- `Cl_total_rms`，但不把它称作 fluctuation RMS；
- `omega^2`、`|domega/dt|`、signed/positive-only/absolute-work torque proxies；
- 逐 shedding-period block 的均值、离散度和最差 block；
- continuity、Courant、finite-force、完整终止时间等 solver health。

物理 joint pass 仍是：总平均阻力至少降低 `2%`、后圆柱 `Cl' RMS` 比值不超过 `1.05`、`abs(mean rear Cl)/zero rear Cl' RMS` 不超过 `0.10`，并满足既有动作幅值/速率限制。相位平均不能掩盖单相位失败；同时报告 macro、pooled 和 worst-phase。

## 5. 最少案例、时间和资源

2026-10-03 的主机实测基线如下：现有同配置 `t=94..174` 低幅分支单核 OpenFOAM wall time 约 `389 s/case`，raw case 约 `4.2 GiB/case`；对应 curated HDF5 约 `278 MB/case`。当时项目盘可用约 `577 GiB`，统一内存 `MemAvailable` 约 `97 GiB`。

| 阶段 | 案例数 | solver steps | 按实测串行时间 | raw / curated 估算 | 结论权限 |
|---|---:|---:|---:|---:|---|
| commissioning | 9 | 144,000 | 约 58 min | 约 38 GiB / 2.5 GB | 只验证管线 |
| 完整最小矩阵 | 40 | 640,000 | 约 4.3 h | 约 168 GiB / 11.1 GB | 可做 split-backed 模型实验 |

时间是同一机器、单核案例的线性外推，不是承诺；网格/写盘竞争会改变它。正在运行 long-dwell 或 GPU 训练时不启动该矩阵。机器空闲后最多并行 2 条单核 case，预计纯求解墙钟约 2.2 h，再为 staging、QC 和 Curator 留至少 1 h。每次启动前必须满足 `MemAvailable >= 20 GiB`，并预留至少 `250 GiB` 磁盘；不能为了腾空间自动删除已有案例。

9 条 commissioning 必须取自 train split（三个预先选定训练相位、`-0.75/0/+0.75`），通过后补齐其余 train 幅值及 validation/test。它不是缩小版论文数据集，不能据此打开 frozen test 或扩大模型。

## 6. PhysicsNeMo Active Learning 与 Curator 的角色

### 6.1 Active Learning：以后适合，现在不适合决定首批点

当前官方 PhysicsNeMo 源码的 `physicsnemo.active_learning` 是 workflow scaffolding，明确分为 training/fine-tuning、querying、labeling 和可选 metrology；`Driver` 仍需要问题相关的 `QueryStrategy`、`LabelStrategy` 和 `MetrologyStrategy`。官方 surface-CFD 示例用带 variational-GP head 的 GeoTransolver，在冻结验证 manifest 上比较 joint-UQ、random、class-balanced random 和 latent novelty。该示例要求 PhysicsNeMo `26.03+`，并明确 CFD adapter 需按新问题替换；它不是本项目 FNO+OpenFOAM 的即插即用 API。

因此第一批 8x5 采用预声明的 phase-stratified DOE，而不是 AL。理由是：当前 FNO 在独立低幅 validation 上没有胜过 persistence，UQ 未校准，且只有一个严格同初态动作对；用它的误差或置信度选点会把模型偏差反馈进数据集。

当本文件的 seed matrix 完成后，可以在**隔离的新官方容器**中做第二阶段 AL 研究，不改变当前固定 PhysicsNeMo 2.2.2 生产环境：

1. candidate pool 为更密的 phase x amplitude x short action-primitive 网格；
2. query score 由 ensemble/经校准 UQ 的 paired-effect uncertainty 加 phase/action diversity 构成；
3. LabelStrategy 只负责编排有 provenance 的 OpenFOAM matched branches；
4. metrology 固定检查 pooled/worst-phase H100、window-mean、pair-difference 和 persistence；
5. 每轮必须和 phase-stratified random acquisition 做等 CFD 预算、等 seed 对照；
6. frozen test 永不进入 query score、停止规则或 UQ calibration。

2024 年 AL4PDE 表明，用初态和 PDE 参数查询昂贵求解器能提升 neural PDE solver 的数据效率；2025 年 active operator learning 工作进一步把轻量 predictive-UQ 用到包括 FNO 在内的 operator models。这些是采用“seed DOE 后再 AL”的依据，不是当前模型已具备可信 UQ 的证据。PhysicsNeMo 当前 active-learning/UQ 示例中的部分接口仍属 experimental；实施前要按当时官方仓库 commit 重新核对，不能把本设计写成已经存在的 tandem-cylinder API。

### 6.2 Curator：现在适合做确定性 ETL/QC，不负责选点

PhysicsNeMo Curator 官方定位是可扩展的 Source -> Filter -> Sink 离线数据整理，当前项目已有经过测试的 OpenFOAM trajectory 整理路径。下一批可以继续用**项目中已经固定并验证的 Curator 配置/代码**完成：

- VTK/force/action/time 对齐和固定网格采样；
- pressure gauge、mask、四 force channels 和单位检查；
- 每 case 统计、split manifest、源文件/输出 HDF5 hash；
- 对非有限值、时间缺口、动作越界、缺失 force channel fail closed。

Curator 不决定 CFD acquisition、不执行 solver，也不替代 split manifest。官方 Curator 仍标为 beta，API 可能变化；本轮不要为了追随新文档而迁移现有 working pipeline，更不能凭空引入项目中不存在的类名。若以后升级，必须固定 Curator commit/container 并做 byte/statistical regression。

## 7. 何时才尝试更大 FNO

先在相同官方 PhysicsNeMo FNO 架构/训练预算上，用 phase-grouped seed matrix 重训并做至少 3 个训练 seed。只有 validation 同时满足以下条件，才值得进行一次预先声明的容量扩展：

1. H100 terminal `Cd_total` pooled NRMSE `<=10%`，并报告 macro 和 worst-phase；
2. H100 与 onset-to-H100 window-mean 的 total-drag MAE 都严格优于 persistence 和当前 v3 parent；
3. validation 严格 onset pairs 的 `Delta Cd_total` MAE 不大于约 `0.046` 这一 2% 目标尺度；该项只是控制分辨率 gate，不等同于物理 2% long-window pass；
4. 正负/零排序逐相位报告，不能以大量相关 segment 稀释少数错误起点；
5. rollout 稳定、四 force channels 与 normalization/provenance 契约全部通过；
6. 结果随数据增加持续改善，而不是明显的数据/标签或数值问题。

若相同模型仍失败，下一步优先增加**新的独立起始相位或动作 primitive**、检查多步 loss/force supervision，而不是直接堆参数。若通过上述 validation gates，才在不打开 frozen test 的条件下比较一个更大官方 FNO 配置；架构和超参数选定后只评估一次 frozen test。Frozen test 失败则结论仍是 Gate-B 未通过，不能降低 10% 门槛或回头调参后重复使用同一 test。

## 8. 执行前的 fail-closed 清单

- 写入机器可读 acquisition manifest，并冻结相位选择算法、split、动作和窗口；
- 对每个相位验证五分支的 `U/p` SHA256 相同；
- 记录 Git commit、OpenFOAM/PhysicsNeMo/Curator container digest；
- commissioning 仅检查 solver、action、force、时间和整理管线，不看 frozen labels；
- 同步保留失败案例及失败原因，禁止只保留有利动作；
- 新 artifact 名称显式含 `matched_start_acquisition`，避免和 long-dwell screening 混淆；
- 本文件通过审阅后才允许启动 CFD。

## 9. 主要来源

- Zhao et al., *Mitigating the lift of a circular cylinder in wake flow using deep reinforcement learning guided self-rotation*, Ocean Engineering 306 (2024) 118138: [PolyU official record](https://research.polyu.edu.hk/en/publications/mitigating-the-lift-of-a-circular-cylinder-in-wake-flow-using-dee/), [DOI](https://doi.org/10.1016/j.oceaneng.2024.118138).
- Musekamp et al., *Active Learning for Neural PDE Solvers* (2024), [arXiv:2408.01536](https://arxiv.org/abs/2408.01536).
- Winovich et al., *Active operator learning with predictive uncertainty quantification for PDEs* (2025), [arXiv:2503.03178](https://arxiv.org/abs/2503.03178).
- *Deep Reinforcement Learning Discovers a Novel Control Algorithm for Mitigating Flow-Induced Vibrations in Underactuated Tandem Cylinders* (2026 preprint), [arXiv:2605.20778](https://arxiv.org/abs/2605.20778). This is context for the continuing 2024–26 tandem-cylinder/closed-loop trend, not evidence for the present fixed-cylinder objective.
- NVIDIA, [PhysicsNeMo Active Learning user guide](https://docs.nvidia.com/physicsnemo/26.05/user-guide/active_learning.html) and [official repository](https://github.com/NVIDIA/physicsnemo). Live source inspected at commit `b45a5c810c741e6b41f8515be24c51121f8fc21f`, including `physicsnemo/active_learning` and `examples/cfd/external_aerodynamics/active_learning_aero`.
- NVIDIA, [PhysicsNeMo Curator official repository](https://github.com/NVIDIA/physicsnemo-curator) and [PhysicsNeMo for PyTorch users](https://docs.nvidia.com/physicsnemo/latest/user-guide/physicsnemo_for_pytorch.html).
