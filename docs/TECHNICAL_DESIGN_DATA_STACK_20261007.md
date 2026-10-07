# 串列圆柱代理建模与真实反馈控制：技术架构及数据工具栈方案

> 2026-10-07 只读设计说明；不授权训练、PPO、CFD、转换或晋级。范围仅为 Re=100、L/D=5、后圆柱旋转案例；不承诺从空 Git 仓库一键重建外部大文件。

## 1. 目标、总体路线与能力边界

目标是把真实 OpenFOAM 数据变成可审计的 FNO 样本，在 HydroGym/Gymnasium 环境中用冻结代理训练 SB3 PPO，再将冻结 CPU 策略接回真实 OpenFOAM。默认部署链是 **CPU PPO → 真实 CFD → 新观测**；没有在线 FNO 或 MPC。代理精度门和真实控制物理门分开，前者仍未完整通过。

```text
OpenFOAM v2512 case
  └─ U,p,phi + probes + forceCoeffs + omega table
      └─ foamToVTK(U,p; no-boundary)
          └─ PhysicsNeMo Curator: VTKSource → 项目 Filter/Sink
              └─ HDF5Reader → 项目 Dataset/DataPipe
                  └─ PhysicsNeMo FNO → checkpoint + manifest
                      └─ 项目 HydroGym FlowEnv adapter → Gymnasium API
                          └─ SB3 PPO + VecNormalize → policy.zip
                              └─ 项目 OpenFOAM 桥接 → control/zero 配对验证
```

## 2. 锁定运行时和职责边界

| 层 | 项目实际绑定 | 官方职责 | 项目职责 |
|---|---|---|---|
| CFD | OpenFOAM v2512 容器及批准中的 image digest | `pimpleFoam`、边界、probes、`forceCoeffs`、`foamToVTK` | case/网格、旋转表、分段推进、配对分支、日志、cleanup |
| ETL | PhysicsNeMo Curator 0.1.0 | `Source/Filter/Sink`、`VTKSource`、pipeline | case发现、ROI采样、时钟/标签、质量Filter、HDF Sink |
| 读取 | PhysicsNeMo 2.2.2 `HDF5Reader` | HDF首维随机读取为CPU tensor；Dataset/DataLoader抽象 | window/rollout、normalize、mask、action拼接、metadata |
| 模型 | PhysicsNeMo 2.2.2 官方 `FNO`/checkpoint API | 二维谱算子、官方保存/加载 | 7通道语义、loss、冻结范围、训练/评估编排、manifest |
| 环境 | HydroGym commit `4ab9854dea3d84e38a59c25e0f5835a00cf8225f` | `FlowEnv` 与 Gymnasium 生命周期 | tandem flow/solver/reward wrapper、因果history、终止条件 |
| RL | Stable-Baselines3 2.7.1 | PPO、DummyVecEnv、VecNormalize、policy存取 | 固定超参、冻结FNO、诊断callback、policy/Vec身份 |
| 部署 | Docker + systemd transient unit | 进程/资源隔离 | CPU策略、CFD桥接、动作filter、receipt、配对统计 |

`uv.lock` 通用环境锁 PhysicsNeMo 2.2.2、PyTorch 2.14.1、h5py 3.16.0；Curator py312 环境实查为0.1.0。PPO是批准中逐文件哈希的vendored runtime，不能用当前site-packages替代。

## 3. 物理、shape和时钟契约

- 两个固定串列圆柱，`D=U∞=ρ=1`、`ν=.01`，所以`Re=100`；中心距`L/D=5`，仅后柱旋转。
- solver步长`.005 D/U`；每`.1 D/U`写学习帧并反馈一次，即20个solver steps。
- `t*=tU∞/D`。动作`ω*=ΩD/U∞`；本配置数值等于OpenFOAM角速度，表面速度比为`ω/2`。
- action Box为`[-.75,.75]`，每周期`|Δω|≤.1`；区间内使用线性omega table。
- H1从真实状态一步预测；AR100把上一步预测递归输入。epoch、optimizer update、LBFGS closure、feedback step、solver step不可混用。

## 4. OpenFOAM：真值与控制执行

输入是`constant/polyMesh`、`system/controlDict/fvSchemes/fvSolution`及restart目录的`U,U_0,p,phi,phi_0,uniform/time`。case为准二维一层网格，`frontAndBack=empty`；后柱`rotatingWallVelocity`，入口`U=(1,0,0)`、出口压力0。

输出契约：

- 每`.1 D/U`的`U,p`；求解器另维护`phi`和旧时层；
- 32个wake probes的三分量速度，在线观测取`u,v`；
- 前/后柱`forceCoeffs`，项目顺序`[front Cd, front Cl, rear Cd, rear Cl]`；
- `functionObjectProperties`中的pressure/viscous分量，不能用pressure-only替代总力；
- applied omega、solver log、restart tree、case inventory。

启动前核image/source/restart SHA、输出不存在、MemAvailable、磁盘和cgroup。运行与后审计按各自合同拒绝实际读取到的非有限场、观测或力，以及时钟缺口、冲突restart样本、错列头和越界动作；这不表示每个solver step的全部网格字段都经过独立finite复查。进程只清理自己捕获的container ID。新配对实验的control/zero须从匹配的同一初始物理状态出发，并使用相同网格和时间窗；E109/E114这类既有连续轨迹恢复必须分别使用受控、zero各自正确的分支restart，不能把已受控分支重置成zero状态。

## 5. VTK与Curator：case到HDF5

### 5.1 VTK导出

实际成功命令语义是`foamToVTK -case <readonly-view> -fields '(U p)' -no-boundary -name <output>`。raw case只读，输出独立。验证完整`.1 D/U`网格、每帧`U/p`、有限且递增的VTK `TimeValue`。float64源时钟是force/omega插值基准；不能先转HDF float32再倒推标签。

### 5.2 Curator pipeline

官方Curator提供`Source/Filter/Sink/Param/run_pipeline`和`VTKSource`。项目扩展负责：

1. `Source`产生case record并绑定绝对force/action/config路径；
2. `VTKSource`读取网格及`U,p`；
3. 项目Filter在固定ROI `x∈[8,25], y∈[4,11]`采样`128×256`，产生`state=[u,v,gauge-p]`；
4. 质量Filter检查finite、mask覆盖、时间单调和frame数；
5. 项目Sink写split、attrs、manifest和normalization。

Curator不生成物理真值；OpenFOAM才是真值源。采样、标签、split和schema是项目代码。pipeline可用sequential/n_jobs=1保持可审计顺序。

### 5.3 HDF schema

| key | 单帧shape/dtype | 含义 |
|---|---|---|
| `state` | `[3,128,256]`, float32 | 物理`u,v,gauge-p` |
| `mask` | `[1,128,256]`, float32/bool语义 | 流体区1，固体区0 |
| `omega` | 标量或`[1]`, float32 | 端点applied action |
| `force` | `[4]`, float32 | 前/后Cd、Cl；下一端点为监督标签 |
| `time` | 标量或`[1]`, float32 | 无量纲`t*`；另以float64源时钟核验 |

每个HDF带case/split attrs；根目录有`manifest.json`及train-only `normalization.json`。字段、shape、finite、time cast、原force/omega、split或normalization SHA任一不符即失败。

## 6. HDF5Reader、Dataset与DataPipe

`src/fluid_control/tandem_datapipe.py`用官方`HDF5Reader(path, fields=[state,mask,omega,force,time])`按index读CPU tensor；项目`TandemWindowDataset`/`TandemRolloutDataset`定义语义：

- state按train-only统计量normalize后乘mask；
- 当前/下一动作广播为`[1,128,256]`；
- 单步输入`x=[state3,mask1,omega_now1,omega_next1]`，shape`[6,128,256]`；
- 目标`delta=normalized(next_state)-normalized(state)`，shape`[3,128,256]`；
- force取下一端点，按固定channel mean/std归一化；
- H100返回`state[3,H,W]`、`target_state[100,3,H,W]`、`omega[101,1]`、`target_force[100,4]`、`mask[1,H,W]`。

metadata保存case/step/split/rollout_steps，不进入模型。Reader只负责IO；项目transform定义时间和物理语义。首尾H100需与h5py逐值对照。

## 7. 官方FNO与项目输出语义

官方二维FNO接受`[B,Cin,H,W]`并输出同空间分辨率。项目基本模型`Cin=6,Cout=7,dimension=2`：

- `0:3`是归一化状态增量；`q_next=(q+delta)*mask`，不是直接下一状态；
- `3:7`是四力空间读出图，在mask内平均池化并反归一化为Cd/Cl；
- 保留B的流场分支是冻结K1；后续气动力训练不等于全流场重训。

训练编排、H1/AR loss、冻结、梯度累积及候选协议均是项目代码。模型官方checkpoint fresh reload后，项目consumer再核kind/status、architecture、epoch、normalization、source/runtime SHA和precision。非有限、错channel或manifest漂移立即失败。

## 8. HydroGym/Gymnasium与SB3 PPO

HydroGym `FlowEnv`提供Gymnasium生命周期；项目`TandemSurrogateFlow`与`TandemFNOStepper`封装冻结FNO。canonical reward wrapper的`reset()`从真实curated frame及62点因果force history初始化；这项62点要求属于canonical因果reward合同，不应泛化成所有历史或rear-only `TandemSurrogateFlow`适配器的通用要求。`step(action)`执行动作限制、代理一步、reward ledger和终止检查。

canonical四力观测为float32`[69]`：`32×(u,v)=64`、四力、applied omega；action为float32`[1]`。旧rear-only bridge为`[67]`，不可与69维policy混用。项目wrapper检查Gymnasium `(obs,reward,terminated,truncated,info)`的shape/finite/max_steps。

SB3 PPO固定核心参数：`n_steps=128,batch_size=256,n_epochs=4,lr=3e-4,gamma=.99,gae_lambda=.95,clip=.2,max_grad_norm=.5`。四环境用`DummyVecEnv`；`VecNormalize(norm_obs=False,norm_reward=False)`仍是冻结部署身份。FNO设`eval()`和`requires_grad=False`，并验证未进入PPO optimizer。

reward是项目`canonical_joint_v1`六项成本，不是HydroGym/SB3默认reward：drag screen、2% drag gate、rear-Cl波动比gate、均值偏置gate、动作幅值、动作变化。它依赖真实因果history，不允许未来力补历史。

## 9. 真实OpenFOAM部署桥接

冻结`policy.zip + VecNormalize`在CPU加载。每`.1 D/U`：

1. 从真实case读取同一目标时刻32 probes、前后forceCoeffs、上次applied omega；
2. 组成`[69]`并核headers、坐标、列顺序、时钟唯一性和finite；
3. 一次`policy.predict`得到请求omega，经方向恢复和唯一幅值/速率filter得到applied omega；
4. 写起止omega线性table，推进20个`pimpleFoam` steps；
5. 读新观测，同时推进同restart的zero分支，记录request/applied/source/log。

部署不调用FNO、不更新PPO、不是MPC。约1.37秒/反馈仅是本机墙钟，不证明硬实时；动作平方成本不是净节能。

## 10. 资源、存储、版本与接口治理

- DGX Spark是统一内存架构；保持物理`MemAvailable≥20 GiB`，实际批准常用startup 50 GiB/runtime 22 GiB门。
- GPU任务显式GPU0、allocator/cgroup；CPU CFD/转换使用noSwap、CPUQuota、MemoryMax、RuntimeMaxSec及disk reserve。GPU利用率不是科研进度。
- Git仅存source/config/tests/approval/report/SHA inventory；HDF、VTK、case、checkpoint、PPO zip、VecNormalize和镜像留Spark artifact store。
- 每个运行绑定绝对source/input/runtime SHA、唯一unit/invocation/output；输出存在即拒绝。R1失败与R2恢复都保留。
- train/dev/fixed-six/真实CFD是不同协议；重复开发集不是独立测试。
- force与observation必须同一端点时钟；请求/施加动作分开；state/force normalization、channel order及mask是跨层接口合同。

## 11. 安全dry-run与外部artifact

真实入口仅做预检：

```bash
cd /workspace/fluid_control
python3 scripts/reproduce_canonical_closed_loop.py
```

预期`PREFLIGHT_PASS_NOT_RUNNING`。它核批准、driver、policy、VecNormalize、restart、镜像、SHA、内存和磁盘；不创建输出、不启动CFD。执行需要新approval/unit/空output；本文不授权`--execute`。

单独Git clone不足以从零复现。还需curated HDF/normalization、K1/B checkpoints/manifests、E082 policy/VecNormalize、OpenFOAM restart/case、锁定镜像及inventory中existence-only大payload。权威索引：`docs/FINAL_MODEL_AND_REPRODUCTION_MANIFEST_20261007.md`、`docs/CANONICAL_CHAIN_INPUT_INVENTORY_20261007.json`。

## 12. 官方primary资料与访问记录

以下于**2026-10-07**核对；`latest/master`只解释API，项目仍以锁定版本/source SHA为准：

- PhysicsNeMo FNO：<https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/models/fnos.html>
- PhysicsNeMo HDF5Reader：<https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/datapipes/physicsnemo.datapipes.readers.html>
- PhysicsNeMo DataPipe：<https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/datapipes/physicsnemo.datapipes.html>
- PhysicsNeMo Curator：<https://docs.nvidia.com/physicsnemo/latest/user-guide/curator.html>
- Curator官方仓库（beta/API可能变化）：<https://github.com/NVIDIA/physicsnemo-curator>
- HydroGym官方仓库/API：<https://github.com/dynamicslab/hydrogym>、<https://dynamicslab.github.io/hydrogym/>
- Gymnasium Env：<https://gymnasium.farama.org/api/env/>
- SB3 PPO：<https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html>
- OpenFOAM rotatingWallVelocity：<https://doc.openfoam.com/2312/tools/processing/boundary-conditions/rtm/derived/wall/rotatingWallVelocity/>
- OpenFOAM forceCoeffs：<https://doc.openfoam.com/2606/tools/post-processing/function-objects/forces/forceCoeffs/>

## 13. 当前结论

已实现并审计：真实CFD→VTK→Curator/HDF→Reader/DataPipe→官方FNO→HydroGym/SB3 PPO→冻结CPU policy→真实OpenFOAM反馈。限定工况B闭环通过物理门。未完成：完整代理精度、跨工况泛化、有效FNO-MPC、网格/时间步独立性、物理实时与净能耗。工具链连通不能替代科学准入。
