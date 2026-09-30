# HydroGym 旋转圆柱基准

## 配置

- HydroGym commit：`4ab9854dea3d84e38a59c25e0f5835a00cf8225f`
- 求解器：Firedrake `SemiImplicitBDF`
- 环境：`RotaryCylinder`，`Re=100`，medium 网格，`dt=0.01`
- 接口：HydroGym `FlowEnv` / Gymnasium
- 控制：单一圆柱角速度，动作范围 `[-π/2, π/2]`
- 观测：50 个尾流压力探针
- 强化学习：Stable-Baselines3 PPO

## 验证结果

官方环境 smoke 完成 3 个动作步，每个动作包含 2 个 CFD 子步，退出码为 0。输出包含 PVD 时序、Firedrake checkpoint 和探针观测日志。

PPO smoke 完成 256 个环境步和 4 次 rollout，耗时约 55 秒，平均约 5 FPS。最后一次训练统计为：

| 指标 | 数值 |
| --- | ---: |
| `approx_kl` | 0.00129 |
| `clip_fraction` | 0.0000 |
| `explained_variance` | 0.787 |
| `policy_gradient_loss` | -0.00170 |
| `value_loss` | 0.102 |

最终模型和 `VecNormalize` 统计均已生成。该运行只验证完整软件链路和数值稳定性；256 步不足以评价阻力、升力脉动或控制收益，不能作为论文控制结果。

## 物理量审计

从同一 checkpoint 分别运行零动作、固定种子随机动作和冻结 PPO 200 步，丢弃前 50 步后统计：

| 策略 | 平均 `Cd` | `Cl RMS` | 动作 RMS | `Cd` 相对零动作 | `Cl RMS` 相对零动作 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 零动作 | 1.47093 | 0.24608 | 0.00000 | — | — |
| 随机动作 | 1.46892 | 0.25323 | 0.23740 | -0.14% | +2.90% |
| PPO smoke | 1.47374 | 0.26431 | 0.06762 | +0.19% | +7.41% |

PPO smoke 没有产生控制收益。该审计跨度只有 2 个无量纲时间单位，短于典型 Re=100 圆柱涡脱落周期，因此只用于检查方向和量纲。正式对照需要覆盖多个涡脱落周期并使用至少 5 个随机种子。逐策略统计保存在 `policy_audit.json`。

HydroGym 当前旋转圆柱默认目标为瞬时阻力 `Cd`，reward 为 `-dt × Cd`；其中没有升力波动或控制能耗项。串列双圆柱正式任务应使用明确的滑动窗口 `Cl RMS`、平均阻力和转速能耗组合，并逐项记录，以免策略利用未惩罚的高频或高幅值旋转。

## 环境兼容性

HydroGym 的锁定依赖会把基础 Firedrake 镜像中的 NumPy 1.24 和 mpi4py 3.1.5升级，导致 PETSc/Firedrake 扩展 ABI 不兼容。项目镜像保留基础镜像的 mpi4py 构建，并固定 NumPy 1.24.4。修复后 Firedrake、PETSc、HydroGym、PyTorch 和 Stable-Baselines3 导入检查通过。

全栈 Blackwell 镜像的压缩层体积超过当前工作盘安全余量，因此改用 HydroGym 仓库 Dockerfile 指定的 Firedrake 基础镜像。该选择保留官方 Firedrake 后端和 HydroGym API，同时减少无关 GPU/HPC 后端占用。
