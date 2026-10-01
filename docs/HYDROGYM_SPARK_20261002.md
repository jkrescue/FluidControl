# HydroGym 在 DGX Spark 上的 ARM64 验收（2026-10-02）

## 软件与真实数据

- HydroGym 官方源码固定 commit `4ab9854dea3d84e38a59c25e0f5835a00cf8225f`；Firedrake 官方 ARM64 基础镜像固定 digest `sha256:b276f7252ca072742b3aa3461b376894c74a7b2fe135590c0485ae96e5bbd442`，派生配置见 `docker/hydrogym/Dockerfile.spark`。
- 仅在派生容器中安装 HydroGym、Stable-Baselines3 2.7.1 和 PyTorch 2.6.0 CPU 版，不改变宿主机 Firedrake/Python。官方旋转圆柱网格 `medium.msh` 随仓库提供；ARM64 缺少可用 gmsh Python wheel，因此不安装只用于另行制网的可选 gmsh。
- 环境从 HydroGym 官方公开的 `dynamicslab/HydroGym-environments` 下载 `RotaryCylinder_2D_Re100_medium_FD`，快照修订 `cf5ef6748443ae72e9647f398487e98d421a7243`，共 22 个真实 Firedrake checkpoint。缓存只保存在 DGX Spark 的项目 artifacts，不进 GitLab。
- 官方 `FlowEnv`、`RotaryCylinder`、`SemiImplicitBDF`、50 个压力探针和 Stable-Baselines3 `PPO("MlpPolicy")` 已在 ARM64 上实测运行；官方 64 环境步示例保存了模型与归一化统计。

## 可审计的物理奖励

`scripts/train_hydrogym_rotary_spark.py` 只包装官方环境的奖励和日志，不更换流体求解器或创建新模型。每个 `dt=0.01` 控制步：

`r = -dt [ Cd + 0.2 Cl² + 0.01 omega² + 0.001 (omega-omega_prev)² ]`

日志分别保留四个奖励分项、`Cd`、`Cl`、请求和执行动作。256 步软件烟雾测试后的 5 个未见 checkpoint × 2 策略 × 30 步回放，共 300 行；所有指标有限值，逐行四项和与奖励差值为 0。短策略在多数相位降低阻力不到 0.1%，但增大 `Cl RMS`，**不能声称控制有效**。详见忽略目录 `artifacts/hydrogym/physical_reward_smoke_single/audit.json`。

正式训练由 `scripts/run_hydrogym_rotary_spark.sh` 调用 8,192 个 PPO 步，容器限制 4 CPU/8 GiB、无 GPU，结束后对末尾 5 个未见 checkpoint 分别执行零控制与冻结策略的 600 步配对评估，前 100 步不计入指标。训练启动时结果尚未产生。

HydroGym 官方 JAX Kolmogorov 示例默认 `reward_alpha=0`，其默认奖励仅惩罚动作且每步谱求解成本很高；该默认示例不适合用来论证流动控制效果，已终止这条无效试跑。Spark 上的 HydroGym 基准仍是单个旋转圆柱，**不等于本项目串列双圆柱工况**；双圆柱代理和 OpenFOAM 闭环验证按原路线继续。

来源：[HydroGym 官方仓库](https://github.com/dynamicslab/hydrogym)、[Firedrake 官方安装与容器说明](https://www.firedrakeproject.org/firedrake/install.html)、[官方公开环境数据](https://huggingface.co/datasets/dynamicslab/HydroGym-environments)。
