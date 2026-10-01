# 恒定后柱旋转的粗/中网格配对 CFD 检验（2026-10-02）

本检验在 DGX Spark `/workspace/fluid_control` 的独立 OpenFOAM 容器中运行，用于补足当前19,290单元训练数据尚未做网格敏感性检验的缺口。**启动中，不代表已经网格收敛**。入口为 `bash cfd/tandem_cylinders/run_constant_rotation_grid_pair_spark.sh`，实时日志为 `artifacts/tandem_cylinders/constant_p100_grid_pair_pipeline.log`。

两条工况均为二维 Re=100、D=1、前后圆柱中心距5D、入口速度1、运动黏度0.01、后圆柱恒定 `omega=+1`、前柱不转。使用 OpenFOAM v2512 `pimpleFoam`、二阶 `backward` 时间格式、`dt=0.005`、`t=0..160`，并从同一规则定义的前柱后方2%横向速度扰动出发。粗网格 `control_small_p100` 约19,290单元、后柱96个壁面；中网格 `control_grid_p100_medium` 约77,160单元、后柱192个壁面。两者均由项目已有 `make_baselines.write_case` 生成，在生成时执行 `blockMesh`、`checkMesh`、`setFields`；中网格脚本和粗网格脚本均拒绝覆盖既有同名 case。

粗网格全场输出间隔为0.1，供原有小数据集使用；中网格为2.0，降低重复场存储。两者的 `forceCoeffs` 和32个 `wakeProbes` 均每时间步输出，因此最终可在相同 `t=80..160` 窗口比较前/后柱平均阻力、升力 RMS、脱涡零交叉频率及探针统计。输出频率不同不应影响求解时间步；任何差异均须从实际日志确认，而不是仅凭配置假定。

求解器镜像固定为 `opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319`，每次容器 `--network none --read-only --cpus 4 --memory 8g`，不改宿主环境。启动前要求项目磁盘可用至少300 GiB、主机 `MemAvailable` 至少40 GiB；GPU0 训练另由5秒轮询守护保持至少20 GiB可用统一内存。

验收顺序：两条求解必须均达到 `t=160`、`checkMesh` 显示 Mesh OK、数值日志无异常且力/探针序列完整，再运行 `validate_constant_rotation_grid_pair.py` 形成独立数值 QC，最后运行 `compare_convergence.py control_small_p100 control_grid_p100_medium`。数值 QC 会检查各32,000步、Courant数、连续性、粗/中网格分别1,600/80个全场快照、32探针、前后柱力系数和 `t=160` 后柱壁面速度。若任一指标的网格差异大，不能称粗网格结果定量收敛；即使本恒定转速工况接近，也不能推断所有时变动作日程或闭环轨迹收敛。最终比较 JSON 与数值 QC 结果将在计算完成后另行同步，不在此预填虚构结果。
