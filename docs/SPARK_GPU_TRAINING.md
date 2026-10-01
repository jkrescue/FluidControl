# DGX Spark 单 GPU 训练入口（2026-10-02）

目标数据是 OpenFOAM 生成的串联双圆柱扩展版真实 CFD 轨迹；该入口不替代 HydroGym 的单圆柱强化学习基准。仅在 `data/curated/tandem_cylinders_expanded_v1/manifest.json` 完成并通过数据质检之后运行。

```bash
# DGX Spark 上，在 /workspace/fluid_control 内
SMOKE=true BATCH_SIZE=4 OUTPUT_DIR=artifacts/tandem_fno_expanded_spark_smoke \
  bash scripts/run_tandem_fno_spark.sh
# 冒烟和数据审计通过后，另起正式目录；默认扩展配置为 80 epoch
BATCH_SIZE=4 OUTPUT_DIR=artifacts/tandem_fno_expanded_spark_v1 \
  bash scripts/run_tandem_fno_spark.sh
```

- 使用固定的 `fluid-control-physicsnemo:2.2.2` 派生容器，仅暴露 GPU 0；`--network none`、`--memory 64g`，不改主机 Python 环境。现有 PhysicsNeMo 官方 `FNO`、`HDF5Reader`、`DataLoader` 与训练辅助 API 保持原样；模型结构不因迁移而改变。
- 单进程 `python`，批量默认为 4；PyTorch allocator fraction 固定为 0.20，按 GPU 报告总量约 121.69 GiB 算上限约 24.34 GiB。这个限制只覆盖 PyTorch 缓存分配器，不是对所有 CUDA 分配的硬上限。
- `spark_gpu_guard.py` 运行前要求 `MemAvailable >= 20 GiB + allocator 上限 + 4 GiB`，并每 5 秒检查一次；低于 20 GiB 时向整组训练进程发 SIGTERM，必要时 SIGKILL。该时间间隔内的瞬时波动不能绝对排除，正式训练还须观察日志。
- Spark 是 CPU/GPU 共享 DRAM 的 UMA，没有独立“显存”指标。`nvidia-smi` 显示 N/A 是预期行为。实测 CFD 写盘时 Linux page cache 超过 100 GiB，`cudaMemGetInfo` 仅显示约 8 GiB free，但 `MemAvailable` 约 113 GiB。依据 [NVIDIA DGX Spark Porting Guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html) 的说明，前者未计入可回收 OS 内存；本守护程序同时记录两者、用 `MemAvailable` 判断共享内存余量，不清理其他进程缓存。
- `SMOKE=true` 仅表示原训练脚本的一轮低密度抽样与验证，不代表控制收益或正式训练达标。正式判定需看独立验证/测试集的误差、rollout 稳定性以及与零动作 CFD 基线的物理指标对照。

启动预检发现：`torchrun --standalone` 在隔离容器 `--network none` 下持续尝试解析容器主机名，无法进入训练；改用静态回环地址虽进入进程，但单卡 NCCL 初始化报 CUDA OOM。两次均为小型预检，没有训练或改写数据。
因此 GPU 0 启动器改用 PhysicsNeMo 官方 `DistributedManager.initialize()` 支持的单进程 `python` 路径，不建立无意义的单卡 NCCL 通信。已在同一容器和 20 GiB 余量守护下验证 `rank=0`、`world_size=1`、`cuda:0` 及 CUDA 分配；完整 FNO 训练仍待全量数据质检。

在完整归一化数据就绪前，`scripts/validate_tandem_fno_real_frame_spark.py` 已用首条真实 CFD HDF5 的 4 个相邻时间窗做官方 47,222,525 参数 FNO 的 GPU 0 前向/反向验收：输出 `[4,5,128,256]`，损失和梯度均有限值，CUDA 峰值 allocated 0.859 GiB、reserved 1.004 GiB。默认模式**不执行优化器更新**、使用未归一化原场，打印的损失不是模型精度。

同一脚本新增可选 `--optimizer-step --batch-size 4` 预检，在真实 CFD HDF5 上完成一次内存中的 AdamW 参数更新，并逐项检查更新后参数有限值。隔离容器内 GPU 0 实测 CUDA 峰值 allocated 0.949 GiB、reserved 1.180 GiB，守护启动前 `MemAvailable` 109.133 GiB；未写 checkpoint，也不代表已完成训练或证明模型精度。

进一步以 `--capture-step --batch-size 4` 使用容器中实际安装的 PhysicsNeMo 2.2.2 官方 `StaticCaptureTraining`（`use_graphs=False`、`use_amp=False`、梯度裁剪）在相同真实 CFD 数据上完成一次内存中的更新，参数确实变化且保持有限值，守护退出码 0。GPU 0 峰值 allocated 0.949 GiB、reserved 1.180 GiB；启动时 `MemAvailable` 108.818 GiB。日志保存在 Spark 项目 `artifacts/tandem_cylinders/expanded_fno_capture_preflight_spark.log`；未写模型 checkpoint，不能视作先导模型精度。

同一真实 CFD 轨迹上，对官方 `StaticCaptureTraining` 单次更新计时（已排除容器启动，未含 DataPipe I/O）；所有测量均在 GPU 0、同一 20 GiB 余量守护和 0.20 allocator fraction 下通过：

| batch | 单步秒数 | CUDA reserved 峰值 GiB | 估算样本/秒 |
| ---: | ---: | ---: | ---: |
| 4 | 0.201 | 1.180 | 19.9 |
| 16 | 0.401 | 3.148 | 39.9 |
| 32 | 0.618 | 6.037 | 51.8 |
| 64 | 1.142 | 11.908 | 56.0 |

据此，自动流水线的抽样冒烟与 5 epoch 先导训练暂选 `BATCH_SIZE=64`，而手动启动器仍保留保守默认值 4。单步数字不是端到端吞吐或模型精度，正式运行需继续监控 DataPipe、峰值内存和验证误差。

Spark 自动衔接脚本 `scripts/run_tandem_fno_pipeline_spark.sh` 等待 Curator 完整质检标记，然后在同一项目内做官方 HDF5Reader/DataLoader 验证、一次抽样冒烟（训练轨迹步长 80、验证步长 160）、5 epoch 实数据先导训练，以及留出测试集 1/10/50 步 rollout 评估。每一步失败就停止，训练与评估都不绕过 20 GiB 内存余量守护。80 epoch 完整训练要在先导模型和测试集评估后决定；不把冒烟指标当物理结论。

流水线的阶段标记只在 `scripts/validate_tandem_fno_stage.py` 通过后打印：检查冒烟/先导训练 epoch 连续且指标有限，留出集必须包含 4 条真实 CFD 测试轨迹的 1/10/50 步稳定预测，并计算相对于持久性预测基线的场误差比值。比值只是诊断，不把先导训练的有限误差自动判为合格控制策略；物理收益仍需独立 CFD 回放。

进入 DataPipe 前，流水线会对全部 32 条 HDF5 逐帧检查，并将转速及前后柱 Cd/Cl 对齐到原始 OpenFOAM 动作表和 `coefficient.dat`；只有完整原始标签审计通过才放行训练。

此外，训练前运行 `scripts/audit_curated_control_response.py`，要求 32 条轨迹与共同无控制重启场初始受力接近、后半窗受力非退化地偏离，并把逐条数值保存在 Spark `artifacts/tandem_cylinders/expanded_control_response_spark.json`。此为数据物理响应粗筛，不把逐时刻差异当成控制性能。

先导模型还会在同一留出 CFD 流场上比较实际转速、零转速和符号翻转转速输入的 1/10/50 步误差，作为动作条件响应诊断。零/翻转输入没有对应的新 CFD 真值，只是代理模型输入消融，不能把误差差异当成物理控制收益。

三种动作输入的 1/10/50 步场与后柱力误差还将由现有 `scripts/summarize_tandem_action_sensitivity.py` 汇总为 `artifacts/tandem_fno_expanded_spark_5epoch/action_sensitivity_summary.json` 和同名 Markdown 表。只有观察到实际动作输入相对错误动作更可靠，才可考虑后续代理控制实验；仅有这张表仍不足以证明 CFD 控制收益。

此文档记录的是运行方案及内存守护测试；32 条 CFD 已通过数值质检，但全量 Curator HDF5 仍在整理，不得宣称 FNO 已训练完成。
