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
- 单进程 `torchrun`，批量默认为 4；PyTorch allocator fraction 固定为 0.20，按 GPU 报告总量约 121.69 GiB 算上限约 24.34 GiB。这个限制只覆盖 PyTorch 缓存分配器，不是对所有 CUDA 分配的硬上限。
- `spark_gpu_guard.py` 运行前要求 `MemAvailable >= 20 GiB + allocator 上限 + 4 GiB`，并每 5 秒检查一次；低于 20 GiB 时向整组训练进程发 SIGTERM，必要时 SIGKILL。该时间间隔内的瞬时波动不能绝对排除，正式训练还须观察日志。
- Spark 是 CPU/GPU 共享 DRAM 的 UMA，没有独立“显存”指标。`nvidia-smi` 显示 N/A 是预期行为。实测 CFD 写盘时 Linux page cache 超过 100 GiB，`cudaMemGetInfo` 仅显示约 8 GiB free，但 `MemAvailable` 约 113 GiB。依据 [NVIDIA DGX Spark Porting Guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html) 的说明，前者未计入可回收 OS 内存；本守护程序同时记录两者、用 `MemAvailable` 判断共享内存余量，不清理其他进程缓存。
- `SMOKE=true` 仅表示原训练脚本的一轮低密度抽样与验证，不代表控制收益或正式训练达标。正式判定需看独立验证/测试集的误差、rollout 稳定性以及与零动作 CFD 基线的物理指标对照。

Spark 自动衔接脚本 `scripts/run_tandem_fno_pipeline_spark.sh` 等待 Curator 完整质检标记，然后在同一项目内做官方 HDF5Reader/DataLoader 验证、一次抽样冒烟（训练轨迹步长 80、验证步长 160）与 5 epoch 实数据先导训练。每一步失败就停止，不会绕过显存守护。80 epoch 完整训练要在先导模型和测试集评估后决定；不把冒烟指标当物理结论。

此文档记录的是运行方案及内存守护测试；完整 CFD 数据仍在生成时，不得宣称 FNO 已训练完成。
