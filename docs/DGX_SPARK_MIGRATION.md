# DGX Spark 迁移记录

本记录对应代码提交 `ba874b4`。项目代码已迁入 `workspace/fluid_control/physicsnemo_control`；原始 CFD 场、Curator 数据和 PhysicsNeMo checkpoint 不在代码仓库内，不能把代码迁移误认为实验状态已恢复。

## 环境验收

| 组件 | 当前状态 |
| --- | --- |
| 计算节点 | Linux ARM64，单张 NVIDIA GB10；CUDA 容器可识别 GPU |
| OpenFOAM | 固定的 OpenCFD v2512 ARM64 容器镜像已存在，项目的 `run_openfoam.sh` 可调用 |
| CFD 烟雾测试 | `tandem_static` 的 `blockMesh`、`checkMesh` 和 50 步 `pimpleFoam` 已通过；`Mesh OK`，求解器正常结束 |
| PhysicsNeMo | 从 26.06 ARM64 容器派生的 `fluid-control-physicsnemo:2.2.2` 已构建；GB10、CUDA、项目关键 API 导入通过；正式训练与 DataPipe 尚待数据恢复后验收 |
| GitLab | 已为此节点登记专用认证公钥；`ssh -T`、`git fetch` 和 `git push --dry-run` 均通过，`main` 跟踪 `origin/sanitized-main`，当前提交为 `ba874b4` |

## 数据恢复判断

已重建 `tandem_backward_dt005` 无控制基线。它使用 `Re=100`、`L/D=5`、`dt=0.005` 和二阶 `backward`，覆盖 `t=0..160`，保留 `t=80` 重启场。该场是后续控制轨迹和长窗 OpenFOAM 验证的共同起点，不是代理模型训练集。

生成顺序为 `make_baselines.py --case tandem --profile backward`、`blockMesh`、`checkMesh`、`setFields`、`run_baseline.sh backward`。所有输出位于 `cfd/tandem_cylinders/cases/tandem_backward_dt005/`，不覆盖其他算例。

求解器正常完成 32,000 步，最大 Courant 数 0.26588，80 个全场时间快照与 32,000 个探针样本通过有限值检查。`t=80..160` 后柱平均 `Cd=0.909244`、平均 `|Cl|=1.069065`、最大 `|Cl|=1.647262`，与迁移前归档的 0.909245、1.069066、1.647281 一致。原始算例约 320 MB；分析保存在算例的 `analysis.json`。

```bash
cd ~/workspace/fluid_control/physicsnemo_control
cat cfd/tandem_cylinders/cases/tandem_backward_dt005/exit_code
python3 -m json.tool cfd/tandem_cylinders/cases/tandem_backward_dt005/analysis.json
```

若能从备份恢复原 FNO checkpoint、数据 manifest 和 `t=80` CFD 重启场，可先继续长窗闭环验证，无需重做全部训练轨迹。若 checkpoint 无法恢复，PhysicsNeMo 控制器必须重新训练；应先确定目标动作范围、验证初相位和单 GPU 训练预算，再按需生成真实的受控 OpenFOAM 轨迹及独立测试集，不应把旧 32 条轨迹机械重跑并宣称沿用旧模型精度。当前不启动 HydroGym 训练。

后续正式训练使用 `docker/physicsnemo/Dockerfile` 构建的隔离镜像，并重新执行 Curator、DataPipe、模型训练和独立 CFD 验证。当前完成了 2.2.2 容器的 GPU/API 烟雾测试，不能替代完整训练验收。
