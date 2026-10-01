# PhysicsNeMo 双圆柱代理 5-epoch 实测（2026-10-02）

## 阶段结论

在 DGX Spark `/workspace/fluid_control`，基于 32 条真实 OpenFOAM v2512 轨迹的独立 v2 数据划分，官方 PhysicsNeMo 2.2.2 FNO + DataPipe 已完成 GPU0 训练、保存 epoch 5 检查点，并在 4 条从未训练的动作日程上完成 1/10/50 步自回归评估与动作输入消融。所有自动阶段门禁通过（`PHYSICSNEMO_SPARK_5EPOCH_OK`、`PHYSICSNEMO_SPARK_HELDOUT_OK`、`PHYSICSNEMO_SPARK_ACTION_ABLATIONS_OK`），没有 OOM 或非有限值。它证明真实 AI-CFD 案例跑通，不证明控制器有效。

## 物理与软件范围

- Re=100，前后两圆柱中心距 L/D=5；后柱旋转 `|omega|<=5`，同一 `t=80` CFD 重启场，`dt=0.005`，`t=80..160`，每条 801 帧。OpenFOAM 粗网格约 19,290 单元；尚无网格收敛证明。
- 24/4/4 条训练/验证/测试动作日程，无跨划分完整轨迹重复。测试仅代表固定 Re、间距、初始场、网格下的未见动作日程；新增验证/测试的动作 RMS 稍超训练上限，详见 `INDEPENDENT_SPLIT_REPAIR_20261002.md`。
- 模型为官方 PhysicsNeMo FNO 类，容器 `fluid-control-physicsnemo:2.2.2`、`--network none`、GPU0。数据由官方 Curator 路径整理并经官方 DataPipe 装载。学习目标为状态增量与后柱 Cd/Cl；这是监督式动作条件代理，不是强化学习或闭环控制。
- 5 个 epoch、batch 64；此为短程工程/泛化试训，非最终充分训练。检查点仅保留在远程 `artifacts/tandem_fno_expanded_spark_5epoch/best/`，没有将约 0.5 GiB 的二进制权重提交 GitLab。

## 数值结果

| 评估跨度 | 流场 MAE | 保持场基线 MAE | 相对改善 | 后柱力 MAE | 保持力基线 MAE | 相对改善 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 步 | 0.004085 | 0.008973 | 54.48% | 0.1281 | 0.09986 | **-28.31%** |
| 10 步 | 0.03682 | 0.08478 | 56.57% | 0.2016 | 0.9677 | 79.17% |
| 50 步 | 0.1289 | 0.1679 | 23.23% | 0.8648 | 1.9817 | 56.36% |

各跨度均对 4 条测试工况取等权平均、总计 64 个片段，`segment_stride=50`；全部片段数值稳定。流场误差单位与 CFD 状态（无量纲 u、v、p）一致，力为无量纲 Cd/Cl 的绝对误差。1 步后柱力比保持基线差，50 步流场优势收窄；因此不宜直接用当前代理做安全关键闭环决策。5 个 epoch 的验证集流场 MAE 由 0.00610 降至 0.003858，力归一化 MAE 由 0.1431 降至 0.06275；这些不是测试集指标。

动作消融相对“使用真实观测动作输入”的误差变化：

| 动作替换 | 1 步流场/力 | 10 步流场/力 | 50 步流场/力 |
| --- | ---: | ---: | ---: |
| 置零 | +2.12% / +138.70% | +1.62% / +138.45% | -0.46% / +79.60% |
| 反号 | +16.83% / +366.90% | +15.44% / +331.82% | +13.56% / +149.81% |

这说明模型的力预测明显依赖动作输入；但 50 步置零动作的流场误差反而略低于真实动作，反映长期场预测仍不充分。消融是在固定真实轨迹上替换模型输入，其结果不能作为反事实 OpenFOAM 控制收益或因果效应估计。

## 安全性与可核验产物

训练的 5 秒采样保护进程记录最小主机 `MemAvailable=98.305 GiB`，高于要求的 20 GiB；CUDA 进程分配上限设为设备总内存 20%（约 24.34 GiB）。DGX Spark GPU/CPU 使用统一内存，`nvidia-smi` 不给可靠的独立显存数字，因此用主机可用内存与 PyTorch 上限共同约束。磁盘在运行后仍剩约 672 GiB。

本目录下已提交可核验的小型 JSON：`expanded_fno_5epoch_training_history.json`、`expanded_fno_5epoch_heldout_evaluation.json`、`expanded_fno_5epoch_action_zero.json`、`expanded_fno_5epoch_action_sign_flip.json`、`expanded_fno_5epoch_action_sensitivity.json`。原始训练/评估日志和检查点留在远程 `artifacts/`。完整原始 CFD/HDF5 不提交 GitLab，但保留在远程项目目录。

## 下一步判定

当前首先需要改善 1 步力误差和 50 步场误差，至少完成多随机种子/较长训练的真实留出验证，并对粗网格结果做数值收敛性评估。后续可把经过验证的代理接入 HydroGym 风格的动作/观测/奖励接口做候选策略训练；必须以独立 OpenFOAM 闭环重放验证阻力、升力波动、控制能耗和稳定性。官方 HydroGym 单圆柱 PPO 基线已另行记录，不能直接当作此双圆柱闭环成果。
