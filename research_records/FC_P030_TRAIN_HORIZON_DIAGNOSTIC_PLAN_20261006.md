# FC-P030 train-only horizon diagnostic plan（2026-10-06）

## 状态

本文件只定义一个可独立审查的 CPU/source/metadata 协议草案。它不批准实现、HDF/model payload 读取、GPU 执行、优化器、checkpoint、validation/frozen 访问或 PPO。P029 完整 formal 的终态仍优先；本诊断不会改变现有门槛，也不会产生新的质量阈值。

要检验的唯一假设沿用 `docs/FC_P029_NEXT_HYPOTHESIS_REVIEW_20261006.md`：P029 只接受 H10 flow 目标，短期 train44 收益可能在同一训练轨迹的 H25--H100 内反转。该面板只能支持或削弱这一解释，不能证明长时域因果关系。

## 固定面板：44 条轨迹各一个机械选定窗口

使用 P029 原训练集合的 44 条 train-only 轨迹，但每条只选择一个 H100 窗口。选择规则在查看任何预测指标之前固定为：

1. 按 `TandemRolloutDataset` 的原始 composed-dataset 顺序保留 base20、train8、train16，各 family 内使用路径排序；
2. 对每条轨迹选择该 dataset 枚举的第一个合法 H100 窗口，即最小 `step/start=0`；
3. 不采用随机 sampler 中“第一次被抽到”的窗口，因为那会把固定 seed 的 shuffle 位置误作物理选择规则；但所选 `(case,start=0,dataset_index)` 必须确实属于原 1368 个训练窗口清单；
4. 在执行前生成并绑定 44 行 selection manifest：`family,dataset_index,case,start=0,rollout_steps=100,canonical_phase,action_profile,source_manifest_sha256,hdf_sha256`。其 SHA 必须在任何模型 forward 前固定，且两模型逐行相同。

这给出 base/train8/train16=`20/8/16` 个窗口。按既有 source-phase mapping，phase 覆盖为 b00/b02 各 15 个、b04/b06 各 7 个：base 每相位 5 个静态动作，train8 每相位 PRBS/multisine 各一个，train16 仅 b00/b02 各 8 条历史 exploratory-PPO episode。不得把 train16 描述为四相位或最终策略 on-policy 数据。

## 首窗合法性与时间语义

无需读取 HDF payload 即可由已绑定 metadata 确认帧数：base20 每轨迹 801 帧、train8 每轨迹 201 帧、train16 每轨迹 129 帧。`TandemRolloutDataset(..., rollout_steps=100)` 的索引规则是 `range(0, count-100, stride)`，所以三类轨迹的 `start=0` 均合法，并返回：

- 初态 `q_0`；
- 目标场 `q_1 ... q_100`；
- 动作/角速度样本 `omega_0 ... omega_100`；
- 与 transition 对齐的目标受力 `F_1 ... F_100`。

实际执行若获批，loader 必须逐行核 `split=train`、`start=0`、`rollout_steps=100`、case/family/phase、mask、101 个动作及 100 个 target 的形状与有限性。时间必须全部 finite 且严格递增，并精确复用 `scripts/diagnose_p028_short_horizon_comparison.py:108-114` 的既有规则：对 101 个时间点比较 `times` 与 `times[0] + 0.1*np.arange(101)`，要求最大绝对残差不超过 `max(2*max(spacing(abs(times).astype(float32))), 1e-7)`，同时记录每条 case 实际使用的 tolerance；不能改成逐步 `diff` 容差而隐藏累积漂移。绝对起始时间只作为 case 身份，不得跨轨迹据其数值推断或比较 shedding phase，phase 必须来自已绑定 source-phase mapping。任何 action/target 错位、manifest/HDF SHA 不符均 fail closed。此处不预读 payload。

该选择只覆盖每条轨迹最早的 100 个 transition：base 的后 700、train8 的后 100、train16 的后 28 个 transition 不进入面板。因此它不是 1368 窗口分布的无偏样本，也不能报告“全训练集 H100 性能”。

## 唯一比较与数值语义

只比较两个冻结候选：原 K1 flow 与 P029 flow；两者都使用完全相同的冻结 K1 aerodynamic FNO、官方 predict 路径、default-TF32/high、mask、原 train-only normalization 和 recorded actions。动作输入继续按原 manifest 的 `action_scale=0.75` 归一化，101 个 stored omega 样本 `omega_0...omega_100` 不插值、不改写。不要额外运行 P028，以免把一个已有对照变成第三份推理成本或引入新的选择空间。

每个模型从真实 `q_0` 只做一次 100 步 free-AR rollout。第 `j=0...99` 步使用当前预测状态 `qhat_j`、同一 mask 和 `omega_j/omega_{j+1}`；aerodynamic 输出与 `F_{j+1}` 比较，flow residual 更新为 `qhat_{j+1}` 并与 `q_{j+1}` 比较。这与 P029 训练时序一致，但没有 backward。

主报告使用与 `scripts/evaluate_tandem_fno.py` 相同的 **at-lead endpoint** 语义，而不是把前缀内所有时刻混在一起。对 `h in {1,10,25,50,100}`，只比较连续 rollout 的 `qhat_h` 与 `q_h`，以及第 `h-1` 次调用得到的 force 与 `F_h`。H1 只表示由真实 `q_0` 出发的第一个 free-AR lead；它不是 100 个 teacher-forced H1 样本的序列，也不得在 H10/H25/H50 处重置为真值后拼接。

每个 at-lead 必须保存足以独立重算的统计量：

- 输入 state 保持原 train-only normalization（SHA `f1b460...92bc1`）；计算指标前，预测和目标场都用该同一份固定 `state_std/state_mean` 还原到物理 `u/v/p`，并使用与 official evaluator 相同的 binary mask。令 `physical_error=(qhat_h-q_h)*state_std`，对每个 fluid-cell mask，调用/逐值复现 `evaluate_tandem_fno.field_error_sums(physical_error, physical_target, mask)`，累加每通道 error SSE 与 reference SSE；
- 调用/逐值复现 `evaluate_tandem_fno.relative_field_metrics`：每通道 relative-L2 为 `sqrt(sum_error_SSE/sum_reference_SSE)`，velocity relative-L2 为 `sqrt((u_error_SSE+v_error_SSE)/(u_reference_SSE+v_reference_SSE))`。任何 reference denominator 为零时返回 `None` 并 fail closed 于解释层；不得加入 epsilon 或把 undefined 写成 0；
- aerodynamic 输出先用同一份原 train-only `force_std/force_mean` 还原成四通道物理系数，再报告四通道 absolute error、rear-Cl absolute error 和 `abs((frontCd+rearCd)_pred-(frontCd+rearCd)_truth)`。跨 case/family/phase 聚合是物理 absolute error 的算术均值，并明确 count；不得把 normalized channel error 相加冒充 physical total-Cd；
- 保留每 case 的 at-lead 原始行，并按 family、canonical phase、action profile 汇总 P029-minus-K1 paired delta、`<0/=0/>0` 符号计数及有限值 count。场的 pooled relative-L2 必须先合并 SSE 再开方，不能平均 case-wise ratio。

可另报明确命名为 `cumulative_prefix_1_to_H` 的辅助指标：使用同一次 rollout 已保存的 `qhat_1...qhat_H` 与 force_1...force_H，把 case 和 lead 两维的 field error/reference SSE 合并后调用同一 `relative_field_metrics`，物理 force MAE 则在全部 case×lead absolute errors 上取算术均值。它不能替代主 at-lead 指标，也不能与 formal H100 数值直接比较。

一次 H100 forward 应保存 100 个 predicted-state endpoints、100 个与 P029 时序相同的 force outputs，或等价的逐 lead 充分统计量；因此 at-lead 和 optional prefix 都不得触发额外 model forward。

不把任意百分比改善设为新 gate，不选择“最好 horizon”，不依据本面板调 loss、LR、clip 或早停。由于 K1/P029 在 lead1 的 aerodynamic 输入都是相同真实 `q_0`、同一 mask、`omega_0/omega_1`，且冻结 aero 字节相同，at-lead H1 force 必须在同一 native protocol 下逐值相同；H1 field 可以因 flow 不同而变化。若这一身份关系不成立则整个结果无效。

## 覆盖与预算

面板总计 `44 windows x 100 transitions x 2 flows = 8,800` 次 flow transition；若每步都计算一次 frozen-aero force，则另有 8,800 次 aerodynamic state evaluation。理论总 forward 调用为 17,600，不包含 loader I/O，且必须在结果中标为按循环合同推导的数量，不冒充硬件 profiler counter。

与全 1368 窗口的两模型 H100 扫描相比，flow transition 从 273,600 降到 8,800，约缩小 31.1 倍。实际显存、时间和 batch 可行性仍未知；任何执行前必须另做代码审查和资源批准，不能从该算术预算推定 GPU 可运行。实现应单 window staging、no-grad、无optimizer/save，复用既有双 20 GiB 守卫与官方镜像，但这些属于后续审批范围。

## 预声明解释

本面板不产生三分类 gate。只根据完整的 at-lead paired delta、符号计数和 family/phase/action 分解写 **provisional consistency**、**provisional inconsistency** 或 **inconclusive**，不得用未预声明的“多数”“稳定”或百分比阈值把描述升级为判定：

- 如果同一 44 行面板中 H10 的 P029-minus-K1 与 H25/H50/H100 的方向发生清楚的 lead-dependent 改变，并且场与至少一个物理力指标的 case-level delta/符号表显示该改变不只来自一个 family，记录为“与 horizon-exposure 假设一致的暂时证据”，同时逐项列出例外；这仍不是因果证明。
- 如果 P029 在 start0 train-only 的 H100 paired deltas 没有显示相对于 K1 的长时域退化，而 validation b01/b05 H100 仍更差，记录为“与训练内 horizon 反转不一致；held-out phase/trajectory coverage 更值得后续检验”。这也不自动证明覆盖是唯一原因。
- 如果方向依 family/phase/action 或 at-lead/prefix 定义而变，或 start0 H10 与旧 origin51 H10 方向不同，结论必须是 inconclusive，并明确 origin sensitivity；不得据此扩大训练或挑选更有利的汇总。

若场在 H100 保持而力单独恶化，可把 frozen-aero distribution mismatch 作为后续候选解释；若场和力一起反转，则 readout-only 解释被进一步削弱。无论哪种结果，本诊断都不自动批准 H100 fine-tune、新 CFD、MPC、PPO 或门槛变更。

## 与既有 H10 结果的关系

已有 P027/P028/P029 H10 比较使用每条轨迹 `origin=51`；本计划固定 `start=0`。两者的初态、动作历史和相位位置不同，旧 H10 数值不能直接填入本面板，也不能要求逐值复现。新面板必须同时重新计算 K1 与 P029 的 start0 H10，旧 origin51 结果仅作为 origin-sensitivity 的外部背景。

该计划也不重复 FC-E003--E005：那些 H100 训练的 parent、数据、epoch 或 loss 不同，不能成为 P029 同 parent、同 171-update H10-vs-long-horizon 的 matched 因果对照。

## 来源绑定（执行前必须重核）

- `src/fluid_control/tandem_datapipe.py` SHA-256 `c939e4553dbef9e227b6a3a4d5f36242114a690b32ff907339b5be2a4ec693ae`；
- base manifest `5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2`；
- base train split `1eaa84f12ebd56e2fc9c4da3b51fa6392e94276fa44f6d2539db66c325296b89`；
- train8 manifest `a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35`；
- train16 manifest `7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b`；
- normalization SHA-256 `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1`；
- P029 objective/runner SHA-256 `904fc903b35b6a754faf1243217b3a7f4a5f552afe90e1e8e3a33654d4524516` / `e7dd8b9a261b4a47dbfcec6ce4be792dce0294454460b9924e81c37fb87c4aa0`。

实现若获批还须绑定 K1/P029 两个实际 flow checkpoint pair、共同 frozen-aero pair、完整 44-HDF inventory、source-phase mapping、config 和 formal terminal identities；本草案不制造这些未来批准字段。
