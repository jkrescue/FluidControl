# FC-P031：固定预算 H25 控制相关流场训练对照计划

状态：**仅条件设计，未授权 CPU 实现、真实数据/模型读取、GPU 探针、训练或评估。**
FC-P030 r2 已由 `docs/FC_P030_RECOVERY_TERMINAL_REVIEW_20261006.md`
（SHA256 `4e21fc05f543b5c90a74e318e9e7ae999b27b8350d7573817b49c4c6670dc5b4`）
独立复算通过；本计划不改变任何既有 development/formal/physical gate，也不授权 PPO 或 CFD。

## 证据与有限假设

FC-P030 在原 44 条 train 轨迹、每轨迹机械选择 start0、同一 recorded-action
H100 协议下比较 K1 parent 与 P029。结果文件
`artifacts/fcp030_train_horizon_diagnostic_20261006_r2/result.json`
（SHA256 `b1042b94fde60aed135c60d348431aa1c6177b1b9b12b1ae8ba56bc9fab6ee7f`）显示：

| lead | velocity relative L2, K1→P029 | rear-Cl MAE, K1→P029 | total-Cd MAE, K1→P029 |
|---:|---:|---:|---:|
| H1 | .001914209→.001852940 | .016739125→.016739125 | .003664564→.003664564 |
| H10 | .014489678→.014354783 | .023525983→.034870542 | .008205518→.009638433 |
| H25 | .023227661→.025825655 | .043777336→.060033828 | .014329363→.021341599 |
| H50 | .034159498→.039380923 | .045018195→.049584774 | .017509734→.029109003 |
| H100 | .050849721→.062902582 | .055620571→.066475976 | .015652438→.018948569 |

H25/H50/H100 的 velocity error 在 44/44 cases 都变差；H10 的 field aggregate
仅小幅改善，而两项 force metric 已变差。P029 只通过 H10 展开训练，因而一个可证伪的
解释是：固定更新预算下，H10 objective 没有约束足够长的 closed rollout，终端更新在
H25 后累积出系统性 state drift。这个观察**不是因果证明**：force readout mismatch、
train/formal distribution shift、action profile 和 gradient clipping 仍可能是主因；而且
P029 在同一 train H10 force metric 上已经变差，所以加长 horizon 很可能同样失败。

唯一待检验假设是：**在其它训练条件完全不变时，把同一 control-aware objective 的
反传展开由 H10 改为 H25，能否减轻 H25 以后 drift，并改善固定 frozen-aero 对 predicted
state 的 force readout。** 结果只适用于该 parent、数据和有限预算；失败不能证明所有
long-horizon training 无效，成功也不能绕过原 formal/CFD 验证。

## 单一干预和严格对照

FC-P031 不从 P029 terminal 继续训练，而与 FC-P029 从完全相同 parent bytes 重新开始：

- P009 flow epoch0：model
  `dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31`，
  state `4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e`；
- frozen P026 K1 aero epoch1：model
  `e2f67dbde0ab28ccd7aa46b34ee3904178c7549cd1539f4a3ae40e2bd17e67b5`，
  state `ab2fe103bca0a8c84156e2c9fd7ded5336f2d4236436b593fc414e564d7e92d3`。

保持原 44 train trajectories、1368 个 H100 start identities、sampler order
`177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f`、
seed20261003、每次顺序累积 8 windows、171 AdamW updates、lr `1e-5`、
betas(.9,.999)、eps `1e-8`、weight decay `1e-4`、clip norm1、mask、残差更新、
action/target timing、normalization、可训练 tensor scope、unused flow force-row 恢复、
一次 terminal checkpoint 和 official save/reload 全部不变。没有中间 checkpoint 选择、
早停、scheduler、重启或 extension。

**唯一数值变化**：每个已有 H100 window 只使用前 25 transitions 参与同一 objective，
而不是前 10 transitions。每步 aero 仍用当前 `qhat_(s+j)`、`omega_(s+j)`、
`omega_(s+j+1)` 预测 `F_(s+j+1)`；j=0 force term 对 flow gradient 为零，
j=1..24 通过既有 flow rollout 传梯度；`qhat_(s+25)` 只有 field supervision。
不得 detach、截断 BPTT 或输入未来 truth state/force。

Loss 保持 P029 项目自定义（不是 PhysicsNeMo 官方 recipe）：25 步平均的 normalized
field MSE 与四通道 normalized force MSE，按固定 50/50 相加。为避免“horizon+重新定标”
成为两个因素，**不重算 scales**，绑定 P029 已批准的 parent scales：

- `S_field = 0.001456146538716282`
- `S_force = 0.003364271827125755`

即 `0.5*mean25(field)/S_field + 0.5*mean25(force)/S_force`。这不会保证两项
gradient 等权；H25 改变了时间平均中的样本和 gradient direction。P029 update1/171 的
preclip norm 分别约571.68/643.14、clip scale约.00175/.00155，说明 clipping 强烈；
因此即使其它字段固定，也只能把结果归于“完整 H25 training program”，不能声称纯粹
由更长 exposure 或某一 loss 梯度导致。

## 最小实现边界（获批后）

只允许以下新/小改动，旧 frozen sources 和 P029 artifacts 永不覆盖：

1. `scripts/p029_control_aware_flow_objective.py`：把内部 rollout horizon 做成显式参数，
   默认仍精确为10；允许的 P031 值固定25。现有 P029 tests 必须逐值保持原结果。
2. 新建 `scripts/train_p031_control_aware_flow_h25.py`：从
   `train_p029_control_aware_flow.py` 复用 data/order、scope、optimizer、row/moment restore、
   official load/save/reload；protocol 明确 H25 和上述固定 P029 scales。
3. 新建 `scripts/run_p031_control_aware_flow.py` 及 focused tests：沿用已审 Docker/CID
   ownership、窄只读 train mounts、exclusive output、guard 和 immutable source closure；
   新 status/kind，不能冒充 P029。
4. 测试至少覆盖：25 个 field/force time indices、j0 force-gradient=0、j1..24 非零、
   terminal q25 只受 field supervision、无 detach、mask、frozen aero 无 param grad/version
   变化、unused rows/Adam moments 恢复、8-window sequential accumulation、171-order、
   finite guards、official save/reload 和默认 H10 backward compatibility。

不得改变 official FNO architecture/API、增加 CFD、换 split、换 parent、扫 horizon/lr/weight、
引入 activation checkpointing，或让 validation 指导训练选择。

## 独立候选身份和可评估性闭环

P031 不能以 P029 身份保存或进入旧 runner。实现必须在完整训练审批前（no-save resource
probe 可更早进行）闭合以下工程身份；这些修改只增加 exact profile，不改变数值公式：

- `src/fluid_control/dual_fno.py` 增加唯一 manifest kind
  `FC_P031_LONG_HORIZON_CONTROL_AWARE_FLOW_REPAIR`。标准 `flow`/`aerodynamic` roles 不变：
  flow 是 P009 parent 经 P031 epoch1/171 updates 的新 pair，`frozen=false`；aero 是上述
  P026 K1 epoch1 pair 的 byte-identical copy，`frozen=true`。禁止交换 roles、wrong parent、
  epoch、in-channels、config/norm、precision 或 manifest kind。
- flow checkpoint metadata 至少精确绑定：P031 status/kind、`training_experiment=FC-P031`、
  epoch1、H25、1368 windows、171 updates、8-window accumulation、sampler-order SHA、
  optimizer fields、P009 flow parent、P026 K1 aero parent、P029 fixed scales/receipt、
  no selection/validation/frozen/PPO。aero metadata 和 bytes 必须继续满足原 P026 K1 contract。
- terminal candidate auditor 使用 P031 专属 status，逐项 hash 标准7文件，核 result/protocol/
  manifest、parent、order、171 records、allowed tensor scope、aero byte identity、unused flow
  final rows/bias及其 optimizer moments 每步恢复、无 validation/frozen/PPO/admission。不得把
  result 自报的171 records 冒充独立 optimizer-state 反序列化证明。
- 新 official CPU reload receipt 必须由 pinned official loader 在 `high`/default-TF32 协议下
  fresh load 两个 roles，核 exact epoch/metadata/tensor SHA 和一条有限 forward；不创建
  optimizer、不保存模型、不访问 validation/frozen，也不授予科学结论。
- original formal runner/validator、dev30 identity overlay、evaluate/force-window callers 只增加
  P031 exact-kind profile，复用原 frozen numerical tree、阈值、horizons 和 aggregation；任何
  legacy/P026/P028/P029 path 保持原行为。完整训练完成、terminal audit和CPU reload通过、Lead
  另签 formal approval 后才可运行。
- `scripts/p030_train_horizon_core.py` 的算术不改；P030 driver/spec/launcher 只增加显式 P031
  candidate profile和结果 label，核同一 frozen K1 aero。默认 P029 profile及旧结果保持不变，
  禁止把 P031 输出写成 `p029` key 或复用 P029 approval。

上述文件的 exact source closure、tests、candidate schema 和 receipt schema 都需独立审查。
PPO/HydroGym P031 identity 支持只有 original formal PASS 后才进入另一审批，不是本训练实现的
先决条件。

## 先决资源探针与停止条件

P029 的实际 H10 no-update probe（
`artifacts/fcp029_resource_probe_20261006/payload/result.json`，SHA256
`48e356dcda21fb70b74129aa611744e693973ae09d2f6cb7929ea00d7066f161`）在固定 window816
上 peak CUDA allocated/reserved 为2,986,973,184/3,409,969,152 bytes；H10 完整训练
peak 为3,562,740,736/4,104,126,464 bytes。H25 不能据此线性宣称可运行。

在训练审批前只允许一次独立批准的 **H25 no-update full forward/backward probe**：同
window816、同 parent/scales、无 optimizer、无 step、无保存、无 validation/frozen。
现有 trainer 的内存策略保持不变：batch1；八个 windows 逐个 forward/backward 并只累积
grad，绝不同时保留八张图；一次 update 后清 grad；没有 gradient checkpointing。
探针记录 elapsed、peak allocated/reserved、每层 finite grad、aero/parent tensor before/after。

容器仍为 pinned b40、12GiB/no-extra-swap、allocator `.06`、GPU0；启动前
MemFree≥30GiB、MemAvailable≥50GiB，运行中 MemFree/MemAvailable/CUDAfree 均≥20GiB，
probe 900s、未来训练最多14400s。除20GiB floor外，还要把 P029 实测
optimizer/transient 差额（reserved peak约多0.65GiB）作为非保证性余量披露；若 H25
probe 接近 floor、OOM、超时、出现 nonfinite/byte drift，结论是资源上不确定或失败，
停止，不在同一实验中加 checkpointing、降 horizon、改 batch/lr 或盲重试。
P029 曾在表面 startup30GiB 条件下发生一次训练期 guard failure，因此 startup 数值和
no-step probe 都不是完整训练可行性的保证；探针后还须单独审查当时的实际空闲资源、
上述 optimizer/transient 余量和 launcher deadline，才能签 full-train approval。

完整训练期间任一 memory floor、非 finite loss/gradient/parameter/moment、sampler order、
frozen-aero/unused-row confinement、terminal metadata 或 official reload 失败均 fail closed；
不得根据 loss/diagnostic 自动早停或改为其它 checkpoint。

## 固定评估和决策

若资源与训练分别获得批准并完成，先运行与 FC-P030 完全相同的 train-only 44×start0
H100 sufficient-stat diagnostic，新增 P031 arm，报告 H1/H10/H25/H50/H100 at-lead 和
prefix、family/phase/action-profile paired deltas。H1 force 应因相同 frozen aero/true input
而 byte-identical；H1 field 未被假定相同。该诊断是机制证据，不是 admission gate。

为避免事后挑指标，预先计算 P031-minus-P029 的九个 pooled at-lead deltas：H25/H50/H100
各自的 velocity relative L2、rear-Cl MAE、total-Cd MAE。另完整报告相对 K1 的值及每个
family/phase/action-profile 的 delta/sign count，不允许用 subgroup 掩盖总体结果。

九项都严格为负记为**同向描述性支持**；九项都严格为正记为**不支持**；包含零或正负
混合记为**不确定/mixed**。这些标签不是新的质量阈值、admission gate 或 checkpoint
selection，不能据 subgroup 事后翻转，也不触发 horizon/loss sweep。

每个 operationally valid 的唯一 P031 terminal，无论上述标签为何，都固定生成同协议
P030 train diagnostic，并在另行完成资源/身份/执行审查后运行**原封不动**的 complete
formal suite；train panel 不得成为扣留 formal 的新 gate。Formal 保持相同 precision、
H1/10/50/100、dynamic、window、action-minus-zero 和所有既有阈值，且只评 terminal
checkpoint。任何 formal FAIL 即拒绝候选；即使 formal PASS，也只允许另行审查兼容 PPO，
再做 paired real-CFD closed-loop 原物理目标验证。FC-P031 不修改阈值、不复用 validation
选 horizon，也不声称 H25 是 long-horizon causality 的证明。
