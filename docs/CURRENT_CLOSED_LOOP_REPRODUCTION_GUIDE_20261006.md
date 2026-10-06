# 当前串联双圆柱闭环复现指南（2026-10-06）

## 时效说明（2026-10-07）

下面的命令保留为原投影控制链的历史复现入口，**不是后来 canonical 坐标控制器的新启动命令**。最新实现与验收边界见 [当前交付证据补充](CURRENT_DELIVERY_COMPLETION_MATRIX_20261007.md)：[E082 canonical 训练](P064_B_SYMMETRY_CANONICAL_PPO_TERMINAL_REVIEW_20261007.md)、[E083 b00 真 CFD](P064_B_SYMMETRY_CANONICAL_CFD_TERMINAL_REVIEW_20261007.md)、[E084 第二指定 seed 训练](P064_B_SYMMETRY_CANONICAL_SEED20261006_PPO_TERMINAL_REVIEW_20261007.md)、[E086 同协议 b00](P064_B_SYMMETRY_CANONICAL_SEED20261006_CFD_TERMINAL_REVIEW_20261007.md)，其报告绑定各自实际不可变源码和批准文件。两个指定 seed 主统计窗均通过原物理标准，但部分早期窗口失败；完整预测精度仍 FAIL，不能称全目标完成。

术语澄清：P064 微调的是气动力 FNO 分支的 28 个参数张量（两 bias 冻结），不是仅末层 readout；独立 flow FNO 冻结。用户所需官方组件、RL 与真实 CFD 在线反馈的基本链已实现；部署在线 FNO/MPC 是可选后续方法，不是原基本闭环必须补做的步骤。E089 长 b02 采集、E090 CPU 转换及另批的 train-view 整理均已独审完成，后续训练仍需另批。以下历史命令未经改写，不冒充上述新实验的复现命令。

## 结论先行

当前固定工况案例已经验证了一条可追溯核心链：`真实 CFD 数据 → 官方 PhysicsNeMo FNO → HydroGym 接口中的冻结 FNO 环境 → Stable-Baselines3 PPO → 冻结 PPO 的真实 OpenFOAM 配对反馈`。范围严格限定为 `Re=100`、`L/D=5`、前圆柱固定、后圆柱旋转。

从已经冻结的数据和 K1 checkpoint 开始，这条链已有三个独立、可执行且有实际终态证据的入口；但这些入口尚未在新输出上做过一次由本指南驱动的端到端复现，亦没有“一键从原始 OpenFOAM 数据重新构建全部阶段”的包装器。当前数值链不缺已知接口修复，但复现操作仍需实际验证。代理完整精度仍不满足 formal gate。PPO 学习贡献的 [FC-E078 matched control 独审](P064_INITIAL_POLICY_CFD_TERMINAL_REVIEW_20261006.md)已完成：相同 seed/b00 初相位、投影/滤波和逐值相同 paired-zero 下，初始权重主窗减阻 −0.007557%，训练后 B 为 +3.895284%；初始策略两项升力标准通过、仅减阻失败。此结果支持本次训练权重有实际贡献，不证明跨 seed 泛化、优于所有简单控制器或去掉投影仍达标，也不能抹去模型与复现限制。

实际部署阶段是 **CPU PPO 策略 + 真实 OpenFOAM 观测/动作反馈**。FNO用于 PPO 训练环境，不在部署时在线推理；部署也不是 MPC。HydroGym提供环境接口，不是 CFD 求解器；真实物理真值由 OpenFOAM `pimpleFoam` 产生。

## 组件与职责

| 组件 | 固定版本或身份 | 在本案例中的职责 | 不承担的职责 |
|---|---|---|---|
| NVIDIA PhysicsNeMo | host `.venv-curator-py312`: `nvidia-physicsnemo 2.2.2`；正式复核另用固定 b40 镜像 | 官方 FNO、HDF5 DataPipe、checkpoint 保存/加载 | 不运行最终真实 CFD 闭环 |
| PyTorch | host PPO/训练环境 `2.14.1`；b40 镜像标签 `2.13.0a0+8145d63` | FNO 与 PPO 张量运行时 | 不定义 CFD 方程；两种 runtime 不可混写成同一环境 |
| Stable-Baselines3 | `2.7.1` | PPO 训练、策略保存和 CPU 部署推理 | 不提供 FNO 或 OpenFOAM |
| Gymnasium | `1.2.3` | PPO 环境 API 依赖 | 不提供求解器 |
| HydroGym | 上游 vendored commit `4ab9854dea3d84e38a59c25e0f5835a00cf8225f` | `PDEBase`/`TransientSolver` 等环境接口；项目适配器把冻结 FNO 暴露给 PPO | 不拥有或替代 OpenFOAM |
| OpenFOAM | `2512`, build `_87ed40d2-20251219` | `pimpleFoam` 单进程真实 CFD、力系数和流场真值 | 不调用 FNO；不训练 PPO |
| 项目代码 | 下述 immutable source SHA | 数据适配、动作镜像投影、单次幅值/速率限制、配对控制/零控制编排、证据和资源保护 | 不是官方 PhysicsNeMo/HydroGym 算法 |

host `.venv-curator-py312` 与 PPO overlay 还固定了 NumPy `2.5.3`、h5py `3.16.0`、OmegaConf `2.3.1`。正式模型 CPU reload / formal evaluation 使用的 b40 镜像 ID 是 `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`，不应与 host 环境合并描述。真实 OpenFOAM 镜像 ID 为 `sha256:24205c9677d39c95221eb903988094dd7a228fc41a2054df3eaa13f80e465fcb`。

## 已冻结的前置数据与 K1 父本

本指南的可执行复现边界从已有数据和 K1 checkpoint 开始，并非从空目录生成 CFD 数据：

- 原始场和力来自真实 OpenFOAM case。完整数据设计与入口见 `docs/MATCHED_START_FULL40_CURATOR_DESIGN_20261003.md`；批处理入口是 `scripts/run_full40_curator_batch_spark.py` 和 `scripts/orchestrate_matched_start_curator.py`。
- OpenFOAM VTK 由官方 PhysicsNeMo Curator `VTKSource`/Source-Filter-Sink 管线采样，固定包为 `physicsnemo-curator==0.1.0`。最终 HDF 由项目适配器封装，但训练读取走官方 `physicsnemo.datapipes.readers.hdf5.HDF5Reader`；实际 reader 源 SHA 是 `cafa65d615555e1e4b1d6cb58895983682aae826957765c105142b71e934caa0`。项目 DataPipe 入口是 `src/fluid_control/tandem_datapipe.py`。
- P064-B 的直接 parent 不是随机初始化，而是 `artifacts/fcp026_history_training_k1_20261005/candidate/dual_model_manifest.json`。K1 批准、协议和终态来源见 `artifacts/fcp026_history_training_k1_20261005/execution_approval.json`、`candidate/training_protocol.json` 和 `docs/FC_P026_K1_TERMINAL_REVIEW_20261006.md`。
- B 还使用已冻结的 controlled-data view `artifacts/b00_controlled_train_dataset_view_20261006`，其来源是既有真实闭环场的受审 Curator/HDF 转换，不是本次 FNO 训练在线生成的数据。

若目标真的是“从原始 case 全部重建”，必须先按这些文档重新执行 OpenFOAM 获取、VTK 导出、Curator 转换、manifest/normalization finalization 和 K1 parent 训练；本文没有审计出一个覆盖这些前置阶段的一键入口，也不把后三阶段入口称为全链从零复现。

## 从已有数据和 K1 checkpoint 开始的三个执行阶段

以下命令都应在 DGX Spark 仓库根目录执行：

```bash
cd /workspace/fluid_control
```

### 1. P064-B FNO 候选训练

用途：保持 K1 flow 权重不变，以相同小预算协议训练 aerodynamic readout。它不是全 flow 重训练，也没有取得完整 surrogate admission。

- 实际入口：`artifacts/fcp064_training_source_20261006_immutable/scripts/train_fcp064_controlled_aero_ab.py`
- 入口 SHA：`8066f4a1e092c566e1b84f706ba56737afa06e13998446fee2f79dc2590920dd`
- source manifest SHA：`05c00539ad03cc9059dc101cf9c9cab47d523a8b5bdae29b4af1cea88c66a9c0`
- 实际 unit / invocation：`fluid-control-fcp064-aero-arm-b-20261006.service` / `450ef57c25c14ec38e722cbd597ffb50`
- 实际批准：`docs/FC_P064_ARM_B_TRAINING_APPROVAL_20261006.json`, SHA `a1e79d108f5067027742f08f3e04b2d73cb059286e2bd433e53f4e0d51247b29`
- 终态 result：`artifacts/fcp064_controlled_aero_arm_b_20261006/result.json`, SHA `9167e8d811f64cf001cc87bfd45d9ed2d48f5c588a19b951f7be2c826637b980`
- dual manifest：`artifacts/fcp064_controlled_aero_arm_b_20261006/dual_model_manifest.json`, SHA `92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891`
- 独审：`docs/FC_P064_ARM_B_TERMINAL_ENGINEERING_REVIEW_20261006.md`, SHA `d646e98ee086426992be5d2152b4276c551c0129a0ebfef32b3ec45f5c229104`

实际完整 argv 很长，包含三个 train-only 数据根、原始 K1 parent、固定 order、B 数据视图及所有 SHA。不要手工重录；以下命令可从已保存 unit 逐字取回作为历史执行证据：

```bash
systemctl --user cat fluid-control-fcp064-aero-arm-b-20261006.service
systemctl --user show fluid-control-fcp064-aero-arm-b-20261006.service \
  -p InvocationID -p ExecMainStatus -p ActiveState -p SubState
```

实际终态为 256 windows、32 optimizer steps、官方 fresh reload。旧输出目录已存在，因此不能直接重跑同一 argv；复现时必须生成新批准和新独占输出目录。

### 2. 使用冻结 FNO 训练 fresh PPO

用途：在 HydroGym 接口暴露的冻结 FNO 环境中训练 PPO。FNO 参数不进入 PPO optimizer，训练前后 FNO tensor digest 保持不变。

- supervisor：`artifacts/p064_candidate_policy_source_20261006_immutable/scripts/supervise_p064_candidate_ppo.py`
- supervisor SHA：`47e677084ec3affb868b61b6cfbda534eb1a40ac6a02b668dc8d622828354d94`
- 批准：`docs/P064_B_PPO_APPROVAL_20261006.json`, SHA `ae327fee310bad562aceef35029595d20c9a3421d5d5be3dc3b68bb82649e9fe`
- unit / invocation：`fluid-control-p064-b-ppo-32768-20261006.service` / `f613395cbf1140549dc60e7b046e0f6b`
- 终态 result：`artifacts/p064_b_diverse_h5_32768_ppo_20261006/payload/result.json`, SHA `3c70e21327baae98f682fc0982ca3c3910cf6d1902f3d62175f980fd685817b3`
- policy：`payload/ppo_final.zip`, SHA `f764463983355779efff8d1b1994cfaf560ab7274d54b014d34a1f084b4b307e`
- VecNormalize：`payload/vecnormalize.pkl`, SHA `8c07ef15bd41a8981f2ec0d241c85092b643ca740fea9f44866eecceac1197ad`
- 独审：`docs/P064_B_PPO_TERMINAL_REVIEW_20261006.md`, SHA `0cc1494286f85b930b43b1a11713ae6ac3719d8ff599cfa0780a8d9fe71b0d65`

已执行 worker argv 如下；它缺少 systemd 的 `MemoryMax`、`MemorySwapMax`、CPU/Tasks、RuntimeMax、TimeoutStop、KillMode/OOMPolicy 等外层合同，只能作为历史证据，不能作为独立的安全启动命令。直接复跑还会因独占输出而拒绝：

```bash
/workspace/fluid_control/.venv-curator-py312/bin/python -u \
  artifacts/p064_candidate_policy_source_20261006_immutable/scripts/supervise_p064_candidate_ppo.py \
  --approval docs/P064_B_PPO_APPROVAL_20261006.json \
  --approval-sha256 ae327fee310bad562aceef35029595d20c9a3421d5d5be3dc3b68bb82649e9fe \
  --execute
```

实际协议为 fresh initialization、32768 timesteps、H5、24 fixed resets、69 维观测、62-sample reward、seed `20261006`。这是 surrogate-based policy training，不是实际 CFD 控制收益。

### 3. 冻结 PPO 的真实 OpenFOAM 配对闭环

用途：从相同 restart 分别运行 PPO 控制与 zero control；每个控制周期读取真实 CFD 观测，计算一次 PPO action，经镜像投影和一次幅值/速率 filter 后作用于 OpenFOAM。

- driver：`artifacts/p064_candidate_policy_source_20261006_immutable/scripts/run_p064_candidate_projected_32768_ppo_long_cfd.py`
- driver SHA：`83e08d66aa6c52b4f0164b9ab41eb3a161a52d7b50fa67db078d6b65cf2cbb26`
- 批准：`docs/P064_B_PROJECTED_PPO_LONG_CFD_APPROVAL_20261006.json`, SHA `5fc8ab36e69e7e6ea27ed7c4d60ae207bc67be3c9513e89800cedccf46970a99`
- unit / invocation：`fluid-control-p064-b-projected-ppo-long-cfd-20261006.service` / `3a078c62ed9e4f7b8876f0f166bdb510`
- result：`artifacts/p064_b_projected_ppo_long_cfd_20261006/result.json`, SHA `8b31091d5e69edfbfd5ea78ba99dd7709623e6eeb0bd4f13c54e984b7fc28907`
- 独审：`docs/P064_B_PROJECTED_PPO_LONG_CFD_TERMINAL_REVIEW_20261006.md`, SHA `7b453d9c529d9d5c52988608c89d61050204510d41549cbb2be620fcdcebbe04`

已执行 worker argv 如下；同样必须由实际 unit 中的 8 GiB/no-swap/CPU/超时/清理合同包裹，不能把下列 Python 命令视为完整启动命令：

```bash
CUDA_VISIBLE_DEVICES= \
PYTHONPATH=/workspace/fluid_control/.runtime/exploratory-h5-ppo-py312 \
/workspace/fluid_control/.venv-curator-py312/bin/python -u \
  artifacts/p064_candidate_policy_source_20261006_immutable/scripts/run_p064_candidate_projected_32768_ppo_long_cfd.py \
  --spec docs/P064_B_PROJECTED_PPO_LONG_CFD_APPROVAL_20261006.json \
  --spec-sha256 5fc8ab36e69e7e6ea27ed7c4d60ae207bc67be3c9513e89800cedccf46970a99 \
  --execute
```

实际完成 800 个控制周期（80 D/U）。主窗口 `(168,228]` 的配对结果为：总阻力降低 `3.895283883%`，后圆柱升力波动 RMS ratio `0.8156230434`，平均升力偏置 ratio `0.0113781469`。这三个主物理条件满足原固定阈值；早期首 6.2 D/U 的偏置条件仍失败，不能写成所有窗口通过。

结果中 `fno_inference=false`、`mpc_action_selection=false`。也就是说，部署闭环不是在线 FNO 或 MPC。b01、b07 已用同一冻结策略完成另外两个相位的 800 周期复验，可作为已完成证据，不是本指南要求重跑的前置步骤。

## 无计算的证据复核

下列命令只核身份与已有终态，不读取大模型、不启动 GPU、不重跑 CFD：

```bash
sha256sum \
  artifacts/fcp064_controlled_aero_arm_b_20261006/result.json \
  artifacts/fcp064_controlled_aero_arm_b_20261006/dual_model_manifest.json \
  artifacts/p064_b_diverse_h5_32768_ppo_20261006/payload/result.json \
  artifacts/p064_b_diverse_h5_32768_ppo_20261006/payload/ppo_final.zip \
  artifacts/p064_b_diverse_h5_32768_ppo_20261006/payload/vecnormalize.pkl \
  artifacts/p064_b_projected_ppo_long_cfd_20261006/result.json

systemctl --user show \
  fluid-control-fcp064-aero-arm-b-20261006.service \
  fluid-control-p064-b-ppo-32768-20261006.service \
  fluid-control-p064-b-projected-ppo-long-cfd-20261006.service \
  -p Id -p InvocationID -p ExecMainStatus -p ActiveState -p SubState
```

预期六个 artifact SHA 依次为：

```text
9167e8d811f64cf001cc87bfd45d9ed2d48f5c588a19b951f7be2c826637b980
92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891
3c70e21327baae98f682fc0982ca3c3910cf6d1902f3d62175f980fd685817b3
f764463983355779efff8d1b1994cfaf560ab7274d54b014d34a1f084b4b307e
8c07ef15bd41a8981f2ec0d241c85092b643ca740fea9f44866eecceac1197ad
8b31091d5e69edfbfd5ea78ba99dd7709623e6eeb0bd4f13c54e984b7fc28907
```

## 从已有数据与 K1 父本复现后三阶段的正确顺序

历史批准和历史输出是证据，不是可覆盖的工作目录。复现后三阶段时应保持所有数值字段不变，只做以下机械工作：

1. 核对上述库版本、immutable source、既有数据 manifest、normalization、K1 checkpoint、restart tree 与镜像 SHA。
2. 为 FNO 阶段生成一个新批准，唯一改变新 unit 名、新 invocation 记录和不存在的新输出目录；运行后独立核 result/manifest/official reload。
3. 用新 FNO manifest 和真实终态 SHA 生成新 PPO 批准；fresh-init 训练，不从旧 PPO checkpoint 续训。
4. 用新 policy、VecNormalize、PPO result 和独审 receipt 生成新 CFD 批准；保持 800 周期、投影、单 filter、资源约束和 paired-zero 不变。
5. 独立重算原始四力序列和六个窗口；不要用训练 loss、surrogate reward 或单个终端 Cd 代替真实 CFD 判定。

这三个阶段不能安全地用当前历史批准直接串起来，因为每个 consumer 都核上游真实终态 SHA，而且所有输出必须独占。这个限制是可恢复性设计，不是功能缺失。

## 实际缺口与最小修复建议

固定工况核心链已有独立终态证据；已有数据、训练和 CFD 不需要为了写本指南而重跑，也没有已知的数值接口阻断。复现方面仍有两个实际缺口：一是尚未在全新输出上验证本指南驱动的后三阶段复现；二是没有覆盖原始 OpenFOAM 获取、Curator 转换和 K1 parent 训练的总入口。学习贡献 matched 对照已有 [FC-E078 终态报告](P064_INITIAL_POLICY_CFD_TERMINAL_REVIEW_20261006.md)，其限定结论不代表上述复现缺口或代理 formal 精度不足已解决；不宣称完整原目标完成。

若需要更方便地复现后三阶段，可增加一个薄 orchestration 命令，但这不是当前交付的必要新框架。它必须满足：

- 复用上述三个 immutable entrypoint，不复制训练、PPO 或 CFD 数值代码；
- 默认只做 dry-run 和 hash/preflight，显式 `--execute` 才能启动；
- 每阶段生成新批准、新 unit 和新独占输出，不覆盖历史证据；
- 只有上游 exit0、result/manifest/policy/Vec SHA 和独审 receipt 完整时才允许下一阶段；
- 沿用现有 MemAvailable、no-swap、cgroup 和 cleanup 约束；
- 不把它描述为新的科学实验、模型改进或在线 FNO/MPC。

在没有这个薄编排器时，本指南加上 `systemctl --user cat` 保存的实际 unit 合同足以明确各阶段职责和入口，但其可复现性仍需在新批准、新输出上实际检验。无需为当前固定案例的操作交付扩展到新 Reynolds 数、移动几何、新架构或重新训练已拒绝的 H25 候选。

## 边界声明

本案例证明的是固定 Re100/L-D5 数值工况下、有限相位的真实 CFD 反馈收益。它不证明跨几何/跨 Reynolds 泛化、统计独立性、实验室硬件实时性或净能耗收益。P064-B 完整 surrogate formal gate 仍为 FAIL；这限制模型作为通用预测器的采用，但不抹去已经独立复核的固定工况真实闭环结果，也不应被提升为所有用户交付的前置条件。
