# PROJECT_STATE — 串联双圆柱主动流动控制

最后重建：2026-10-04 13:40 UTC（北京时间 21:40）；以实际文件/日志为准。本文是科学状态，不是实时资源看板。

**当前优先级（2026-10-05）**：FC-P009的cache-only提取、固定`alpha=0`四相留出分析及固定50/50共享头CPU诊断均已完成。原completion/result/cache/cross-domain SHA为`0893ec75…c214f`/`1321c30a…91daf`/`fc1b84fd…8ca84`/`ffd48eba…7516`，共享头结果SHA为`931fcd2d…f2b0bc`；代码tensor前后不变，未保存候选、未访问validation/frozen、未运行PPO。136800个window-step row只对应19648个唯一CFD端点。专用头显示强domain tradeoff；固定共享头则在两个域都优于未校准C亲本，但不支配两个专用头：H100 AR域rear-Cd/rear-Cl MAE为`0.01612/0.04610`，H1域为`0.01593/0.03035`。这只支持一个折中受力头具有train-only可表达性，不是科学准入。Lead仅批准下一步实现和CPU测试：用已缓存系数构建独立epoch0候选，严格限制为四个force rows/bias、default-TF32/high、官方保存/重载及固定train-first H1 wiring sanity；尚未批准GPU构建、formal或PPO。项目最终目标仍是合格代理、兼容控制器和真实CFD闭环共同通过。

当前科学状态（2026-10-05）：FC-P003C及其两项128/64 train-only校准均已完成并被拒绝；代理PPO仍未授权。固定C特征仿射受力读出v1在默认TF32数值wiring检查中fail-closed；独立数值probe确认这是TF32非结合运算顺序差异，不是数据错位。经单独批准的v2仅在该诊断内禁用TF32并使用highest FP32，在不改`2e-5`容差下完成（result/cache/completion SHA：`44920594…7654d`/`947309d2…f23ab`/`501a0544…fdabc`）：wiring误差`3.5763e-7`、设计矩阵满秩129但保留条件数`1.9016e5`。prefix rear-Cd/rear-Cl action MAE从`0.04408/0.11869`降至`0.00475/0.00733`，而late rear-Cd反而上升10.15%、rear-Cl仍为`0.08337`。后续纯CPU、prefix-only相位留出ridge诊断选定`alpha=1e-6`，相对alpha0将held-phase MSE降44.1%；late rear-Cd/rear-Cl MAE降至`0.03876/0.06274`，但仍较高。这支持病态/高方差是重要贡献，不证明唯一根因或准入；不能与default-TF32正式C/D015直接比改善，也不改变field/window门槛或PPO阻断。

FC-P008 train-only候选及原formal suite均已完成；恢复代次`fluid-control-fcp008-posteval-r3b-20261005.service` exit 0，总receipt SHA为`14fd24d9…edcb5`，其中19项文件SHA独立重算无差异，frozen未访问、PPO未启动。r2在完整validation10之后因host复核checkpoint路径视图不一致退出，r3只在CPU receipt identity检查退出；两次失败均保留，r3b没有重跑validation GPU。科学总体FAIL：validation10 pooled H100 total-Cd NRMSE=`0.01664246`且start0 delta-Cd MAE=`0.04284067>0.023`；dynamic6全rolling H100 pooled Cd NRMSE=`0.03056368`，与development的六条start0/window-derived endpoint pooled值`0.01386137`不是同一聚合，后者delta-Cd MAE仍为`0.03346293>0.023`。force-window仅`b01_zero`联合通过（1/6）；`b05_zero`虽Cd和Cl′ RMS通过，却因rear mean-Cl误差`0.044486>0.029387`失败。

与FC-P003C同协议相比，P008 validation H100 rear-Cl MAE只从`0.05625532`微降至`0.05500051`，rear-Cd却从`0.01008058`恶化至`0.02760489`；dynamic H100 rear-Cl从`0.13225451`降至`0.11231363`，rear-Cd从`0.04211036`恶化至`0.04871456`。四个旋转分支window rear-Cl′ RMS误差均下降，均值从`0.17577385`降至`0.08756985`（-50.18%），但mean-Cl/Cd错误使五个分支失败，不能写成整体成功。validation和dynamic各H的velocity/u/v/p指标均与C逐值相同，验证force-row confinement但不是field改善。FC-P008不满足代理准入，PPO继续阻断。

候选completion/result/model/state SHA分别为`90ab8a4d…a278d`/`cbfbf7c3…69409`/`3f92fbf5…29504`/`5cfd6bc1…644f7`。独立从真实NPZ重算六个alpha的四折OOF（最大差`1.34e-17`）并确认预定规则选择`alpha=0`，系数最大差`1.86e-12`；physical OOF MAE为front-Cd/front-Cl/rear-Cd/rear-Cl=`0.000294/0.001742/0.012857/0.022035`。全train native MAE由亲本`0.000582/0.004991/0.025576/0.064053`降至候选`0.000234/0.001459/0.010462/0.017277`。default-TF32下ideal/native仍有系统差，rear-Cd physical bias为`-0.004621`，不能以ideal fit代替native评估。官方2.2.2 CPU lineage与force-row confinement重跑PASS，optimizer=0、候选构建未访问validation/frozen/PPO。

## 1. 不变的目标

针对两个固定串联圆柱，通过后圆柱旋转，在降低两圆柱总平均阻力的同时限制后圆柱升力波动和平均偏置。完成真实 CFD → 官方神经算子代理 → 控制器 → 真实 CFD 在线反馈的可验证流程。

当前验收场景固定为 Re=100、L/D=5、后圆柱旋转；不是跨 Re 泛化研究，也不是涡激振动或硬件控制实验。跨工况稳健性是后续扩展，不能写成已完成。

与港理工相关研究的关系：继承串联圆柱自旋流动控制方向，采用独立OpenFOAM/代理实现。现有原文核查仍有未确认参数，且项目主目标为整体减阻并约束升力，不能称为对原论文奖励和全部设定的逐项复现。参考`docs/POLYU_ZHAO_2024_SOURCE_AUDIT_20261003.md`；旧`docs/PAPER_REPRODUCTION.md`包含较早阶段数据/控制描述，不作为当前运行事实或未核实文献参数的依据。

最终真实 CFD 配对验收不变：

| 量 | 要求 |
|---|---|
| 两圆柱总平均 Cd | 相对相同初态无控制降低至少 2% |
| 后圆柱 Cl′ RMS | 控制/无控制 <=1.05 |
| 后圆柱平均升力偏置 | abs(mean Cl)/无控制 Cl′ RMS <=0.10 |
| 旋转动作 | abs(omega)<=0.75，每 0.1 D/U 的变化 <=0.1 |
| CFD 评估 | 配对 80 D/U；舍弃前20，统计最后60 D/U |

omega 为本案例配置的角速度；无量纲旋转比 alpha=omega D/(2 U∞)。在本例 D=U∞=1 的标度下 alpha=omega/2，不能将 omega 与 alpha 混用。

## 2. 实际架构，而非计划示意

```text
OpenFOAM 真实流场/力/动作记录
  → 官方 Curator 流程 + 项目校验与转换
  → HDF5 + 固定训练集归一化 + 官方 DataPipe/项目适配器
  → 官方二维 FNO：当前 ROI 流场、mask、当前/下一动作
  → 下一 ROI 流场 + 学习得到的四个受力输出
  → HydroGym 接口 + 项目 FNO/OpenFOAM 适配器
  → SB3 PPO（MPC 为待评估对照，不是已实现结果）
  → 真实 OpenFOAM 状态反馈/后圆柱旋转/下一状态
```

FNO 当前不是 FNO-3D，也不是直接编码长段历史的模型。输入6通道，输出7通道；按单步0.1 D/U递推。预测区域256×128、x/D=[8,25]、y/D=[4,11]，不是整个 CFD 域。四力来自学习输出的空间平均，不是对预测壁面压力/剪切积分。模型、受力和动作响应需要分别验证。

实际参数U∞=1、D=1、运动黏度nu=0.01。40案是同一基准极限环的8个起始相位×5个旋转动作，不是40个独立工况。train为b00/b02/b04/b06，validation为b01/b05，frozen为b03/b07。完整full40数据中的冻结10案已生成并封存；开发用dev30仅包含20训练+10验证，不物化或暴露冻结数据。两种数据视图不能混淆。16个训练配对复用4个初态及各自零动作轨迹，不能当作16个独立初态。

## 3. 已有证据与缺口

| 层次 | 实际状态 | 不应推断的结论 |
|---|---|---|
| CFD 数据 | base20训练、validation10；额外train8动态、train16历史PPO轨迹已整理；16个同初态旋转/零动作配对已核验 | 数据足够覆盖所有未来控制策略或多Re |
| CFD-only PPO | b00/b01配对真实反馈分别减阻4.2212%/4.2502%，满足既定升力要求 | 减阻由FNO带来；跨独立工况泛化已证实 |
| FNO train16 两分支 | 完整后评估已执行；动态端点阻力诊断通过，但时间窗口力学检验失败 | 端点通过等于闭环可用 |
| 最新配对统计损失 λ=0/10 | 两支FC-P001同协议后评估均已核验：validation10/Dynamic6端点通过，但force-window均仅2/6 zero分支通过，总体development FAIL；λ10对四个旋转分支Cl′ RMS误差的影响混合，均值仅降1.68% | 端点PASS等于时间窗受力通过；λ=10有效解决了控制误差 |
| FNO辅助PPO/真实闭环 | 当前新候选尚未通过进入控制训练的要求 | 完整目标已完成 |

最新两支 epoch2 内验证 terminal-total-Cd NRMSE：λ0=0.7446%，λ10=0.7921%；这不是完整动态或闭环结果，也不证明配对损失有益。训练内 paired loss 同样不能替代独立验证。

目前没有单一“全指标最佳模型”。保留同协议基线和不同候选；按字段、动作响应、力窗口和最终闭环联合评价。

## 4. 当前主阻碍

上轮 train16 候选在动态动作下不能准确预测时间窗口内的平均阻力和升力脉动。对加权分支的已有诊断显示：4个动态动作分支的主频/相位较接近CFD，但相对零动作的升力幅值变化方向错误。因此“只修相位”“简单加大网络”均没有充分依据。

已有数据支持控制相关预测不足；尚不能断言唯一根因是网络容量、归一化、ROI裁剪或工况数量。需要同协议对照实验区分。

## 5. 验证规范

- 字段：u/v/p与速度联合相对L2，H1/H10/H50/H100分别报告；压力基准、mask、ROI一致。
- 受力：前/后Cd、Cl，整体Cd，时间均值、Cl′ RMS、平均偏置；动作相对零动作的差异和排序。
- 动态：同初态、固定动作序列，幅值/相位/频率和有限时间窗口误差。短窗口频谱不代表长时间统计收敛。
- 现有端点标准与后来增加的 development 时间窗口标准分开记账。后者：每分支平均Cd误差<=同窗零动作Cd的1%；Cl′ RMS和meanCl误差分别<=同窗零动作Cl′ RMS的2.5%。这不是最终物理减阻标准的替代。
- 跨Re/跨U、涡量误差、物理一致性、统计收敛及多随机种子：未完成的项目明确标 NOT_EVALUATED，不填造数值。二维Re100中的脉动动能诊断不得无说明称为三维湍动能。

## 6. 下一项已优先批准的科学工作

FC-P001已按固定协议完成并拒绝当前λ10干预：两支均未通过力窗口。FC-P002已将主要失败定位到动态动作响应的phase/sign交互；Lead已在`docs/FC-P003_APPROVAL.md`批准唯一改变paired监督时序的FC-P003。它保持Main-e2父模型、16次paired update、λ10、数据、归一化、模型、学习率、两epoch、seed和完整后评估不变，仅把原先前16个batch集中施加的paired update均匀分布到整个epoch。不启动代理PPO。

Lead随后以`docs/FC-P003B_APPROVAL.md`明确修订了“仅当FC-P003失败后才准备dynamic8”的旧fallback等待条件：FC-P003B可以在Worker并行预检和执行，因为其依据是FC-P001/FC-P002与已完成的train-only同重启配对QC，不使用尚未知的FC-P003结果挑选方案。FC-P003B保持相同父模型、regular数据及顺序、λ10、16个interleaved update、两epoch与seed；唯一比较因素是paired监督内容。它只有8条不同动态pair，每epoch各出现两次，不能写成16条独立动态轨迹，也不能覆盖或中断FC-P003。

控制链的软件兼容层已经补齐但尚未获得科学准入：D012数值producer（commit `1a7c8f4`）从已校验证据重算canonical window/dynamic兼容门槛；候选感知PPO launcher（commit `0e05aa4`）仍默认CPU dry-run并要求全部门槛；候选策略到既有OpenFOAM反馈入口的证据适配器（commit `6af0c9e`）只做SHA绑定和路径兼容。λ0/λ10的D012结果均为canonical dynamic PASS但window仅2/6、总体FAIL，因此这些code-ready里程碑不授权PPO、不证明FNO辅助闭环，也不改变当前正在运行的FC-P003/FC-P003B训练与后评估顺序。

FC-P003 Main两epoch训练及不可变v3后评估均已完成。训练completion receipt SHA为`3783f3be698ddcfd18effa34a71380064e199de7ebaee7cf41bd4b396434f4a7`，GPU guard观测到的最低统一`MemAvailable`为62.154 GiB；best为epoch2，checkpoint SHA为`d953a7ed19abd94edd4fdf6f43c41ea5c05d84e6d0e4c297a54513275f997df0`。完整posteval receipt SHA为`7b7f55b3593b150578d65ab6c403e5a84c678a5dc6d043f702c4027334cadabb`。validation10端点与dynamic6动作诊断通过，但force-window仍只有两个zero分支通过（2/6）；四个旋转分支rear Cl′ RMS误差为0.165159/0.213692/0.178018/0.157702，与历史frontloaded λ10基本不变。因此development admission FAIL，FC-P003的均匀interleaving假设未获支持，不授权代理PPO。真实CFD反馈wall-time观测代码（commit `a4399f3`）已就绪，但它不改变动作、solver或验收门槛，且不构成候选控制执行授权。

FC-P003B的两epoch训练、不可变恢复及同协议后评估已完成。首次后评估的`/workspace/gatedata`与`/workspace/devdata`路径合同故障和原输出保留；recovery r2（commit `2331301`，runner SHA `ffa5…1978`）在复用validation10前重算SHA/有限性/计数/checkpoint。Worker→Spark transfer receipt SHA为`bc665a9a7bd468393ce2a32ceaaf1df7e8b27540ea4297f46ffb09e74d856fcc`，posteval receipt SHA为`98f3d336b79a7816f17c204bfd521c1bdbcc83eed1b5ddf5df59c04995314258`，checkpoint为`ed0da140da2b81b99da200847b57fbbd97a885dbdc030691bac3437ce8ac2e08`。validation10 pooled total-Cd NRMSE 0.00504413、dynamic endpoint通过，但force-window仍只有2/6 zero分支通过；四个旋转分支rear Cl′ RMS误差为0.160242/0.206943/0.172678/0.153707，较FC-P003分别仅降2.98%/3.16%/3.00%/2.53%，均值降2.94%，仍约为阈值的5.9倍。development admission FAIL，frozen/PPO均未访问/启动。

606个同物理端点的true-state H1诊断中，四个旋转分支rear Cl MAE为0.191916/0.156482/0.157575/0.191730，zero分支仅0.009827/0.006964。这不支持“误差只是递归积累”，但不证明唯一原因。FC-P003B的dynamic6 H100 pooled velocity/u/v/p relative-L2为8.7648%/7.1992%/23.4929%/27.2588%，与FC-P003比较也是混合变化；详见`docs/FC_P003B_FIELD_ACCURACY_AUDIT_20261005.md`。因此后续不能因force指标改善就缩窄为force-only目标，完整field报告仍为必须项。

Lead已在`docs/FC-P003C_APPROVAL.md`授权工程实现和CPU测试：保持Main-e2亲本、模型/数据/顺序/归一化/学习率/seed/两epoch/16次update/λ10/通道权重不变，只将旧9个窗口统计配对项替换为true-state每端点action-minus-zero四力误差。实现commit `5ac306a`已通过独立的旧regular objective loss/梯度精确等价、chunked-vs-monolithic、单clip/单step、nonfinite拒绝、两pass配对身份和官方PhysicsNeMo 2.2.2 CPU回归测试。该里程碑仍不包含完整GPU训练；单次混合梯度/内存probe需绑定最终SHA并获Lead单独GO，新候选仍须通过全部field/force/dynamic/window原门槛才能考虑PPO。

FC-P003C 的单步 mixed-loss 技术探针现已完成，completion/result SHA分别为`f95a6f554e20964125628a2d33a88709827daf298e935de653471aaa1708b1cd`/`614264a6626d5411df23d6994c025796c043d8c29f1e2aeb5c5b9f948a0a30ea`。它严格执行一次optimizer step、未保存candidate、未访问validation/frozen，parent前后字节一致；外层guard观测最低`MemAvailable`为94.265 GiB。该结果只证明混合真实DataLoader/梯度/一步更新可执行，不是科研PASS。

Lead随后以commit `f034b15`、审批JSON SHA `4e4c8142b9f243589b042a44ddd99d9a164e1cc0951bc5becd8ec169336766b8`批准固定两epoch正式训练。候选根为`artifacts/tandem_fno_true_state_paired_step_lambda10_20261005`。训练以exit 0完成：completion receipt SHA为`8411d422d373de1d7bad6098304bc3543c4abc677345d948ecc89333e014e3b8`，best epoch2 model/state SHA分别为`f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4`/`a63186685c998115507c423c5db8d467e845469dc0a5b087b47f341b361a8b4a`。不可变后评估也已exit 0，总receipt SHA为`0ee2468b193b25e09cf2bc1b2c71a43fc918888788db92b2f115149f21cb8338`，其SHA表重算无差异，frozen未访问、PPO未启动。

2026-10-04 21:11 UTC的epoch 1训练内selection score为0.0265716、terminal-total-Cd pooled NRMSE为0.0132990；这些数值只来自epoch内validation，不能替代固定validation10、dynamic6、force-window或development admission。辅助的16位置梯度诊断在第一个位置的regular-total梯度分解一致性检查处fail-closed；首位置debug测得相对残差`3.9183e-5`，超过未改变的`2e-5`容差，且未产生optimizer step、候选checkpoint或validation/frozen访问。该独立失败没有终止或改变主训练，不能被写成FC-P003C的科学结论，也不授权放宽容差或重试。

epoch 1的单轨迹可视化preview已由收据SHA `3c6877f45493fc341dbd6e16ae9bca7c9cdb3a31dfac9ca2ecd55bddbce4b4e9`绑定：仅使用validation的`b01_plus`、start0和H1/H10/H50/H100。velocity relative-L2依次为0.002099/0.020483/0.063993/0.092529；H100 pressure relative-L2为0.282826，单端点rear-Cl MAE为0.124421。长递推图中可见小尺度结构误差，但单条轨迹和图像不能区分相位、频谱、递推或局部映射原因，也不能代替多分支统计。该preview不参与best epoch选择、不修改formal gate、不提前授权PPO；正式结论仍等待完整posteval receipt。

FC-P003C训练完成后，独立评价按以下原始路径读取，不从训练日志推断正式结果：流场及各物理量误差位于`posteval_fc_p003c/validation10/evaluation.json`、`posteval_fc_p003c/dynamic6/evaluation.json`及相应`segments.json`；端点受力/动作诊断位于`validation10/endpoint_gate.json`和`dynamic6/diagnostic.json`；6.15 D/U力窗口和最终开发门禁位于`force_window/result.json`及`development_gate.json`。只有`posteval_fc_p003c/receipt.json`完整绑定上述文件、checkpoint、未访问frozen且未启动PPO后才进入D012兼容重算。commit `5f495c8`仅补FC-P003C专用外部step-receipt binder及candidate-readiness的严格trained-source合同，软件兼容完成不等于任何科学门槛通过。

后评估完整且经独立SHA复核之后，允许的CPU-only C→D012顺序是：先运行`derive_d012_fcp003c_step_receipts.py`，输入`posteval_fc_p003c/receipt.json`和其中lineage绑定的checkpoint SHA，在原bundle之外生成`derived_step_receipts/{dynamic6,force_window}.json`；再运行`produce_canonical_surrogate_compatibility_gates.py`，输入同一posteval的`force_window/result.json`、`dynamic6/{evaluation,segments,diagnostic}.json`、两个派生step receipt、真实dynamic6数据、physical-QC、预声明、D012协议及同一best checkpoint，输出独占的`canonical_window_gate.json`与`dynamic_action_gate.json`。两步都只重算兼容证据，不启动PPO；任一canonical或development gate失败仍保持PPO阻断。

FC-P003C的完整同协议结果现已拒绝其科学假设。validation10 H100 velocity/u/v/p relative-L2为0.043591/0.036515/0.113918/0.142884，rear-Cl MAE为0.056255；dynamic6 H100 velocity/u/v/p为0.087211/0.071377/0.235243/0.277954，rear-Cl MAE为0.132255。与FC-P003B比较，速度仅微幅改善，而pressure、force和rear-Cl是混合或变差。dynamic endpoint组件PASS（pooled start0 total-Cd NRMSE 0.005719，delta MAE 0.013126），但force-window仍只有两个zero分支通过；四个旋转分支rear Cl′ RMS误差为0.164160/0.206895/0.173657/0.158384，相对FC-P003B的均值反而增加约1.37%。development admission为FAIL，说明true-state每端点配对监督未修复关键升力窗口瓶颈，不授权PPO。CPU-only D012兼容重算已以绝对路径完整validator复核后执行：canonical window SHA `07a2ceaf4345b75032c7f54f8972ee8fab88ce888f99eef9fd69b1119c34c376`仍FAIL，dynamic SHA `551275aa225d249d0de44513415ab6af0c6076cfbc3d0fd0ec015d2b6b8ae186`为PASS，两者`ppo_authorized=false`。

D015的工程实现和CPU验证已获批准，但GPU实验仍须等待CPU验证与独立审查。该诊断必须按D015的三面板时域及动作/相位分组执行，不改gate，不把clipping或ω距离事后定为因果。

以下命令只作为complete receipt出现并独立验证后的固定CPU handoff；当前不得提前执行：

```bash
CANDIDATE=artifacts/tandem_fno_true_state_paired_step_lambda10_20261005
POSTEVAL="$CANDIDATE/posteval_fc_p003c"
COMPAT=artifacts/canonical_surrogate_protocol_completion_fc_p003c_20261005
CHECKPOINT_SHA=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' "$POSTEVAL/lineage.json")
CHECKPOINT_EPOCH=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_epoch"])' "$POSTEVAL/lineage.json")
python3 scripts/derive_d012_fcp003c_step_receipts.py \
  --posteval-receipt "$POSTEVAL/receipt.json" \
  --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --dynamic-output "$COMPAT/derived_step_receipts/dynamic6.json" \
  --force-output "$COMPAT/derived_step_receipts/force_window.json"
python3 scripts/produce_canonical_surrogate_compatibility_gates.py \
  --force-window "$POSTEVAL/force_window/result.json" \
  --force-step-receipt "$COMPAT/derived_step_receipts/force_window.json" \
  --evaluation "$POSTEVAL/dynamic6/evaluation.json" \
  --segments "$POSTEVAL/dynamic6/segments.json" \
  --dynamic-diagnostic "$POSTEVAL/dynamic6/diagnostic.json" \
  --posteval-receipt "$POSTEVAL/receipt.json" \
  --dynamic-step-receipt "$COMPAT/derived_step_receipts/dynamic6.json" \
  --dynamic-data data/curated/tandem_cylinders_full40_dynamic_validation_v1 \
  --physical-qc artifacts/tandem_cylinders/full40_dynamic_validation_real_openfoam_qc_20261003.json \
  --predeclaration artifacts/tandem_cylinders/full40_dynamic_validation_predeclared_20261003.json \
  --protocol docs/CANONICAL_SURROGATE_PROTOCOL_COMPLETION_20261005.md \
  --checkpoint-dir "$CANDIDATE/best" --checkpoint-epoch "$CHECKPOINT_EPOCH" \
  --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --window-output "$COMPAT/canonical_window_gate.json" \
  --dynamic-output "$COMPAT/dynamic_action_gate.json"
```

true-state paired-force backward工程探针是独立技术检查，不是训练实验。v1在forward前因mode-600 manifest在原容器UID/cap-drop配置下不可读而`PermissionError`退出，无optimizer、权重保存或候选模型；失败输出已保留，不对更底层的rootless/user-namespace机制作未验证归因。Lead事后明确批准了operational-only v2单次GPU技术预检（immutable launcher SHA `705c6d2f…339e`，CPU mount preflight SHA `6e4f4a0a…c489`），数值合同、数据、模型和容差不变。v2已完成：T20整段/分块loss为0.00853258837/0.00853258773，最大参数梯度绝对差1.86e-9；H100 normalized loss 0.06177457，梯度全部有限且47,210,800/47,222,711个元素非零，CUDA峰值allocated/reserved为5.886/6.537 GiB，最低`MemAvailable`105.921 GiB。模型parameters/buffers前后SHA相同，optimizer step=0，未保存candidate，未访问validation/frozen。result/completion SHA分别为`773af049…f9c8`/`eb23c661…12d4`。这只说明单个配对loss的梯度/内存工程可行，不说明模型改善或科学准入。

最新执行核查（2026-10-04 15:52 UTC）：FC-P001两支总收据均已回主节点并通过内容复核，λ0 SHA `ab90a921…c677`、λ10 SHA `03b7358d…ef4`，两者均为`DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL`。评估执行阶段完成不等于科学假设成立，更不等于项目完成；代理PPO仍被门禁。

旧工作清单只覆盖上一轮后评估的问题已修正，FC-P001 paired后评估的完成/失败结论已纳入监控；新运行代次继续接入。单纯“无告警”不作为无待办证据。初次启动为补齐实际归一化/模型/ZIP内容检查而主动终止，保留在`posteval_fc_p001_preflight_gap_v1`；v2使用5534f8d审查后的不可变启动脚本，未修改模型或验收阈值。

### D015 true-state H1训练拟合/验证分解

FC-P003C完整后评估科学FAIL后，Lead批准了固定epoch2模型的D015 no-grad诊断。v1因单元素HDF time数组显式标量转换问题在生成首行前operational fail，原失败保留；v2只修复该转换并在同一模型/数据/窗口下完成。Worker最低`MemAvailable`为112.695 GiB，未创建optimizer或candidate，未读取frozen，未执行PPO。result/worker receipt SHA分别为`b311715724287c34aa496405f0381fb034089123d5dcf3bec51f80633f293b54`/`f8de7c01baf7ab73e63d11ad9d6c11186f3587ae092c1a2edb51bf918ef6049b`。

三窗口rear-Cl absolute MAE为0.08112/0.07332/0.11646，非零action-minus-zero rear-Cl MAE为0.11880/0.10753/0.17185（train paired/train late/validation late）。因此误差并非只在离开extra-paired时间窗后出现：训练前缀本身仍有明显rear-Cl映射误差；同时validation比同索引train late更差，说明还存在分布转移缺口。现有formal dynamic6 H1四通道绝对误差的2424个值被复算到最大差`3.5763e-7`。这些是诊断证据，不改变FC-P003C gate FAIL，不授权PPO或新训练。

## 7. 证据入口与分工

- 架构：`docs/FNO_HYDROGYM_PPO_OPENFOAM_INTEGRATION_AUDIT_20261004.md`
- 真实控制：`docs/DIRECT_CFD_PPO_RESULTS_20261004.md`
- 上轮候选：`artifacts/tandem_fno_control_train16_h100_20261004/posteval_complete_v2/`、`artifacts/tandem_fno_control_train16_h100_lift_balanced_worker_20261004/posteval_worker_v1/`
- 本轮训练：`artifacts/tandem_fno_paired_stats_lambda0_20261004/`、`artifacts/tandem_fno_paired_stats_lambda10_20261004/`
- 实验定义、精确协议和指标：`EXPERIMENTS.md`、`experiments/results.csv`
- 决策与下一步：`DECISIONS.md`、`docs/RESEARCH_ROADMAP.md`
- 实时展示：commit `d8d0d71` 的目标/agent/资源/模型状态看板已通过33项测试、Ruff和JS语法检查并在Chrome刷新；它是证据展示层，不改变FC-P001协议、MPC/FNO闭环未完成状态或项目验收结论。

Lead负责目标/批准/综合证据；Physics/Data负责真实数据与参数覆盖；Surrogate负责可复现模型实验；Control/Evaluation负责控制合同与独立验收。四并发槽中控制与评价职责错峰承担，不扩张代理数量。

### D015有界train-fit校准结论

固定C epoch2亲本的128步train-only校准已完成，completion/result SHA分别为`7703e2b1b94dd64be706c64099a5355fd1c2d7dfa9280643c0ffd6cff5873bfa`/`f2397fed145f604892a18edf94d3e690efeddaa2ea2ddae71fd4fa41ca4ce423`。执行包含连续128个regular update、64个paired update和dynamic8的八轮完整遍历，未访问validation/frozen且未执行PPO。rear-Cl action-minus-zero MAE在paired/late窗口仅下降3.02%/2.57%，但absolute MAE上升4.92%/5.89%，velocity relative-L2上升6.80%/5.28%，u/v/p均退化。由已记录计数分离出的zero rear-Cl MAE从0.005973/0.006371恶化到0.025236/0.024862。因此增加相同delta-only监督曝光的假设被拒绝，不进入正式后评估或PPO。

Lead只批准了下一单因素的CPU实现：保持亲本、128/64预算、regular field loss、数据、seed、学习率、λ10和通道权重不变，仅将paired项改为对称的true-state action/zero绝对四力误差。实现commit `5cb65bb`默认仍为原delta路径，并显式记录objective；20项官方PhysicsNeMo 2.2.2 CPU测试通过。该里程碑不授权GPU或说明绝对监督有效。

该absolute单因素随后经单次批准执行并被拒绝。completion/result SHA为`20c19387802ed2297dabb903477ceea6b924171c320c50d65d5f5120a35fa8ed`/`38687ccca3594af8aff6082d57dd090c0765df1dc56cc11850e92b35d1216fdb`；128/64顺序及before readout与delta实验一致，未访问validation/frozen/PPO。paired/late窗口action rear-Cl MAE下降5.68%/5.13%，但zero rear-Cl MAE上升174.9%/145.0%，delta只下降2.19%/1.82%，u/v/p均退化。因此不进入formal validation或PPO。
