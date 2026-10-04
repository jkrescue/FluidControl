# PROJECT_STATE — 串联双圆柱主动流动控制

最后重建：2026-10-04 13:40 UTC（北京时间 21:40）；以实际文件/日志为准。本文是科学状态，不是实时资源看板。

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

606个同物理端点的true-state H1诊断中，四个旋转分支rear Cl MAE为0.191916/0.156482/0.157575/0.191730，zero分支仅0.009827/0.006964。这不支持“误差只是递归积累”，但不证明唯一原因。Lead已在`docs/FC-P003C_APPROVAL.md`授权工程实现和CPU测试：保持Main-e2亲本、模型/数据/顺序/归一化/学习率/seed/两epoch/16次update/λ10/通道权重不变，只将旧9个窗口统计配对项替换为true-state每端点action-minus-zero四力误差。该批准不包含完整GPU训练；新候选仍须通过所有原门槛才能考虑PPO。

true-state paired-force backward工程探针是独立技术检查，不是训练实验。v1在forward前因mode-600 manifest在原容器UID/cap-drop配置下不可读而`PermissionError`退出，无optimizer、权重保存或候选模型；失败输出已保留，不对更底层的rootless/user-namespace机制作未验证归因。Lead事后明确批准了operational-only v2单次GPU技术预检（immutable launcher SHA `705c6d2f…339e`，CPU mount preflight SHA `6e4f4a0a…c489`），数值合同、数据、模型和容差不变。v2已完成：T20整段/分块loss为0.00853258837/0.00853258773，最大参数梯度绝对差1.86e-9；H100 normalized loss 0.06177457，梯度全部有限且47,210,800/47,222,711个元素非零，CUDA峰值allocated/reserved为5.886/6.537 GiB，最低`MemAvailable`105.921 GiB。模型parameters/buffers前后SHA相同，optimizer step=0，未保存candidate，未访问validation/frozen。result/completion SHA分别为`773af049…f9c8`/`eb23c661…12d4`。这只说明单个配对loss的梯度/内存工程可行，不说明模型改善或科学准入。

最新执行核查（2026-10-04 15:52 UTC）：FC-P001两支总收据均已回主节点并通过内容复核，λ0 SHA `ab90a921…c677`、λ10 SHA `03b7358d…ef4`，两者均为`DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL`。评估执行阶段完成不等于科学假设成立，更不等于项目完成；代理PPO仍被门禁。

旧工作清单只覆盖上一轮后评估的问题已修正，FC-P001 paired后评估的完成/失败结论已纳入监控；新运行代次继续接入。单纯“无告警”不作为无待办证据。初次启动为补齐实际归一化/模型/ZIP内容检查而主动终止，保留在`posteval_fc_p001_preflight_gap_v1`；v2使用5534f8d审查后的不可变启动脚本，未修改模型或验收阈值。

## 7. 证据入口与分工

- 架构：`docs/FNO_HYDROGYM_PPO_OPENFOAM_INTEGRATION_AUDIT_20261004.md`
- 真实控制：`docs/DIRECT_CFD_PPO_RESULTS_20261004.md`
- 上轮候选：`artifacts/tandem_fno_control_train16_h100_20261004/posteval_complete_v2/`、`artifacts/tandem_fno_control_train16_h100_lift_balanced_worker_20261004/posteval_worker_v1/`
- 本轮训练：`artifacts/tandem_fno_paired_stats_lambda0_20261004/`、`artifacts/tandem_fno_paired_stats_lambda10_20261004/`
- 实验定义、精确协议和指标：`EXPERIMENTS.md`、`experiments/results.csv`
- 决策与下一步：`DECISIONS.md`、`docs/RESEARCH_ROADMAP.md`
- 实时展示：commit `d8d0d71` 的目标/agent/资源/模型状态看板已通过33项测试、Ruff和JS语法检查并在Chrome刷新；它是证据展示层，不改变FC-P001协议、MPC/FNO闭环未完成状态或项目验收结论。

Lead负责目标/批准/综合证据；Physics/Data负责真实数据与参数覆盖；Surrogate负责可复现模型实验；Control/Evaluation负责控制合同与独立验收。四并发槽中控制与评价职责错峰承担，不扩张代理数量。
