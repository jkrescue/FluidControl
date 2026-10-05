# DECISIONS

本文件于2026-10-04重建。旧决定的记载是事后整理，不伪装成历史预注册。新决定先登记再实验。原始失败与证据不覆盖。

| ID | 决定/状态 | 原因和证据 | 后续检验/改变条件 |
|---|---|---|---|
| D001 | 保持固定Re100、L/D5、后圆柱旋转；accepted | 当前数据和已验证CFD控制均为该场景；不能把用户举例的多U当已有数据 | 当前闭环验证完成后再单独批准跨Re数据设计 |
| D002 | 以整体减阻及升力约束为目标；accepted, reaffirmed | 真实CFD既定>=2%减阻、RMS<=1.05、偏置<=0.10 | 不因候选失败而放松；目标变更需独立记录 |
| D003 | 保留官方二维FNO和官方数据组件；accepted | 当前存在可复现数据/模型/控制接口，未证明结构容量是根因 | 仅在误差分布和受控对照支持时提新架构实验 |
| D004 | 当前优先控制相关幅值/动作响应，不只看field loss；accepted | train16动态端点Cd通过但力窗口失败；加权分支幅值动作方向错误 | 完整后评估验证paired λ0/10是否真正改善 |
| D005 | 拒绝当前λ10 paired-stat干预；rejected by FC-P001 | 同亲本/同数据/同协议下，λ0与λ10均只有2/6 zero分支通过force-window；λ10将四个旋转分支Cl′ RMS误差均值仅降1.68%，个别phase有改善也有退化，仍约为限值的6倍 | 保留两支失败证据；完成FC-P002失败图后，只提交一个Lead批准的FC-P003单因素假设，不启动代理PPO |
| D006 | 不推倒已有CFD-only PPO，不强制从MPC重开项目；accepted | 已有真实反馈减阻约4.2%的有效基线 | 可在代理通过验证后加入有限时域MPC诊断对照；MPC也受代理误差影响，不能绕过精度要求 |
| D007 | 不新增统一加权reward替代既定物理约束；accepted | 权重混合目标可能掩盖升力超限；已有动作与物理验收合同 | MPC/RL比较必须同观测/动作/起点/预算/物理约束，reward改变另做消融 |
| D008 | 跨Re/OOD与主动采样暂列后续，不声称现已具备；deferred | 当前不足是固定Re下动作/时序预测；未知不等于已定位跨Re泛化问题 | 先明确训练参数支持、误差图及不确定性校准，再设计新CFD采样 |
| D009 | 论文价值待实验支持；accepted | CFD-only收益不能归因FNO；b00/b01非独立工况；尚无样本效率/净电能收益证据 | 增加冻结测试、多seed、CFD预算与收敛/稳健性证据后再扩大主张 |
| D010 | 以持久状态和实验台账协调代理；accepted now | 多次流程结束、后处理失败与旧状态导致进展混淆 | 每关键节点更新并独立复核；阶段完成不得改project_goal_complete |
| D011 | 区分港理工相关研究背景、独立实现与严格论文复现；accepted now | `docs/POLYU_ZHAO_2024_SOURCE_AUDIT_20261003.md`记录部分原文参数尚未核实，而旧`docs/PAPER_REPRODUCTION.md`含较具体及较早阶段描述 | 不把旧文档更肯定的说法当证据；建立原文页码/公式对照后才宣称一致。当前整体减阻+升力约束是项目目标，不能冒称论文原奖励 |

## D012 — 补全 canonical 代理验证的数值生成程序

2026-10-05 Asia/Shanghai，accepted before FC-P003/P003B完整后评估。
独立源码和历史审计确认：旧window/dynamic receipt仅有consumer字段合同，
没有历史数值producer。采用2026-10-04已经固定的development误差标准补全，
不降低阈值、不将今日定义追认成历史预注册，也不把共享证据的receipt描述为
独立科学实验。实现须从校验后的原始评估证据重算，而非改名PASS布尔值。
详见`docs/CANONICAL_SURROGATE_PROTOCOL_COMPLETION_20261005.md`。
只授权CPU producer/诊断；PPO及最终真实CFD物理验收要求不变。

## D013 — 先恢复已完成的后评估阶段，技术探针不代替科学实验

2026-10-05 Asia/Shanghai，accepted operational decision。FC-P003B首次后评估在validation10 GPU推理完成后因audit挂载路径合同不一致失败。保留原失败，只允许不可变恢复程序在重算SHA、checkpoint、有限性、计数和数据合同后复用完整推理pair；成对文件缺失、哈希不符或未知故障必须停止，不得为赶进度重跑或改阈值。端点组件PASS不得越过dynamic6、force-window或development gate。

true-state paired-force GPU工程探针只被批准检查固定真实train-only pair上的causal索引、chunked gradient等价性和峰值内存；无optimizer、无权重保存、无validation/frozen数据。v1因原容器身份下mode-600输入不可读而在forward前失败，不对更底层rootless/user-namespace原因作未验证声称。Lead后续明确批准的v2只修复容器UID/挂载可读性并使用新独占输出，数值合同不变；它已通过梯度等价与资源检查，且模型parameters/buffers未变。该技术PASS不是候选代理、PPO准入或闭环成果；正式损失实验仍须等FC-P003B完整结果并单独批准。

## D014 — 拒绝FC-P003B并仅批准FC-P003C工程实现

2026-10-05 Asia/Shanghai。FC-P003B在与FC-P003相同的亲本、regular顺序、λ10、update数、epoch数和评价协议下，只将static16配对监督内容改为dynamic8×2。四个旋转分支rear Cl′ RMS误差较FC-P003均值仅降2.94%，force-window仍2/6、development FAIL，因此该干预不足以支持代理PPO。true-state H1的旋转rear-Cl MAE仍为0.156--0.192，说明失败不能只归因于自回归累积。

Lead以`docs/FC-P003C_APPROVAL.md`批准单因素下一步的工程实现和CPU测试：保持所有数据、顺序、模型、归一化、超参数及验收不变，只将配对损失替换为true-state每端点action-minus-zero四力误差。固定`w=[1,1,4,1]/7`和λ10不意味新旧损失梯度等强；必须报告通道损失/梯度贡献，不得看到validation结果后再改权重。完整GPU训练需另行批准，原准入门槛不变。

## D015 — FC-P003C若完整失败，先分解训练拟合与泛化缺口

2026-10-05 Asia/Shanghai，conditional，只在FC-P003C完整后评估仍失败后执行。epoch1 `training_history.json`显示true-state paired的rear-Cd/rear-Cl加权贡献为0.0589506/0.00223312（约26.40倍），16个组合更新的pre-clip norm均大于1（最小11.7792）。源码证明这16个组合更新占用1368个regular batch中的预定位置，每epoch总optimizer step仍为1368，不是1384。这些是诊断现象，不足以把clipping定为因果。

若触发本决策，优先用相同true-state H1协议比较现有train8与dynamic6数据，并按动作/相位分组，区分训练拟合失败与泛化失败。不盲目增大λ或网络，不改准入门槛，不授权新readout拟合或GPU实验。

补充的CPU-only动作端点覆盖审计（receipt SHA `eb19fb2aff2628e2383707b23376201994dd17f29b8651c156d3310d1d1e4074`）在train8的1600个transition与dynamic6的1200个transition之间找到1004个精确局部动作特征匹配，归一化最近距离最大为0.07511075。该审计只读`omega/time`，不支持“明显的局部转速幅值/步长覆盖缺口”解释，但未测state/phase/history/force联合覆盖，不能证明泛化，也不得把ω距离单独当作模型失败的因果。审计产生时FC-P003C仍在epoch2训练，该条本身不触发失败后实验。

时域索引另须明确：train8 action HDF每条201帧，zero HDF每条801帧；paired DataPipe只额外监督索引0–100中的targets 1–100，而常规train8 H100/stride2的408个训练窗口仍覆盖到target200。D015已经在FC-P003C完整gate失败后按固定合同执行，分成`train_paired_window` targets1–100（800 delta）、`train_late_window` targets100–200（808 delta）和`validation_late_window` targets100–200（606 absolute/404 delta）三个面板；target100重叠，不是独立重复证据。targets101–200未获额外paired监督，但不是“未训练”。

结果拒绝了“只有验证分布泛化失败”这一单一解释：rear-Cl delta MAE在额外paired监督窗口、同轨迹late窗口、validation late窗口分别为0.11880/0.10753/0.17185；即使训练前缀也没有被准确拟合，同时validation相对train late仍更差。front-Cd delta MAE仅0.00069/0.00128/0.00133，误差明显集中在后柱受力，尤其rear-Cl。下一项若批准应首先针对train-only rear-Cl拟合/优化暴露做单因素检验，而不是扩大网络或降低门槛；本诊断本身不授权训练、PPO或frozen访问。

## D016 — 拒绝单纯增加delta-only曝光，仅批准absolute-paired CPU实现

2026-10-05 Asia/Shanghai。固定C亲本的128/64有界校准使paired/late rear-Cl delta MAE仅下降3.02%/2.57%，但absolute MAE与u/v/p场误差均退化；zero rear-Cl MAE约恶化3–4倍。由于action与zero共享的force偏差会在delta loss中严格抵消，Lead批准一个单因素CPU实现：将paired项替换为`0.5 * (weighted action absolute MSE + weighted zero absolute MSE)`，其余亲本、数据、regular loss、预算、λ、权重、学习率、seed、clip及门槛不变。若未来获批执行，必须分别报告action、zero、delta四力与paired/late场误差；zero改善而action/delta无一致改善，或force改善伴随field退化，均反证该机制。当前commit `5cb65bb`仅为CPU-tested代码，不授权GPU、正式后评估或PPO。

该实验已按固定合同执行并触发反证条件：zero rear-Cl误差显著恶化，delta改善很小，且所有field通道退化，因此absolute监督分支也被拒绝，不做正式后评估。现有64个paired update中，四通道加权贡献占比约为front-Cd 2.83%、front-Cl 0.31%、rear-Cd 93.01%、rear-Cl 3.85%，64个pre-clip norm全部大于1（中位26.46、均值30.96）。action与zero的有符号bias变化在每个通道上数值接近，说明存在共同移动的相关模式；但这与贡献/裁剪统计都不是因果证明，不授权继续增大λ、曝光量或模型。

## D017 — 固定特征读出显示fit窗可读但病态，先做train-only ridge稳定性诊断

2026-10-05 Asia/Shanghai。v1在默认TF32下因为先空间平均再做仿射与原逐点仿射再平均的非结合数值差异而fail-closed；数值probe中原逐点FP32顺序可bitwise复现，禁用TF32/highest FP32后换序差降至2.38e-7。v2保持原`2e-5`容差，在该独立数值协议下完成。CPU从cache重求的1600行float64 least-squares系数逐值相同，所有action/unique-zero/delta和逐pair指标精确复现。prefix四通道action MAE降84.6%–93.8%，但矩阵条件数为`1.9016e5`、系数L2为`717.87`，late rear-Cd MAE增加10.15%且rear-Cl仍有`0.08337`。因此不将fit窗低残差解释为已修复优化、不生成部署checkpoint、不启动PPO。

Lead只批准下一个CPU-cache诊断：在prefix做4-fold leave-one-phase-out，固定`alpha={0,1e-8,1e-6,1e-4,1e-2,1}`，每fold仅用fold-train标准化及对称zero加权，以归一化四通道等权mean-MSE选alpha（并列选较大alpha）；选定后在全prefix重拟，只查看一次late。不为每个alpha扫描late，late仍是同train轨迹时间检查而非独立验证。该诊断只检验高方差/病态假设，不允许把失败直接归因于覆盖不足。

该cache诊断已完成（result SHA `dbeee783…50e0f`）：prefix四相留出选定`alpha=1e-6`，aggregate normalized MSE从alpha0的`0.0064395`降至`0.00359875`（-44.1%）。选定后唯一一次late检查的physical rear-Cd/rear-Cl action MAE为`0.03876/0.06274`，优于OLS的`0.05602/0.08337`，但仍不足以支持控制准入。独立CPU复算的六个alpha score最大差`6.4e-12`，所有selected prefix/late指标复现。结果支持系数不稳定为重要贡献，但late仍高、且全部为train-internal，不允许宣称唯一根因、泛化或PPO准入。

## D018 — 批准FC-P008全train family读出校准的实现与CPU测试

2026-10-05 Asia/Shanghai，implementation and CPU tests approved；GPU执行另审。FC-P008只检验固定FC-P003C表征上的四力末层读出，不改变FNO架构、场输出、数据、归一化、reward或既有科学门槛。它使用全部44条train-only轨迹、19648个H1端点；base20/train8/train16的family share固定为既有regular sampler比例`(720,408,240)/1368`，每端点权重为`share_f/N_f`，其中`N=(16000,1600,2048)`。四折必须按真实source phase划分并以全局endpoint weight聚合OOF，禁止先将缺family的fold等权。

source mapping artifact `57ed2a25…3b92`把四个canonical phase固定为b00/t148、b02/t106、b04/t120、b06/t134；train16只可凭source case/time和curated frame-0 identity并入b00/b02，不能按episode名或标签顺序推断。数值协议固定为正式default TF32/high，不沿用highest-FP32诊断路线。alpha grid、并列取较大值、fold-train-only统计、全train单次refit均预先固定；先前train8 late端点在本次属于全train的一部分，但不得用既往late结果事后调整grid或选择规则。最终只允许改新checkpoint的四个force rows，其余tensor须字节不变；default TF32的严格`2e-5` wiring采用captured pointwise-head→mask-mean执行顺序，ideal pooled-affine与native差异只逐通道报告，不假定严格代数等价。当前批准不包含GPU提取、候选执行、validation/frozen、PPO或真实CFD；完整科学准入仍需后续独立评估。

实现已通过独立CPU审查：9项测试及真实44轨迹inventory验证family权重、缺family的fold聚合、fold-train统计、误导性train16名字不决定phase、去标准化、official fresh reload和force-row confinement。Lead随后只批准一次受守卫的Main全train校准执行；这项执行可生成train-only候选和原生replay证据，但不能自行开启formal evaluation或PPO。执行结果须从真实feature cache独立重算CV与native指标后再决定是否申请后续评估。

## 新决策格式

ID、记录时间、状态、待检验假设、对应实验ID、所依据证据/协议、可选方案、取舍原因、保留的不确定性、撤销/调整条件。只有读取过的产物可作为事实；代理口头报告是待核信息。
