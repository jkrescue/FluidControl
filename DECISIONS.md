# DECISIONS

## 用户授权的一小时指标复评（2026-10-05 14:20 UTC / 北京时间22:20）

用户要求：继续尝试一小时；若仍不能解决，可适当放宽“平均升力偏置不超过基准波动10%”的要求。决策时间不早于2026-10-05 15:20 UTC（北京时间23:20），本小时保持原指标和当前P018正式评估不变。

该授权仅针对真实CFD控制验收中的绝对后圆柱平均Cl / 基准Cl′RMS <=0.10，不自动改变代理预测误差门槛、升力脉动RMS比<=1.05、减阻>=2%、动作/内存限制。先区分实际横向平均载荷与代理预测均值误差，以及波形/幅值误差。若主要失败仍是升力波动预测，放宽平均载荷要求不能视为解决该问题。

一小时后基于同初态配对CFD和预测误差证据评估修订幅度；如采用放宽标准，明确登记新版本，旧10%结果仍并列报告，不回写历史PASS，不把验收标准变化称为模型精度改善。新标准下的最终通过仍须真实CFD闭环验证。用户此次授权优先于历史文件中禁止任何门槛修订的笼统约束，但不授权任意扩大其他指标。

## FC-P015 更新协议决定（2026-10-05；执行后索引既有预声明）

预声明位于`docs/FC_P015_WINDOW_ACCUMULATION_PLAN_20261005.md`（66d591e），启动前审批为`docs/FC_P015_EXECUTION_APPROVAL_20261005.json`（b0c326a）。本段在启动后补充索引，不冒充新的事前记录。

P014表明原固定六窗H1/AR目标均退步，并非纯偏移；原1368次preclip梯度均超过clip1，但这不证明唯一原因。拒绝仅做全训练集bias标量校准，因为常数偏移不能修复仍失败的Cl′RMS。选择同P009亲本、同数据顺序/loss/官方架构的8窗梯度平均更新协议，171次更新；不改LR、不加轮数、不选择中间模型。更新次数、Adam moments与累计weight decay随协议共同改变，结论不得归因于单独降低噪声。固定终态仍须原完整formal与真实CFD闭环，P013失败不被覆盖。

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

该次校准已完成且独立重算通过：固定规则选中`alpha=0`，OOF physical rear-Cd/rear-Cl MAE为`0.012857/0.022035`；全train native rear-Cd/rear-Cl MAE由亲本`0.025576/0.064053`降至`0.010462/0.017277`。同时default-TF32的ideal/native rear-Cd bias仍为`-0.004621` physical，说明不能用理想仿射拟合替代真实执行。Lead据此批准一次原封不动的formal suite；该批准是“值得测”而非“已通过”，不得改`alpha=0`、阈值或跳过force-window/development gate，PPO仍未授权。

formal suite已完成并反证准入：总receipt SHA为`14fd24d9…edcb5`，development FAIL，force-window仅1/6联合通过。四个旋转分支rear-Cl′ RMS误差相对C均明显下降，但dynamic delta-Cd、validation rear-Cd以及多个window mean-Cl/Cd指标退化；因此不能将“升力RMS局部改善”改写为整体控制准确度成功。FC-P008不进入PPO。后续FC-P009只研究train-only free-AR隐藏特征及matched-weight H1对照，不能改变或追认本次阈值。

## D019 — 批准FC-P009 free-AR隐藏特征的train-only CPU实现与测试

2026-10-05 Asia/Shanghai，implementation and CPU tests approved；GPU提取另审。FC-P009检验固定FC-P003C/default-TF32/high模型的free-AR训练窗口隐藏特征能否改善四力末层读出，不改变模型架构、数据、归一化、reward或任何科学门槛。`alpha`固定为`0`，不运行六alpha选择；四相OOF只评价固定读出，不能用于选参。输入仍为既有regular sampler的1368个H100训练窗口、每窗100个相对时刻，共136800行隐藏特征；它们只覆盖19648个唯一真实CFD端点，同一端点在不同AR起点/relative horizon下的隐藏状态不得冒称独立物理样本。

family顺序固定为base20/train8/train16，share仍为`(720,408,240)/1368`，而不是按端点数分配；对应window-step行数为72000/40800/24000，唯一CFD端点数为16000/1600/2048。四折按canonical source phase进行，OOF须按预定全局row weight汇总，不能在缺family的fold内重新等权。原regular sampler暴露保持，但这不是严格单因素消融：相对FC-P008，隐藏状态从true-state H1换成free-AR，同时同一真实端点因AR起点/relative horizon重复出现，row权重结构也随之改变。为分离该权重混杂，CPU阶段必须从现有P008 H1 cache将相同19648个目标映射成同一136800个window-step目标、保持目标值与phase逐项相同，并以固定`alpha=0`计算matched-weight H1对照；AR fit只能与这个对照和原P008两者并列解释。两种fit必须共用从同一HDF raw四力按P008 canonical NumPy-float64规范化后转float32得到的标签；官方DataPipe的torch-float32规范化另作同raw端点审计并记录两条算术路径的差异，不能用直接`array_equal`误判舍入差，也不能放宽容差掩盖case/step错配。不得读取validation/frozen选参，也不得因validation10已FAIL而事后改变阈值或权重。实施先限于一次cache-only特征提取及CPU拟合；生成候选、native replay、formal和PPO均须另审。

FC-P009已完成且未生成候选。completion/result/cache/CPU交叉分析SHA分别为`0893ec75…c214f`/`1321c30a…91daf`/`fc1b84fd…8ca84`/`ffd48eba…7516`。固定`alpha=0`下，free-AR fit在held free-AR域的all-step rear-Cd/rear-Cl MAE为`0.013660/0.039128`，优于matched-H1 fit的`0.026794/0.056831`；H100为`0.013683/0.043003`对`0.032974/0.067989`。但同一free-AR fit在held H1域的all-step误差为`0.022449/0.053706`，明显差于H1 fit的`0.013412/0.022520`；H1-step为`0.020044/0.042925`对`0.011691/0.019504`。因此证据支持状态分布特异的可读性，而不是可部署的共同受力头；不生成P009候选、不申请formal或PPO。

后续固定50/50共享头与受限候选构建分别经独立审批执行；这不是对上述专用头结论的追认。候选原formal receipt SHA `ac5c0dd0…e231c`的18项文件SHA一致，validation10和dynamic6端点组件通过，但force-window仅两个zero分支联合通过。四个旋转分支rear-Cl′ RMS误差`0.06750/0.12501/0.07045/0.07987`仍全部超限，故FC-P009 development FAIL，不进入PPO。唯一批准的后续是从既有train-only cache按原固定共享头重算100步及尾62步的mean-Cd/mean-Cl/Cl′ RMS并按family/case/phase分组；该CPU诊断不调alpha/mix、不读取validation/frozen、不形成新候选。

上述CPU诊断已完成（SHA `f6c122a6…b626`）。joint full/phase-OOF在free-AR尾62步的Cd/mean-Cl/Cl′ RMS MAE分别为`0.00671/0.01984/0.01760`与`0.00751/0.02195/0.01985`，说明phase留出只解释小部分退化。train8 family pooled RMS为`0.02936`，最差既有train PRBS/PPO cases已达`0.046–0.058`，所以时间窗幅值误差在训练profile内部异质存在；但这既不证明覆盖是唯一原因，也不证明default-TF32数值误差放大了formal失败。下一步若做native-vs-ideal诊断，必须冻结当前候选、case/window和计算协议，且只能作机制定位，不能修改formal门槛或追认PPO。

冻结候选的4窗native-vs-ideal检查已完成（SHA `2dea49fa…b1ef`）。runtime hidden features与cache逐值相同，pointwise wiring为0；保存float32头的native-minus-pooled-affine尾62步Cl′ RMS差为`-0.001942/-0.000292/-0.000851/+0.000152`，而相应native真值误差为`0.00794/0.12561/0.08839/0.06647`。除zero基准外，执行顺序差异远小于旋转窗误差，故不再把TF32/native归约作为该失败的主要解释，也不继续此诊断分支。后续只评估一个明确针对train-only尾窗幅值的监督干预；它仍须保留完整field/force formal协议，不能用训练窗改善替代准入。

Lead随后只批准一个有界CPU-cache判别：将相同136800行、相同canonical targets/source phase/原row weight的H1与free-AR特征各自归一化为总质量1后乘`0.5`，在每个source-phase fold中只用fold-train联合数据计算一个共享scaler，并拟合一个共享`alpha=0`仿射头。held phase必须在两个域分别报告all-step与H1/H10/H50/H100的逐通道物理误差；不得搜索mixture或alpha，也不得用单个平均数建立新准入阈值。该诊断只判断一个共享头能否兼顾两种隐藏状态分布；不要求它逐项支配两个分别优化的专用头。即使训练内改善，也仍须另行批准candidate/native replay并通过原formal gates。

共享头诊断已完成（SHA `931fcd2d…f2b0bc`）。H100 AR域rear-Cd/rear-Cl/total-Cd MAE为`0.01612/0.04610/0.01607`，相对C亲本`0.01637/0.07392/0.01637`改善但不及AR专用头`0.01368/0.04300/0.01367`；H1域为`0.01593/0.03035/0.01572`，优于C亲本`0.02622/0.06333/0.02580`但不及H1专用头`0.01432/0.02447/0.01411`。这支持折中头的train-only可表达性，不证明validation或控制收益。Lead只批准最小候选实现和CPU测试：复用缓存系数，官方加载C-e2，只改四个force rows/bias，保存为独立epoch0后重载并逐tensor核confinement；default-TF32/high下仅用固定base20首个train batch的H1作`2e-5` pointwise wiring sanity和有限性检查，pooled/native差只报告。候选GPU构建、原formal suite与PPO仍分别需要明确批准。

## D020 — FC-P010尾窗幅值监督显示多域权衡，不构建线性头候选

FC-P010在固定P009 train-only cache上保持H1/free-AR逐步rear-Cl损失各占一半，只新增free-AR尾62步centered-RMS项，并仅优化rear-Cl行。结果SHA `d69033fd…8f94`的输入绑定和指标已独立重算：free-AR四个phase OOF RMS误差均改善`1.45%–7.34%`，但H1有三个phase恶化约`13%`；full fit为free-AR改善`9.02%`、H1恶化`10.51%`，Cd因其它三行冻结而逐值不变。全部五个LBFGS fit都用尽200次且梯度未达容差，因此不能宣称已收敛或找到最优头。

该结果不要求每项支配作为新gate，但其方向一致的跨域代价不足以支持候选构建。停止继续扫描固定线性头、loss mixture或相位权重；下一假设必须是有界的train-only官方FNO训练干预，之后仍用原field/force/dynamic/window formal协议裁决。不得把训练窗改善、有限完成或CPU诊断状态写成PPO准入。

## D021 — FC-P011训练完成但固定train诊断为权衡，以原formal裁决而非训练指标挑选

FC-P011两臂均按同一1368-window顺序完成且scope auditor通过。A只训练rear-Cl行，free-AR rear-Cl逐步/RMS约改善1.5%，但true-state H1退化；B额外训练最后decoder hidden linear层，free-AR rear-Cl逐步/RMS反而恶化6.65%/3.37%，同时free-AR field与H1 RMS改善。该结果既不支持按训练指标直接接受任一臂，也不允许把A→B之外的变化归因于scope。两臂原formal现均已完成并FAIL：A/B窗口都只通过两个zero分支，旋转rear-Cl′ RMS误差分别为`0.070312/0.122247/0.068076/0.083214`和`0.062584/0.119208/0.075209/0.076236`；B的validation10 delta-Cd还以`0.024318>0.023`失败。故A、B均拒绝进入PPO，原门槛保持不变。

## D022 — FC-P011后先做train-only梯度分解，不自动切换路线或增加训练

FC-P011的局部scope干预未修复formal窗口失败，但训练记录显示A从不clip、B每步都clip；这些统计不能单独证明是field还是force梯度造成。下一优先只准备一个no-optimizer、no-save、无validation/frozen/PPO的train-only诊断：固定既有6个窗口，在P009亲本与P011B终态上分别重建相同H100 loss graph，独立计算field与weighted-force梯度的范数、夹角和合成前后clip尺度，并按允许tensor组报告。它只用于判断B的scope权衡是否伴随梯度竞争/尺度失衡，不构成新gate，也不从梯度相关性宣称因果。用户最终目标和force/world-model→PPO→真实CFD路线不变；任何训练干预仍须另行审批并通过原formal。

canonical surrogate的reward warm-up与direct-CFD reset不一致属于接口缺陷而非新科学变量。commit `962c165`只让canonical路径从同一绝对restart时钟恢复真实62点prehistory，并严格绑定来源；32项CPU回归及两次独立复核均PASS。该修复不追认历史候选、不放宽门槛，也不等于控制执行授权。

## D023 — FC-P012不支持以强梯度尺度失衡或反向冲突解释FC-P011B

FC-P012按预声明在P009亲本和P011B终态各复算相同六个train-only H100窗口的field与`0.2×balanced-force`梯度。五个非zero-action-history窗口中，两模型的hidden组范数比`>10`计数均为`0/5`，cosine`<-0.2`计数也均为`0/5`，未触发既定的`4/5`或`3/5`解释条件。zero窗另列而不混入计数。故不根据该诊断调整loss权重、clip或重启训练；它也不证明不存在局部或其它参数组的优化问题。

result/completion SHA为`4142cdc5…d814d`/`86f9d931…b26c`，12行均有限且模型tensor前后相同。direct-total与组件和的relative residual为`3.09e-5–9.84e-5`，按预声明仅作observational、没有数值等价阈值，不能从COMPLETE推断等价PASS。下一representation-capacity诊断或训练方案须独立批准，原formal门槛和PPO阻断不变。

## 新决策格式

## D024 — FC-P013独立受力FNO正式训练与独立诊断调度

2026-10-05 06:13 UTC。资源探针v2完成且未更新任何模型后，Lead按`FC_P013_INDEPENDENT_FORCE_FNO_PLAN_20261005.md`批准一次固定1368窗口训练。冻结P009的流场递推，另一个同架构官方FNO从相同P009初始化，训练完整可训练气动力表征；H1/free-AR等权，四力均方误差与rear-Cl均方误差等权。它是组合系统实验，不是梯度冲突已获证实或单独容量因果实验。

最终源码`1634c05`通过24项CPU回归；此前核心数学、官方冻结参数和双模型契约已有独立审查。三名代理均反复因模型服务容量错误退出，因此Root接管最终trainer恢复和审查，并如实记录额外独立复核未完成，不能据此宣称科学准入。原固定六窗物理指标转移到独立只读评估，严格保持原数据/窗口/H1-H100/tail62定义，不用于模型选择。正式训练只保存终态并要求官方双模型fresh reload。

训练4小时上限，统一内存守卫20 GiB、allocator<=0.45，实际训练集只读挂载；无validation/frozen/PPO。执行批准SHA `1bdcfcf7…1a120`。训练结束仍需核验实际1368条记录和所有来源SHA，完成固定诊断和原完整formal suite。若失败，保存负结果并分析下一可检验假设；不改科学门槛，不自动把训练损失下降写成减阻成功。

ID、记录时间、状态、待检验假设、对应实验ID、所依据证据/协议、可选方案、取舍原因、保留的不确定性、撤销/调整条件。只有读取过的产物可作为事实；代理口头报告是待核信息。
