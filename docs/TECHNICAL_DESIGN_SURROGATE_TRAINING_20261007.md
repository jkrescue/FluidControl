# 技术设计：串列双圆柱代理训练、诊断与分层验证

2026-10-07。本文是基于实际源码与已执行证据的技术设计，不是新增执行批准。只新增本文，不改变默认B、数据、阈值或训练器。未来实验均需独立批准；当前科学截止12:20:45 UTC、归档截止12:50:45 UTC不因本文顺延。

## 1. 当前可复现基线与研究边界

物理对象为Re100、中心距L/D5、固定中心串列双柱，后柱旋转；不是VIV或三维尾流验证。OpenFOAM步长.005 D/U，动作反馈间隔.1 D/U，800反馈为80 D/U。E114由冻结CPU PPO读取真实CFD观测，无在线FNO；同分支累计160 D/U不等于独立工况。

当前默认B双FNO manifest：`artifacts/fcp064_controlled_aero_arm_b_20261006/dual_model_manifest.json`，SHA `92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891`。
flow archive SHA `dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31`；aero archive SHA `57d4634df22ce96c1c4467a2ed52412be452375129af05b89f10a690e363356e`。
normalization SHA `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1`；config SHA `07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9`。

不能把Git中的当前同名模块当历史原字节；实际运行以approval、immutable source、输入和runtime哈希为准。以下源码路径用于解释接口，不授权无绑定直接重跑。

## 2. 数据接口与时间语义

- 原始CFD保留U/p/phi、backward旧时层、网格、BC、动作表及前后柱受力；Curator只转换，不生成新真值。
- ROI为x=[8,25]、y=[4,11]，state为3×128×256的u/v/gauge-pressure，mask为1×128×256。粗ROI不含逐壁面牵引/法向/面积，集成force标签不能反解成真实局部压力或剪切分布。
- 依据`src/fluid_control/tandem_datapipe.py`，当前state先按固定train统计标准化再乘mask；target为下一帧四力按同一force mean/std标准化。固体零值是mask约束，不是用于拟合的流体真值。
- omega_current和omega_next分别对应相邻存储端点；输入动作除manifest的action_scale。不能另把控制上限.75硬写成所有数据集的normalization尺度。
- 当前动作样本是存储的实际施加端点值，不等价于把名义PRBS命令在float32时钟重新插值。保留源float64时钟与HDF存储时钟的既定对齐规则。
- `scripts/p026_state_history.py::build_input`只收历史state/action、mask、事先选定next action，无future state/force参数。K1只有一帧；历史padding、跨轨迹边界须保持。
- B原数据为base20/train8/train16共44轨迹，另加controlled-b00；窗口步幅分别20/2/2/1。B选256窗=192 original+64 b00，8窗累积，不把每窗100个高度相关端点当独立实验样本。
- 侧车45轨迹20,493帧pressure/viscous标签是另一次已核数据转换，未替换原HDF total力；原total与raw分量和存在已报告小差异，不得靠相减伪造“原始viscous真值”。

## 3. 官方模型与项目适配

每个网络使用官方PhysicsNeMo FNO：in_channels6、out_channels7、latent48、5层、二维modes[32,32]、decoder2层/宽128、padding8、coord_features=true。
显式输入为归一化u/v/p、mask、current/next omega；坐标由官方模型内部加入。输出前3通道为归一化状态增量，后4通道为force读出。
项目用mask加权空间平均将后4输出转成frontCd/frontCl/rearCd/rearCl，非真实壁面积分。flow按`q_next=(q+delta)*mask`递推；aero负责四力。

`scripts/train_fcp013_independent_force_fno.py`提供冻结flow rollout与true-state H1序列；`scripts/p026_history_objective.py`构造相同动作的H1/AR两支；`scripts/train_fcp064_controlled_aero_ab.py`及`src/fluid_control/p064_controlled_aero_ab.py`约束B调度。
H1每步使用真实current state；AR使用冻结flow自身预测的current state。aero力预测不反馈成flow输入，不喂真实current force。历史状态/未来动作合同必须在任何新目标下保持。

B训练28个aero参数张量；flow及`spec_encoder.lift_network.0.conv.bias`、`.2.conv.bias`冻结。这里“两个lift bias”指输入升维层偏置，不是物理升力输出bias。
目标为`.5 L_H1 + .5 L_AR`，每支四力normalized MSE权重[.125,.125,.125,.625]；100步分10个等长chunk，每chunk权重.1，8窗累积后clip1和一次AdamW。
实际B：32更新、lr1.5625e−7、betas(.9,.999)、eps1e−8、weight_decay1e−4、seed20261003。历史precision为high/TF32；highest/no-TF32是显式后续协议，不能混用缓存B作对照。

## 4. 现有证据能说明什么

E114真实控制原物理门PASS，但B完整force-window/H100预测仍FAIL。可用策略不意味着精确代理，也不意味着MPC可用。
F纯H1、Absolute64、AR5 reset、反射、pressure-aux、temporal increment、I晚期覆盖均已尝试；未通过既定组合条件者不重命名为新思路。
I原six改善而固定dev退化，说明训练覆盖收益未自动泛化。真状态替代结果也不支持“只有flow累积误差”这一单因解释。
固定40点和Representative256的LBFGS预算结束时loss仍下降；未达到每通道normalized RMSE≤.01，不能据此判定容量不足、收敛或不可拟合。
Representative256 selected loss下降70.84%，同precision six H1/AR却退化15.8169%/14.2703%。b00的64点贡献rear-Cl/total-Cd MAE总改善87.3%/98.7%，train8所选45点四力MAE全退化；原三family前柱力均变差。
该failure-map支持来源/通道权衡与选点外失败并存，不能区分梯度干扰、稀疏覆盖或参数漂移。见`docs/P064_REPRESENTATIVE256_FAILURE_MAP_20261007.md`。

## 5. 优先诊断与可证伪设计（未来，未执行）

先固定问题与计算预算，再决定是否训练；不得靠轮番调lr、loss或modes找偶然通过。

| 优先级/问题 | 最小对照与观测 | 可证伪结论与限制 | 建议预算上限 |
|---|---|---|---|
| P0 覆盖还是来源权衡 | 固定B父本、train-only配额和记录数；按family/动作/原点lead列误差、重复与相邻相关性。已有saved map先用完 | 旧family选中点已退化，排除“只有选点外泛化”解释；不是来源的因果效应 | 已存数组CPU≤2min，无模型 |
| P1 参数梯度干扰 | 预注册各family相同数量训练样本，固定原目标/最高精度；一次无更新分组梯度Gram、cos、方向分量。保留源比例的合梯度，报告原始与单位化统计 | 若方向一致且冲突不集中，则反驳该面板的干扰假设；一面板不代表全训练 | 一次GPU诊断≤10min，0optimizer/0save；先做显存预核 |
| P2 覆盖检验 | 仅当P1/覆盖统计提出具体缺口，固定总sample/forward预算，预声明分层train抽样对照；其余初始化/目标/优化器不变 | 比较等预算训练和独立保留集；改变样本相关性本身是变量，不能称只改数据量 | 两臂预算必须事前合计；不在本轮截止前仓促启动 |
| P3 normalization/读出 | 先核train统计、每通道误差物理换算、mask/面积归约和参数梯度；对真实raw压力/黏性作已存误差分解 | 输出曲率1/σ²大不等于参数梯度主导；已有单窗梯度否定frontCd必然主导 | CPU统计优先；必要一次无更新梯度≤10min |
| P4 优化还是容量 | 固定train面板/初值，明确closure与接受点；只在目标停滞且数值/标签/优化诊断充分后比较有限容量候选 | 未拟合≠容量不足；训练内拟合成功也不证明泛化 | 先复用300closure证据，不直接加一轮 |

P1不同于既有单窗四通道loss-scale probe：问题是固定来源间的梯度关系，而非再比较frontCd/rearCl权重。它仍只是未来建议，不能凭结果直接给某family加权；后续改权重必须另列唯一变量和对照。
梯度夹角不是Adam/LBFGS实际更新的替代。若检查优化器效应，应在同一保存状态上计算实际proposal方向及预算，不能把grad norm、clip后norm和参数delta混称。

## 6. 四层验证与停止规则

1. **工程与训练内拟合：** 检查完整数据身份、loss数学、有限梯度/更新、官方save/freshreload、冻结项和资源。`.01`是既定小面板每通道normalized RMSE工程目标，不是预测科学门或物理控制门；未达如实记录。
2. **原six保留性：** 同一B与候选、同precision、原6来源/起点、H1与continuous AR100、同归一化/动作。原两项均不退化AND不改；保存小force数组支持独立复算。不可用train loss代替。
3. **固定开发与正式预测：** 原16origin×H1–H5、同B comparator和persistence；完整force-window/H100另按既定协议。开发只支持开发结论；已打开b01/b03不重新包装成sealed。原相应AND/门限不因接近而放宽。
4. **真实CFD：** 只有独立明确批准的候选才做控制探索；预测FAIL下的探索须单独标注而非模型晋级。配对同restart zero/controlled、原动作/窗口/策略种子，完整六窗与raw受力验证。

真正物理门：D=1−mean(Cd_front+Cd_rear)_controlled/mean(Cd_front+Cd_rear)_zero≥.02；R=std_de-mean(rearCl_controlled)/std_de-mean(rearCl_zero)≤1.05；Q=abs(mean(rearCl_controlled))/std_de-mean(rearCl_zero)≤.10。
Q不是两支均值差，也不除以meanCl；omega²不是机械功率。未来新候选不能把训练`.01`、PPO reward或短H5 Cl²当这三门。

## 7. 完整预测与控制相关指标

固定报告matrix，不只挑有利总均值；对每case/origin/horizon、来源family与pooled均保留分母、样本数和实际时刻。

- **场：** fluid-mask u/v/p逐通道relative L2及absolute RMSE、空间误差图、bias、压力参考/去均值规则；近零参考范数须另报绝对值，不能靠不透明epsilon掩盖。H1/H5/H100图分别绑定模型/动作/起点，不用历史图冒当前。
- **力：** 四力与totalCd的MAE/RMSE/均值偏差；rearCl去均值RMS与真值比、幅值与长滚动漂移。先求totalCd误差再absolute，不能加两个Cd的MAE代替。
- **相位/频谱：** 在足够长且等采样的固定窗比较主频/PSD能量、cross-correlation时滞与相干性；窗函数/去均值/频率分辨率事前固定。不得平移对齐后只报校正误差；保留未校正指标。H5=.5 D/U不足支持可靠整周期PSD/相位准入。
- **控制动作价值：** 必须同真实q0、因果历史、动作幅值/速率合法的多分支CFD真值；报所有候选cost、pairwise差值/排序/tie、选中动作真实regret及不确定性。realized-action回放不是反事实动作排序证据。
- **奖励：** 复用`src/fluid_control/canonical_joint_v1.py`及原62点历史；endpoint cost与gamma=.99截断return分开。历史缺失时不能伪造/reset后称原reward。无匹配历史则只报物理分量。
- **统计：** 同一轨迹重叠origin不是独立重复；报告组内/组间分布。只有预注册独立轨迹/seed足够时才估计泛化或显著性，不能把160D/U切块当多个随机样本。

## 8. 模型扩容与batch的有条件方案

官方2D FNO已能配置层数、宽度、modes与decoder；增加modes不是默认修复。先核采样分辨率/Nyquist、padding边界、mask几何、训练拟合与等预算对照，再考虑单个扩容变量。
FNO3D必须先验证安装版本官方接口与真实轴语义。本case是二维空间；把时间堆作第三轴是时空算子方案，不是三维物理模拟，且必须严格因果切窗，防把future state作为输入。
K1→多历史帧会改变输入与checkpoint合同，不能仅改维度后宣称同模型。需要新config/consumer、history padding/时钟fixture与独立baseline；历史K4/P026结果先查重。
增加microbatch只可作为数值/资源方案：若保持全panel平均，末小batch按真实N加权（256点为25×10+6，权重10/256和6/256）；等权平均26个batch会改变目标。
effective batch、更新次数、数据顺序、precision和optimizer历史须分开控制。吞吐提高不代表新增训练预算或精度收益；gradient accumulation也不能替代完整AR依赖的数学验证。

## 9. 无泄露、版本与复现工件

- train统计只从批准train来源计算；新normalization属于科学变量，不能悄悄重拟合。保留旧norm字节并报告所有重标度。
- 将source case/起点/lead/action/history/target SHA写入面板；原重复保留，不靠删除困难点改权重。dev不用于拟合或选择每次closure；未打开测试不得提前画图或扫阈值。
- 每次记录代码、环境版本、实际import origin、precision/TF32、随机种子及RNG、模型与optimizer状态、数据/norm/split、官方save/load metadata、实际unit/inv/资源和失败日志。
- 保存initial/final与接受点指标，trial与接受点分开；预算中断要恢复参数/optimizer/grad事务，不能把未接受trial当最佳候选。
- 推理保存truth/prediction/mask/坐标/时钟/action和force，支持CPU复算；大HDF/checkpoint留主节点，Git只保存源/协议/报告/清单，不能宣称Git-only重建完整链。
- 复现入口参考`scripts/reproduce_canonical_closed_loop.py`与`docs/CANONICAL_MULTI_STAGE_RUNBOOK_20261007.md`；默认只预检，旧approval不是新运行许可。

## 10. 本次实际检查的源码与尚缺证据

实际检查：B manifest；`src/fluid_control/tandem_datapipe.py`；`src/fluid_control/p064_controlled_aero_ab.py`；`scripts/train_fcp013_independent_force_fno.py`；`scripts/p026_state_history.py`；`scripts/p026_history_objective.py`；`scripts/train_fcp026_history.py`；`scripts/train_fcp064_controlled_aero_ab.py`；`scripts/evaluate_p064_development_h1_h5.py`；`src/fluid_control/canonical_joint_v1.py`。原six流程另见`artifacts/p064_fit256_fixed_six_source_20261007_immutable/p064_fixed_six_same_precision.py`与独立终态报告。
查重与结果依据：`PROJECT_STATE.md`、`EXPERIMENTS.md`、Representative256训练/fixed-six/failure-map报告、I开发报告、压力aux及temporal-increment报告、B-H5 MPC与saved-action/reward诊断报告。
仍缺：固定来源分组梯度证据；覆盖干预的等预算因果对照；通过完整气动力预测门的候选；足够独立工况的统计泛化；完整壁面牵引输入与消融；净执行功/实验实时性；优于保留B的有效FNO-MPC。本文不把这些缺项写成已实现或因果结论。

**当前决策：** 保留已交付B-PPO真实CFD案例与所有失败证据。本轮只文档；不再盲训。后续研究先回答一个可证伪问题、固定总预算与验证层级，再决定是否值得进行单因素训练。
