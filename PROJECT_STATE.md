# PROJECT_STATE — 串联双圆柱主动流动控制

## FC-E080 当前终态：第二seed未重现减阻，执行链完成但稳健性未证明

同inv111022bf已800/800、PID0/exit0，独审3200raw SHA/1600solver/六窗复算通过，全部zero前后力16000行所有列与旧seed完全相同。主窗(168,228]新seed **减阻−0.6174007%（增阻）**，RMS比1.0062202、bias1.694578%；旧seed+3.8952839%/.8156230/1.137815%。新seed六窗均减阻FAIL、两项升力均PASS，原2%/1.05/10%未改。报告 [第二seed真实CFD独审](docs/P064_B_SEED20261007_CFD_TERMINAL_REVIEW_20261006.md)，result SHA `6221a7d2f8868eba8622f2e6b76d110d206dd4d9304627d890e01d71b6a7d893`。

当前训练/CFD已结束；旧seed三相位收益仍真实，但不能称跨seed稳定或全目标完成。FC-E081 CPU匹配旧轨迹800观测诊断（Root独立核验，报告 [policy-map复核](docs/P064_SEED_MATCHED_OBSERVATION_REPLAY_REVIEW_20261007.md)）显示新seed odd投影RMS .0439748 vs旧.468962、even .591238 vs .402144；只支持相同观测下策略分解差异，不是闭环因果或已验证remedy。预测FAIL/H25拒绝晋级保留，无自动扫seed/加训/改阈值。

### FC-E080 历史启动记录（已结束）

实际unit `fluid-control-p064-b-seed20261007-projected-ppo-long-cfd-20261006.service`，invocation `111022bf633246e69165f7b8eb3edb01`，PID2811798 running；独查22/800周期。批准 `docs/P064_B_SEED20261007_CFD_APPROVAL_20261006.json` SHA `4bd940f088373c3c9e0c2d227d18364b23457e0ef933a4add9d2345d74038364`。输出 `artifacts/p064_b_seed20261007_projected_ppo_long_cfd_20261006`。原83e08 driver、b00 148→228/主(168,228]/六窗/原阈值不变；只替换第二seed最终policy578ab956/Vecac756，CPU推理、无在线FNO/MPC。当前新物理结果未知，旧约3.90%减阻收益不能移植为本次PASS。实际PPO32768步已完成，不再训练。

## FC-E079 当前实际：第二seed20261007 PPO完成并独审，固定b00 CFD交接中

同invocation `38013204a3c94b7fb14da04b78b5fb83` 已PID0/exit0，实际32768 transitions、256 PPO epochs、512 optimizer hooks，FNO冻结且policy权重改变。独审全部24reset时钟/finite日志/74source192runtime/6输出hash通过；最低Available119953592320B，583.45s。报告 [第二seed PPO独审](docs/P064_B_SEED20261007_PPO_TERMINAL_REVIEW_20261006.md) SHA `10f0689cd3bb0200542999f94dc05d35c2444904e589154aae8685e6d6aa1d4d`；result `47dc970756bd608c8a1c3f4744c5b44a1ee87f3524737ea42c4f0fbdf9f82afd`。策略 `578ab9561d104976b16af427c8ee8c89f964ce50b4b3dbf9010e24c987ffb4ce` 已交固定b00审批，不以reward选择，不代表新物理效果。

### 第二seed历史启动观测（以下running文字为当时快照）

实际unit `fluid-control-p064-b-seed20261007-ppo-32768-20261006.service`，invocation `38013204a3c94b7fb14da04b78b5fb83`，PID2785207 active/running；独查真实transitions已到1212/32768（观测快照，实时计数由UI更新）。批准 `docs/P064_B_SEED20261007_PPO_APPROVAL_20261006.json` SHA `2a8b0bca02154ed5bf0e7b35035e01f0695c1763b183063251ee619db4fc6604`。输出 `artifacts/p064_b_seed20261007_diverse_h5_32768_ppo_20261006/payload`。12GiB/noSwap、Available50启动/22运行留20，原预算未变。

科学变化仅seed20261006→20261007；B manifest927669、32768/H5/24reset/69obs/奖励/归一化/最高精度设置保持。只保存final policy，不按reward选checkpoint、不扫seed；finite终态并独审后另批固定b00真实CFD，不自动执行。此前B在真实CFD的约3.90%减阻/18.4%升力波动改善及初始权重无减阻结论保留；FNO完整预测FAIL是另一项，当前不是修好预测或新闭环成功。

## FC-E078 当前终态：学习后权重有贡献；初始权重未获得减阻

同 `dbc0e8f47f994e7280694e9ed6714c56` 已完成800次真实反馈，exit0、容器清理完成；无当前训练/CFD任务由本对照触发。独审3200原始力文件、1600段solver与六窗口通过；两次zero前后力16000行全部列完全相同。主窗(168,228]初始策略减阻 **−0.007557%**（未满足原2%），训练后B为 **+3.895284%**；RMS比分别1.000882/.815623，bias分别1.650279%/1.137815%。原2%/1.05/10%不变，训练后早6.2偏置13.55%失败仍保留。

这支持固定seed/相位/同投影与限幅流程中学习后权重有贡献，不是RL单独归因或跨控制器最优证明。动作平方mean分别3.55615e-6/.219664，属于控制成本代理；原始moment列尚未核验转矩/功率换算，**未计算物理能耗，不能声称净节能**。预测FAIL与H25未采用保留，当前转既有链复现/交付，非全目标完成。报告 [FC-E078独审](docs/P064_INITIAL_POLICY_CFD_TERMINAL_REVIEW_20261006.md)，result SHA `48b2b37ddccb6ab4bbf7f04ec6a51c5c071a9b7ed4de504afe0d725af2cebea1`。

## FC-E078 历史启动记录（已完成，以下为当时观测）

实际unit `fluid-control-p064-initial-projected-ppo-long-cfd-20261006.service`，invocation `dbc0e8f47f994e7280694e9ed6714c56`，独查PID2315536 active/running、49/800周期、t152.9。批准 `docs/P064_INITIAL_PROJECTED_PPO_LONG_CFD_APPROVAL_20261006.json` SHA `f4e35a92a227b29fcf216018f09b3d320b382ffc392d9aad7e73616dc32c3797`；输出 `artifacts/p064_initial_projected_ppo_long_cfd_20261006`。此启动计数是当时观测，最新进度由同invocation的progress与unit共同确认，active/exited不算运行。

唯一干预是同seed20261006原初始策略权重：CPU新建/保存再加载匹配原tensor SHA `6bc539885d8c63fc922eccaba0363593555cf1b85ece5e48783d79d2ea2fa1cf`，实际R3策略包 SHA `8a99bc1ad855b6ca510950206253021accb186cab393d136dbc4935ac3cc0108`。复用原训练VecNormalize字节（identity、冻结）、同float64空间、镜像投影/单次幅值与速率限幅，148→228配对zero800周期/六窗口/原标准不变。R1 CPU序列化错误与R2空间dtype不匹配均保留，R3不弱化guard。

**尚无该对照的物理结论**，不凭近零动作预判。已有b00/b01/b07收益属于已训练B策略，H25候选退化未采用、原B完整预测FAIL均保持。本实验只检验该seed/相位/同变换下学习权重的贡献，不是RL独占因果证明或优于所有简单控制器。CPU策略反馈，无在线FNO/MPC、无PPO优化；独立终态将核raw force、零对照和旧trained B匹配，不自动重试。

## 当前摘要：真实闭环已验证；H25训练完成但预测退化，未采用

FC-E077实际同六案例H100评估已完成（inv `b34a1af84199467bad07b61758b92b49`，24.01秒，非训练/CFD），独审600端点及父B逐条同真值/动作匹配。rearCl MAE .062386→.083012，totalCd MAE .020723→.046724，四对动作差误差 .015652→.020167；六案例velocity/pressure均值及H100终点全部退化。Lead决定H25不晋级PPO，保留旧B成功控制策略；没有新训练或CFD正在由本次结果触发。详见 [独审报告](docs/P064_B_H25_QUICK_AR_TERMINAL_REVIEW_20261006.md)，result SHA `1b7bd2a2e99f9d02398df4cbcefc2d7dc5a486a02866d9a64856d0db9e9dafe0`。

- **训练**：FC-E074 scales已完成1368窗口、0参数更新；FC-E075资源探针已完成1个H25窗口/1次Adam且不保存模型。FC-E076 R1因旧H10累积器拒绝H25记录在首窗后失败，0更新，证据保留。R2同inv `9449ac16be65406eabad0515e88b6513` 已PID0/exited/exit0，实际256窗口/32参数更新；25为训练预测步数，metadata的100只是原采样窗长度。官方保存/新实例reload执行证据及独立字节/记录核验通过，不等于精度通过。
- **闭环**：原B策略在b00/b01/b07各完成800次真实CFD反馈，主物理条件通过。b07总阻力降低3.90%、后升力波动降低18.50%、均值偏置1.29%；原2%/1.05/10%不变，保留b00/b01早期偏置失败和相位非统计独立限制。PPO训练使用FNO；实际部署是CPU策略与CFD反馈，无在线FNO或MPC。新H25模型尚未用于这些闭环。
- **预测精度**：原B完整评估仍FAIL，四条旋转分支升力RMS误差不满足项目开发阈值；不能用物理收益替代预测验收。H25同六条H100短评估现已完成且退化，不自动重训或放宽门槛。当前转向既有成功链的复现指南与交付整合，不是宣称全目标完成。

R2 result SHA `557e0792eee538d8152c4997032309423a1197067c7198089768d0ddb40f5cf7`，manifest SHA `decf5f52bc0087fe07f2d3969e39604f19ea273c191ad193660c0af4c02670c0`；独审 [训练报告](docs/P064_B_H25_TRAINING_R2_TERMINAL_REVIEW_20261006.md) SHA `e72434c773433ffc3fa3f51c0cb8d8de368d0f43caae34b699378fd37d1fd0dd`。scales/probe分别见 [尺度报告](docs/P064_B_H25_SCALES_R2_TERMINAL_REVIEW_20261006.md)、[资源探针报告](docs/P064_B_H25_RESOURCE_PROBE_TERMINAL_REVIEW_20261006.md)。当前摘要优先于下文历史running/pending描述；服务active/exited不代表仍在运行。

## FC-E074 历史启动记录：B父本H10尺度计算R2，非训练

首次 scales unit `fluid-control-p064-b-h25-scales-20261006.service`（invocation `ad33d8d4c9194612b2b175a3661192aa`）在加载实际B父manifest时立即失败：旧P031来源树的`dual_fno.py`不识别P064-B kind；无optimizer、无模型保存、无尺度结果，exit1/OOMfalse且容器已清理。失败证据与输出保留，未冒充科学结果。

R2仅将父本loader替换为已在P064 official CPU proof和600点signed-H1中实际使用的审查版`83ac4e41…3d7b`，其余427个source条目、数值协议和资源合同不变。批准`docs/P064_B_H25_SCALES_R2_APPROVAL_20261006.json` SHA `cb60cf7bcc146a51f085957d6c4c68a7792fe41803fd925fed1aa7c1b3c1d77e`；实际unit `fluid-control-p064-b-h25-scales-r2-20261006.service`、invocation `88e5e31e611c45dab28dbe6c3ad11c8c` 已出现真实`window_complete`，正在计算原44 train-only的1368个H10窗口。该阶段仅重算B父本field/force loss尺度，optimizer=0且不保存模型；H25 scratch probe及32步训练均未授权、未启动。

## FC-E072 当前终态：同B策略b07真实闭环六窗口通过原标准

同inv `c45add13aeff41fe9526e835e384a52d` 已800/800、PID0/exit0，110→190真实CPU PPO/OpenFOAM反馈完成，无在线FNO。独审3200rawSHA/16000点严格网格、1600solver段、800投影单filter、全部六窗及源/资源/清理通过，统计最大差4.44e-16。主(130,190]12000点减阻3.9026772898%、rearCl RMS ratio .814994542203、mean-bias ratio .012920799096；六窗均过原2%/1.05/10%，早首6.2偏置.083038892388。结果SHA `dd579e7443c6693daef4173ed53ea2cb6836878fafff365bc12c1db8fe4ab7fc`；独审 `docs/P064_B_PROJECTED_PPO_B07_LONG_CFD_TERMINAL_REVIEW_20261006.md` SHA `6a76bbb74673dfdfdba57746471a836ce741f4855206efb5605677934133e953`。

同B策略现于b00/b01/b07三相位主窗真实受益；保留b00/b01早期10%失败。b07固定动作H5已打开，不是全新holdout，三相位非统计独立；不声称优于未运行的旧policy b07。完整代理精度FAIL与signed H1真实输入误差仍在，无全目标完成/新训练批准。minAvailable120378871808B、1097.565s、owned容器已清，资源没有被伪造为忙碌。下文running为历史。

## FC-E073 当前诊断已完成：真实输入力误差与自由递推影响均存在

signed H1 batch1同inv `76d21e62134b44c0a97d65b6ad991669` 已独审exit0：6×100真实当前场条件预测，无优化/新训练。result SHA `1eacc9219f2f608c54e6ef48d4856624af64b8ad00eb477bbd5772eff8ad61ef`；报告 `docs/P064_TEACHER_FORCED_H1_TERMINAL_REVIEW_20261006.md` SHA `2141cc0f060e79acf57ad68c530038e1814fb228ba45be2fa4a8ac01da956b36`。600行四力、时钟/动作/真值及411源码独审通过；batch1首步与原AR完全一致。pooled rear-Cl signed bias +.000880643686、MAE .045200950125，pred/truth centered RMS1.173602006843/1.198123401320；跨相位抵消不能冒充每分支偏置通过。

四条旋转分支H1尾窗RMS误差幅值均低于AR，但b01plus及b05minus在真实场输入下仍超原零基准2.5%尺度，不能把问题归因于仅长AR累积，也不构成单一因果分解。B完整精度FAIL和原物理标准保持。独审容器已清、minimumAvailable112.1743GiB；无新训练批准。b07真实CPU闭环仍按原unit独立进行，诊断结束不代表其物理结果完成。历史准备/运行记录保留。

## FC-E072 当前实际：冻结 B-policy 的预定 b07 真实闭环复验运行中

2026-10-06 14:02:02 UTC 实际启动 `fluid-control-p064-b-projected-ppo-b07-long-cfd-20261006.service`，invocation `c45add13aeff41fe9526e835e384a52d`，首查 MainPID1682828 active/running，随后实际41/800周期。批准 `docs/P064_B_PROJECTED_PPO_B07_LONG_CFD_APPROVAL_20261006.json` SHA `df4d7881226f8ebf53da3aa47ac29f32ba2f0c94d59e431dae3a84dced4b7f56`；冻结driver `c3d63d9d9114a2ec32b8a5d6a4e7a6dee5aa8e31143167fb63229951656777d5`。固定110→190、主(130,190]，原800周期/六窗口、策略、投影+单filter和配对zero完全保持。仅初相位改变，检验同冻结B-policy是否继续满足原2%减阻、1.05波动比、10%平均偏置标准。尚无b07终态或物理结论，无新训练/在线FNO。

b07在先前增强计划中预先提出，但其K1固定动作H5数据已打开；这是新的物理开发复验，不是全新holdout或独立泛化证明。旧b00/b01成功与FC-E071代理气动力窗口FAIL均保留，不因预测2.5%门失败阻止经批准的探索性闭环。早6.2启动阶段可另报15%敏感性，但原10%FAIL不改；失败不自动重训、重跑或放宽门槛。实际launch与资源记录见 `docs/P064_B_PROJECTED_PPO_B07_LONG_CFD_LAUNCH_20261006.md`。

## FC-E071 当前终态：完整预测评估已完成，气动力窗口准入仍失败

R3 同 invocation `3bded2dcb4a24f808879987962a9ef8b` 已独立确认 PID0 / exit0；不是仍在训练或评估。复用 R2 已完成两阶段，R3 完成余下六阶段，完整 receipt SHA `30d3d0746580b8423a9f626a0ebe1b76c129acab7800158f74d7d8df3d8a1799`。独审报告 `docs/P064_B_FORMAL_TERMINAL_REVIEW_20261006.md` SHA `62a6234ed0e08ab70532a5252f34e6c8b203a22c6c6abeea5a67ba2635fc8cd6` 核验35输出、411数值源、7候选文件以及原始六条力时间序列；8容器正常退出、无OOM、当前无容器，合并资源记录 minimum MemAvailable116345077760B。

科学结论为 `DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL`：窗口联合通过2/6（两条zero），四条旋转分支均未满足升力波动预测误差标准。validation10 H100 pooled Cd NRMSE K1 .00609718→B .00626447，rear-Cl MAE .04018628→.04311832；场统计因flow冻结不变。dynamic6端点误差有局部改善，但不能抵消时间窗口失败；mean-case和pooled不混比。原物理10%偏置条件与预测2.5%误差条件保持区分、均未放宽。

真实闭环并非未开展：B兼容PPO已经在b00/b01完成实际CFD反馈且主物理标准通过，但这些已观察开发相位及物理收益不代表预测模型全部验收。保持旧成功控制器、H100失败和R1/R2工程失败记录。same-six缓存诊断已完成并归档ed638be，保留batch/precision差异；下一项仅准备signed H1 600端点batch1诊断，尚未执行，不自动重训或改变目标。下文此前running/pending标题均为历史记录。

## 当前实际：完整预测精度评估 R3 同协议恢复运行

Unit `fluid-control-p064-b-formal-r3-20261006.service`，invocation `3bded2dcb4a24f808879987962a9ef8b`，实际PID1582344 active/running，当前资源日志step=dynamic6。批准 `docs/FC_P064_ARM_B_FORMAL_RESUME_R3_APPROVAL_20261006.json` SHA `19be2d6aad903ffc94b807803bd5fd0902c7ec5b7a0f0b4212423744db89cb56`；新独占输出 `artifacts/fcp064_arm_b_formal_resume_r3_20261006`。只修复P064候选CLI身份路由，复用R2 precision/validation10的9个SHA核验文件，仅执行余下6阶段；R1/R2失败及原输出保留，新receipt区分来源。runner `3bb215f93f5d6d468f8b22267b5f1fb485516f865c45d32723fe33b3fe56ec84`，独立5CPUtests通过。没有重训或重算validation10，尚无完整科学判定；b00/b01物理主窗口已独审通过与预测模型门槛保持区分。实际allocator .15、Available50/22保护不变。

## FC-E070 最新终态：同B-policy固定b01复验主标准通过，非新holdout

同inv `432c12de32b0444d9a6f6626e12616d1` 已真实exit0，800周期从130→210完成。独立3200rawSHA、1600solver段、800投影单filter及六窗重算通过，最大统计差4.44e-16。主(150,210]减阻3.9275159299%、rearCl centered RMS ratio .815727792287、mean-bias ratio .027294976565，原2%/1.05/.10标准通过；早首6.2 bias .127825261仍FAIL10%。结果SHA `0be19e0dfdf8df4d60e2f5040673f2a3ec25cbadb133548671ce31f061e75c88`，报告 `docs/P064_B_PROJECTED_PPO_B01_LONG_CFD_TERMINAL_REVIEW_20261006.md` SHA `de7a3f7fa89d50dad190272028f16ed0ea23214ac52cb23323f42c48df9945d3`。

同一新B-policy现于b00/b01两个已观察初相位主窗通过；b00是训练轨迹，b01已用于开发且历史暴露，不能称新holdout/独立泛化。新旧b01 zero全部原始forces逐值相同，新减阻比旧K1-policy仅+.00387868305百分点，不显著优势。全窗lift峰值1.704317136仍高于zero1.647302403，保留transient代价。旧成功policy不替换，formal/H100结论仍须独立终态，不从物理成功推断模型准入。下文live记录为历史。

## 当前纠正：P064-B 完整评估 R2 工程失败，validation10已完成；没有完整科学判定

同invocation `01806bdc150841f7b9efd04360a441f2` 已在12:53:10 UTC退出1/PID0：后续validation_diagnostic的CLI未接受P064-B kind。不是仍在运行，也不是完整科学gate FAIL。已完成validation10的1240端点独立复算通过：B H1 rearCl/Cd MAE .0377850/.00878357，对照K1 .0202176/.00766464；H100 .0431183/.0112626，对照.0401863/.0109280。场误差统计逐值不变。H100真正pooled Cd NRMSE为K1 .00609718→B .00626447；summary的mean-case为.00580968→.00603306，两种聚合不可混比。后续force-window/fullgate尚未完成，无准入结论。新独审 `docs/P064_B_FORMAL_R2_PARTIAL_TERMINAL_REVIEW_20261006.md` SHA `2ead80d6ecbca6485bab8bca20183a3aa05a71e568af39565d11086d677638ad`。R2证据保留；身份CLI最小修复与新输出续跑另行审查，不重训或重算已完成validation10，不修改科学门槛。下方此前running段落仅为历史。

## 2026-10-06 — 区分预测精度要求与实际控制要求

源码与历史终态报告复核后，当前所称的 K1“长时域评估失败”应明确为 **H100 尾部62个采样点的气动力统计预测要求未满足**，不能笼统写成速度/压力流场阈值失败。K1 validation10 的 H100 Cd NRMSE .00609718、动作差分 Cd MAE .0191297 均通过对应要求；速度 relative L2 .0435911 是报告指标。六条力窗口仅1条联合通过，旋转分支的升力波动 RMS 预测误差约 .068–.122。证据：`docs/FC_P026_K1_FORMAL_TERMINAL_REVIEW_20261006.md`。

真实 CFD 控制的减阻≥2%、升力波动比≤1.05、平均升力偏置≤基准波动10%，与代理预测的力统计误差要求不是同一件事。后者1%/2.5%的数值要求是在2026-10-04制定、10-05补齐计算协议的项目开发标准，不是原始文档已经量化的全部用户要求；来源见 `docs/CANONICAL_SURROGATE_PROTOCOL_COMPLETION_20261005.md` 和 `scripts/audit_dynamic_fno_development_gates.py`。本次澄清不修改任何指标，不把未通过改为通过。

B 的 flow 权重冻结，不能据此宣称流场预测改善；aero 权重有变化，所以正在执行的完整评估仍能检验气动力误差及原工况表现是否改善。真实闭环已经开展，不因预测评估尚未完成而被描述为“未开展”。同时，B 的短时 H1 力误差仍高于持力基线，真实控制收益不能代替模型精度结论。待当前完整评估和 b01 物理验证终态后再批准下一项实验；若力窗口仍失败，优先复用已有自由递推结果，与相同模型/动作/时刻的逐步真实流场条件力预测比较，以区分受力预测本身的误差与预测流场误差的影响。该诊断目前仅为条件性建议，尚未执行或批准。

## 并行实际：P064-B 原完整formal评估 R2 已启动，尚无验收结果

资源补充：实际 evaluator/forcewindow 的 torch allocator fraction 为 .15（约18.25GiB），外层 `.06` 只是启动余量核算、不是强制GPU分配上限。Lead在核源码及实际Available109.23GiB后明确批准同R2继续，保留72GiB/noSwap与运行22GiB Available保护；不改源码、不重启、不改科学门槛。

2026-10-06 12:41:45 UTC，unit `fluid-control-p064-b-formal-r2-20261006.service`，invocation `01806bdc150841f7b9efd04360a441f2`，实际 active/running。批准 `docs/FC_P064_ARM_B_FORMAL_APPROVAL_20261006.json` SHA `cd58fd47e991ec6dac200bd82d414347f72b778ea78415377427e430dfd47478`；输出 `artifacts/fcp064_arm_b_formal_20261006`。这是原完整科学协议评估，不是训练；precision阶段已结束，validation10进入CUDA/PhysicsNeMo初始化，尚无完整数值结果，不凭进程或GPU利用率声称forward/通过。R1 invocation `6d7cea57ea7740c991223448dfc45e0a` 因错误cwd相对路径在打开执行源码/容器/GPU前exit2，保留失败；R2同批准仅改正确cwd与绝对路径。b01 CPU真实CFD同432c12继续运行，原H100失败历史未被覆盖。实际launch报告 `docs/FC_P064_ARM_B_FORMAL_LAUNCH_20261006.md` SHA `ac3dc4b3262a943128277b4acf3f53e014dfd3d2a604f99968d3a3ce36c85cda`。

## 当前实际：同 B policy 的 b01 配对800周期验证已启动

2026-10-06 12:37:56 UTC，unit `fluid-control-p064-b-projected-ppo-b01-long-cfd-20261006.service`，invocation `432c12de32b0444d9a6f6626e12616d1` 已实际 running，首查14/800周期。固定130→210，主窗口(150,210]，同B policy、镜像投影及单次动作限制，原物理标准不变。批准 `docs/P064_B_PROJECTED_PPO_B01_LONG_CFD_APPROVAL_20261006.json` SHA `3a19e326ebfd37df24060b8b5717b5af08b405ed63165e4034974aeb7b0abcb6`；immutable driver `4b8fa43f8ac020521ae8cde1047b6ec2f80d35ec0512bc7606d1050835010619`。CPU真实CFD反馈，无新训练、无在线FNO，尚无物理结论。b01是已打开开发相位，不是全新独立测试；不能继承旧policy结果。此前b00已独审通过原主窗口标准，H100失败仍保留。

## FC-E069 最新终态：B新policy在b00保持物理收益，非显著提升或独立泛化

同inv `3a078c62ed9e4f7b8876f0f166bdb510` 已exit0，800周期真实paired CFD完成。独立复算3200原始文件SHA、1600solver段、全部动作及六窗口：主(168,228]减阻3.8952838833%、rearCl centered RMS ratio .815623043405、mean-bias ratio .011378146878，原2%/1.05/.10标准通过。早首6.2窗口bias .135464513仍未通过10%；full80峰值1.679784159高于zero1.647306236，不能宣称全时域峰值改善。结果SHA `8b31091d5e69edfbfd5ea78ba99dd7709623e6eeb0bd4f13c54e984b7fc28907`；独审 `docs/P064_B_PROJECTED_PPO_LONG_CFD_TERMINAL_REVIEW_20261006.md` SHA `7b453d9c529d9d5c52988608c89d61050204510d41549cbb2be620fcdcebbe04`。

新旧zero原始forces逐值完全相同；相对旧成功projected K1-policy主减阻仅增加.0033039437百分点，不是显著control提升。B训练包含b00，此次属in-sample物理确认，不能继承旧policy的b01/b03证据。旧成功policy保持不变；下一项仅准备同新policy固定b01复验，须独立批准。P064 formal/H100未因本结果获得准入。

B官方CPU双模型reload R3已实际exit0、无forward/GPU，审计SHA976e0201…51c36d、重载receipt497e1164…fad87c；R1/R2只读缓存路径失败均保留，R3仅临时tmpfs修复。新receipt权限修复前后SHA不变。当前下文的“running”段落为历史启动记录，不代表仍在运行。

## FC-P064 当前实际执行：B fresh PPO 已独审终态，paired800 CFD 已启动

B fresh PPO 同 unit `fluid-control-p064-b-ppo-32768-20261006.service` / invocation `f613395cbf1140549dc60e7b046e0f6b` 已 PID0、normal exit0。完成 32768 timesteps、256 PPO updates、512 optimizer records；FNO tensor 字节不变。结果/policy/Vec SHA 分别为 `3c70e21327baae98f682fc0982ca3c3910cf6d1902f3d62175f980fd685817b3` / `f764463983355779efff8d1b1994cfaf560ab7274d54b014d34a1f084b4b307e` / `8c07ef15bd41a8981f2ec0d241c85092b643ca740fea9f44866eecceac1197ad`。独审 `docs/P064_B_PPO_TERMINAL_REVIEW_20261006.md` SHA `0cc1494286f85b930b43b1a11713ae6ac3719d8ff599cfa0780a8d9fe71b0d65`；这是工程训练完成，不是 CFD 成功或正式准入。

Lead 随后实际启动唯一一次 B-policy 投影 paired800 CFD：unit `fluid-control-p064-b-projected-ppo-long-cfd-20261006.service`，invocation `3a078c62ed9e4f7b8876f0f166bdb510`，输出 `artifacts/p064_b_projected_ppo_long_cfd_20261006`；批准 `docs/P064_B_PROJECTED_PPO_LONG_CFD_APPROVAL_20261006.json` SHA `5fc8ab36e69e7e6ea27ed7c4d60ae207bc67be3c9513e89800cedccf46970a99`。归档时同 handle active/running，尚无物理终态结论。此前一次缺少 `--execute` 的 CLI probe 在进入执行体前按预期拒绝，没有启动 CFD；只有上述 invocation 是实际执行。旧成功 policy 与三相位结果保持不变。

## FC-P064 实际探索链：B 候选 fresh PPO 正在运行

Lead 已在独立审查通过后启动 B 候选的全新 32768-step PPO：unit `fluid-control-p064-b-ppo-32768-20261006.service`，invocation `f613395cbf1140549dc60e7b046e0f6b`，输出 `artifacts/p064_b_diverse_h5_32768_ppo_20261006/payload`。批准 `docs/P064_B_PPO_APPROVAL_20261006.json` SHA `ae327fee310bad562aceef35029595d20c9a3421d5d5be3dc3b68bb82649e9fe`。运行使用冻结的完整 P064 runtime/import 闭包和实际 B manifest/result/独审 receipt；启动不等于终态成功或 CFD 收益。

该 PPO 保持旧成功链的 32768 steps、24 fixed resets、H5、69D observation、62-step reward、seed 和 fresh initialization；只是显式换入 P064-B FNO 候选。旧三相位成功 policy 保持不变，不被覆盖。后续投影 paired800 CFD 仍是 preparation-only，必须等本次 PPO 终态 policy/VecNormalize/result 的实际 SHA 和独立复核后另行批准；不得预填哈希。P064-B 尚未 formal admit，旧 H100/full-formal 失败和训练诊断 retention 代价继续保留。

## FC-P064 最新终态：B局部开发支持，有retention代价；原成功policy不变

A/B均已完成32updates/256windows并通过独立CPU工程检查。A/B同协议开发评估也已终态独审：各16NPZ/80端点，所有起点/相位/pooled统计重算一致。B pooled H1 rearCl MAE .138998317、totalCd MAE .038065374，相对A .156116880/.039952166严格降低10.9652%/4.72263%，支持预声明的局部开发比较；不是正式模型接受。H1仍差于持力，B H5 Cd比K1差.7884%，b01 H5 Cd比A差.6848%。K1/A/B所有预测流场数组逐值完全相同，flow冻结没有改善速度/压力。

B原6训练窗H1/AR目标由.00348834/.00880538升至.00397686/.00894620；真state尾RMS误差改善但AR尾RMS误差恶化。它们是保存的训练诊断，不是新test，不加事后门槛。比较报告 `docs/P064_AB_DEVELOPMENT_COMPARISON_REVIEW_20261006.md` SHA `2fa7e5d71e4b163bfff2261ec2b38ef43c1c5aa75acc7cea7d47c1f3296c225a`；A结果 `c8b0242658a101120603514e6d2e5076c827c518965c92470810fe9693840fe6`，B `47e7d4c6931fadc62730790500bb9a8a07f44792d1bef82c900d10c36f9a8665`。

B训练450ef57c…已正常exit0，独审receipt `d78f87d041fd907c50ad6b2ca8880498bf8f5ad80e5b6916270ec585105b2915`，28Adam全step32、64b00固定窗及冻结bias/flow一致。B开发6613b11d…正常exit0，最低Available121369776128B。下一阶段仅准备并另批B fresh32768 PPO→投影paired800真实CFD探索，以检验是否转移为物理收益；尚未执行这条新policy链、不替换旧成功policy。原三相位物理成果、10%约束及K1 H100 FAIL保持，B尚未正式准入。下方运行描述均为保留历史。

## FC-P064 最新：A 工程终态已独审，B 已实际启动；尚无候选精度结论

A R2 同 invocation `50de1d8b43ce42ac923752fad76ca4d9` 已 PID0 / normal exit0。独立检查434冻结源码、256条实际采样日志、32×8窗口记录、目标均值与clip、来源/归一化/候选文件；CPU weights-only 检查28个Adam状态均step32且有限，两个冻结bias逐值等于K1，flow文件保持原字节。结果SHA `03951fee3c1ba66ae48d451fe35aeb7735f092e40ef0b76deff90738c9f8e6a1`；独审receipt SHA `2d51c4f84b53fa9b29748c77200cac279ab5a3a4b8ebb5aa6309b4ae9a45f52a`。训练观察最低Available106.844551GiB，cgroup峰值9884188672B。官方fresh reload为producer逐role实际CPU重载；独审未新建模型/forward，不冒称新dual-loader整链实际重载。报告 `docs/FC_P064_ARM_A_TERMINAL_ENGINEERING_REVIEW_20261006.md`。

Root另行批准并实际启动B：unit `fluid-control-fcp064-aero-arm-b-20261006.service` / invocation `450ef57c25c14ec38e722cbd597ffb50`，本次归档独立观测PID528728 / active-running。批准 `docs/FC_P064_ARM_B_TRAINING_APPROVAL_20261006.json` SHA `a1e79d108f5067027742f08f3e04b2d73cb059286e2bd433e53f4e0d51247b29`。B从同K1/fresh Adam独立开始，不续训A；固定32updates/256windows中64窗来自b00，仅运行，不宣称成功。

A开发评估PENDING首次生成因434训练闭包没有旧selector而在写pending前失败，无GPU评估；原失败保留，正独审source-pinned旧selector导入最小修正。A工程完成不是科学准入，A/B尚无同协议开发精度比较；不覆盖原K1 H100失败，不阻断已完成三相位真实物理闭环，也不放宽10%偏置标准。下方旧运行/未批准描述为历史阶段记录。

## FC-P064 当前工程里程碑：A 臂 R1 在训练前失败，R2 同源恢复并持续运行

已冻结并推送 P064 训练、显式 A/B 候选身份、官方 Reader/DataPipe 适配与 CPU 合同测试，代码提交 `977a027d51c3ae03ade858deae0030d194a79eb8`。A 臂首次 unit `fluid-control-fcp064-aero-arm-a-20261006.service` / invocation `cfc40285f8ec49fdba9a99defe2960cc` 在第一个 CUDA 迁移处退出1，完成 **0/32 optimizer updates、0/256 windows**，没有候选输出；批准 `docs/FC_P064_ARM_A_TRAINING_APPROVAL_20261006.json` SHA `c1872be6e816a2f058a111084594882ebfd9d3f0222e642ba58bfba71487ee1e`。这是启动失败，不是科学负结果。

随后一次获批的精确 53 文件 `POSIX_FADV_DONTNEED` 建议任务 invocation `1364cefc6aa14d75a6574dd25931f154` 退出0；没有 `drop_caches`、sysctl、权限或数据修改。它之后的同源 R2 unit `fluid-control-fcp064-aero-arm-a-r2-20261006.service` / invocation `50de1d8b43ce42ac923752fad76ca4d9` 已通过 CUDA 初始化并实际完成至少 **9/32 updates、72/256 windows**；批准 SHA `773daa7329a46930b532586d884502d0c4170ea2c3a689ba73b2fb3d5da8b568`。R2 仍在同一 handle 运行，尚无终态候选、科学指标或 CSV；只能记录“精确建议后 R2 通过初始化并推进”，不能把缓存建议写成 R1 根因已证实。

B 臂 pending `docs/FC_P064_ARM_B_TRAINING_PENDING_20261006.json` SHA `d55a43ae5a68a23296508a555896bb2d8a0c980da558fff0ce13f1974e9f7cba` 的完整 `CUDA_VISIBLE_DEVICES=` dry-run argv 已由 Root 实际通过，但 `execution_authorized=false`，未启动。A/B 都从同一个 K1 父本重新开始并使用 fresh Adam；B 不续训 A，只能在 A 实际终态检查后另行批准。

## FC-E067 最新终态独审：K1 在两开发相位的 H1 力预测均劣于持力基线

同a39f8106任务已退出0且独立核验全部16NPZ/80端点、每起点/相位/pooled统计、378source/192runtime/5inputs。结果SHA `9ea3e0e781e76265bbc65ea52d6fec92ebe5b5cb7a93d91c3c9addb454f7c4de`；独审 `docs/P064_K1_DEVELOPMENT_TERMINAL_REVIEW_20261006.md` SHA `a15a6699cf20d0d3b76a8569357bd3580e1bb11fe232207164bff5cc08ab2fef`。最低Available120994258944B、零优化器/模型不变；独审只重算保存数组，没有再次加载模型。

pooled H1/H5 velocity relativeL2 .010316/.043040、pressure .032042/.137006；rearCl MAE .158871/.179850，对应持力 .090334/.439982；totalCd MAE .039630/.030522，对应持力 .020318/.101058。力误差是系数绝对误差，不是百分比。b01与b03的H1两项力平均都劣于持力；H5优于逐渐陈旧的持力仍不代表普遍准确。该基线仅用于随后同协议P064 A/B开发比较，不是新测试/在线FNO/新控制试验；原三相位物理收益与H100失败不变。

## FC-E067 当前实际执行：K1 开发集 H1–H5 基线已退出0，数值独审中

Lead实际启动 `fluid-control-p064-k1-development-h1-h5-20261006.service`，invocation `a39f8106aa9b4559b1fb1587e7a8e38c`（启动PID425486）；最新同handle PID0/active-exited/ExecMainStatus0。批准SHA `05891a360ffaa17d61e6854a8dce632d818a3ab1ec5b4a616244d4abb6a85b79`；不可变执行源码 `e86ef8ea3c55be63287b0f9d5e9e0cf0034e7fc6959df25e53e49c0fe23d90a8`，canonical精确归档，6CPU源/合同测试通过0.06s。

固定已打开b01/b03开发96帧，16起点×H1–H5=80端点；原K1、原归一化、官方Reader、post-load最高精度/noTF32，12GiB/noSwap、GPU6GiB allocator、Available50/22、600/630s。输出 `artifacts/p064_k1_development_h1_h5_20261006`。本次仅冻结基线推理，无训练/新CFD/候选选择；程序退出0不代替数组与指标独审，精度结论待closed_loop独立复核。A/B使用同协议但尚未执行，不把已有开发相位称为新测试；H100失败不变。

## FC-E065 最新终态：b00完整训练轨迹及专用视图已独审

原f5ee31已PID0/exited/exit0。转换结果SHA `f24f2fbc8b283c0781b2a01189d291b43da0203e9ede28208ec575173bbe0bd9`，801帧官方Reader/HDF全量、真实progress时间/动作/四力、1614原源SHA与旧48NPZ一致；753新packet已删除，未独立重采样，不声称全801NPZ复核。16export容器已清理，最低Available121431912448B。

`artifacts/b00_controlled_train_dataset_view_20261006` 专用hardlink视图manifest SHA `97a82e2a157b87ecf9b9ea626ad079852a8d29ba89628ca5034ef99545929a5f`，保留原归一化；实际官方DataPipe有701个H100窗，首0→100/末700→800数据、101动作和100四力targets exact。源inode权限不改，要求只读使用而非虚构OS只读mount。报告 `docs/B00_CONTROLLED_TRAIN_CONVERSION_TERMINAL_REVIEW_20261006.md` SHA `c265cc7254d1cd423626bce84d0b064e90a2686bff4c651021017e09952657eb`；审核R1仅schema解析失败且保留，R2成功未重跑转换。数据准备不自动批准P064 GPU训练，不影响三相位物理成功、10%原标准或K1 H100失败。

## FC-E066 最新实际终态：b01/b03 开发数据96帧转换已独立验证

unit `fluid-control-development-b01-b03-h5-conversion-20261006.service` / invocation `8ac389027a954b79bc1d766f89029a6b` PID0/exit0；结果SHA `1da29262fa8cbfa7c41687439ad34cc1967ce8c7d9faa35c435668705ca9351f`。216原源文件、16 HDF、96帧state/mask/time/grid及实际动作/四力端点独立核对通过，两个export容器已清理，最低Available121511297024B。官方Reader真实复读完成；时间保留VTK float32误差≤6.104e-6，不声称十进制精确。

独审 `docs/DEVELOPMENT_PHASE_H5_CONVERSION_TERMINAL_REVIEW_20261006.md` SHA `aa1417d9d6d05d282e2b00606985c31b9070bdb31b45378d6dba8dcc746e1c94`。旧progress16/16仍标转换中不是终态权威，actualunit+result已确认完成。仅已打开开发数据转换，无新CFD/推理/训练或科学准入。b00全801帧另项转换仍由同f5ee31任务执行，后续P064等预算A/B及开发评估需单独审批；三相位物理成功与H100失败均保留。

## FC-E066 实际运行：b01/b03 固定96帧开发评估输入转换

Root于2026-10-06 10:27:34UTC启动 `fluid-control-development-b01-b03-h5-conversion-20261006.service`，invocation `8ac389027a954b79bc1d766f89029a6b`，PID379249，归档时独立查询同handle active/running。固定b01/base130和b03/base144，各8个机械起点0,100,…700及后5帧，仅controlled48帧/phase，共96帧/16 mini-HDF/80未来端点。两相位均是已打开development，不是untouched test或统计独立泛化样本；不混入b00训练数据。

批准 `docs/DEVELOPMENT_PHASE_H5_CONVERSION_APPROVAL_20261006.json` SHA `c82e4a865d0d6e0927faee026baa84ebca2fcabb0707daa00f988174349dd5a3`；执行driver `cb7fdb83e620903be89c85f38286a0be0afbb4b95792adb7b7179065f85e8a9c`，selector `26c085d1da91c3452fc514cda9c06332afe75e600be7770f8d476c1029986cb0`，输出 `artifacts/development_b01_b03_controlled_h5_conversion_20261006`。复用R2原样Curator/Reader/导出与cleanup，12GiB/noSwap/CPU1/900s、Available50启动22运行保留20；root-owned导出scratch保留，不改权限。归档时无终态结果，不把启动或CPU tests当成功；无模型/训练/CFD solver。9canonical CPUtests PASS0.09s，终态由非作者recovery独审后另记，CSV不填运行中指标。此前FC-E065 b00全轨迹转换是独立并行任务，不互相代替。

## FC-E065 当前实际任务：b00 整轨训练 HDF 转换中（不是训练或新 CFD）

Root 于10:17:37UTC启动 `fluid-control-b00-controlled-train-conversion-20261006.service`，invocation `f5ee31dd92624f0980a509084de9c756`，独立观测同handle PID348063/active-running、59/801帧。进度来自已写入计数，不代表最终官方Reader验证完成。输入为已完成投影PPO b00真实CFD的801帧；仅整轨train，48已审控制帧复用、753新转换，不混zero轨迹，不重新拟合归一化。

批准SHA `4fa7192e13bf7ad3a141bffb483710e2400fd8ee243caa60a6e67ab695927686`；实际immutable源码 `f96c882a90e7ecaf4a2f8a5fc327764909ab42b4e6bbb11cde8e99205488b075`。12GiB/noSwap/CPU1、Available50/22GiB、3600秒上限，保留root-owned批次导出供诊断，计划磁盘10GiB/启动余量20GiB。无模型/GPU/优化器/solver推进；原H100失败和已有三相位物理结果不变。待实际终态再验证HDF与源文件，运行中不登记科学metrics。

## FC-E064 最新缓存诊断：动作变化已有覆盖，但不能证明晚期受控状态覆盖

仅小JSON分析已完成，unit `fluid-control-train-cache-coverage-20261006.service` / invocation `08658ea0d6a5435a847df4de1223d21f` PID0/exit0。固定原P027全44条、origin51、前5步及H1/AR/持力三流，独立复算误差与分组差异为0；无新HDF/模型/推理/训练/CFD。反转18条中16来自train16，恒定21条中20来自base，变化不反转5条全来自train8；分层与family强混杂，不构成因果识别。反转组五步rearClMAE .020870(H1)/.023243(AR)，低于恒定 .038894/.045342，不能声称动作反转简单导致更大误差。

三family总轨迹虽长80/20/12.8D-U，但本缓存全部仅elapsed约5.1→5.6D-U；时长不代表晚期状态覆盖，也没有测量OOD。P027 high/TF32与FC-E063 highest/noTF32不精度匹配，stored-HDF插值标签不是新增在线因果证据。结果SHA `34ec16db0540c0dabb2f44caf3240ae386893ea85bd28bc1436e8820212b039d`，独审 `docs/TRAIN_CACHE_COVERAGE_TERMINAL_REVIEW_20261006.md` SHA `582f5d04ef7fcab10349a858cd4c618ddc98e44bcf9eb99da84de552d48a8754`。实际1GiB/noSwap/1CPU限制；MemoryPeak未保存，不写成0。源码精确归档1fbd532c…8317，canonical8CPUtests PASS0.03s。下一长时受控数据研究仍按Lead另行批准；本诊断不自动授权新训练，不改变三相位物理结果与H100 FAIL。

## FC-E063 最新预测审查：已执行动作条件回放完成，受控轨迹 H1 力预测劣于 persistence

实际推理 invocation `627cb6b8b59a40aca3bb9159fb617eb3` 已 PID0/exit0；16 个 NPZ、80 个 H1–H5 端点独立复算通过。受控分支 H1/H5 rear-Cl MAE 为 **.186710/.149926**（persistence **.088235/.423392**），total-Cd MAE 为 **.048653/.035954**（persistence **.017126/.084684**）；零控制 H1/H5 rear-Cl MAE 仅 **.013311/.010806**。受控 H1 的 rear-Cl/Cd 仅 3/8、1/8 起点优于 persistence；H5 为 8/8、7/8，但不等于普遍准确。起点0误差较小不能代表后续受控状态，合并两分支会掩盖差异。

结果 SHA `247af0405d9e622f0b3b3b5dbc64e46c20890439fcd5216682b8d973957fd00d`；独审 `docs/PROJECTED_POLICY_H1_H5_INFERENCE_TERMINAL_REVIEW_20261006.md` SHA `197b385617420e5f0e9cb7c8dfb25d957e98f85ea93f280bb2d6c0effc898faa`。378 源文件、192 runtime、5 输入身份通过，最低 sampled MemAvailable 121518190592B；模型不变、无优化器/新 CFD。源码按已执行字节归档，canonical 7 CPU tests PASS，仅工程覆盖。此为给定真实已实现未来命令的离线回放，不是在线 FNO/MPC，不改变三个相位的真实物理闭环结果，也不覆盖 H100 FAIL。下一项仅建议另审 train-only 状态/动作历程诊断，未自动批准训练或调参；FC-E062 的首轮转换失败与历史记录全部保留。

## 当前：真实策略闭环已完成，b00 / b01 / b03 三个观测相位 primary 均通过原标准

FC-E061 b03 已终态，不再运行训练或该 CFD：同一 invocation `47612677a9f64dfc968917fada5e9ba8`，PID0 / exit0，完成 800 个真实策略反馈周期。独立复核 3200 个原始受力文件哈希、全部六窗口、800 次投影与单次动作限幅、1600 个干净求解段及容器清理。Primary **(164,224]** 的 12000 点结果为减阻 **3.89714021%**、rear-Cl centered RMS 降低 **18.51412210%**、均值偏置/配对零旋转 RMS **1.68885808%**；原 ≥2% / ≤1.05 / ≤10% 三项均通过，无阈值放宽。本次六窗口均通过，不能据此改写 b00/b01 的早期窗口失败。

结果 SHA `d4d755faf913d393ca1466ee74614c0fb11f33b8f662a76a1cb7e4de3dd17a2f`；报告 `docs/EXPLORATORY_PROJECTED_32768_PPO_B03_LONG_CFD_TERMINAL_REVIEW_20261006.md` SHA `0d48a7e914ec82ad682d374e6531f2aab6a305aa81dad2dde6cd5c23dca48a0f`。这是冻结 FNO 环境训练出的 PPO 经同一镜像对称处理后，在真实 OpenFOAM 反馈中取得的结果，不是训练更新次数或离线模型预测替代闭环。三个相位来自同一配置/极限环，不是统计独立样本，不外推其他 Re、几何或所有相位；b03 固定动作 H5 数据此前已打开。K1 H100 FAIL 和早期失败继续保留。

FC-E062 下述转换记录保持原状；后续独立批准的 80 端点回顾性推理 invocation `627cb6b8b59a40aca3bb9159fb617eb3` 已 exit0，数值独立审查另行记录，不能将进程成功当作精度结论。以下“运行中/尚未执行”描述为对应阶段的历史记录。

## FC-E062 工程终态：固定 96 帧已转换，尚未执行 FNO 推理

只读回放输入转换已完成并经独立复核，但这不是模型精度或物理控制结果。首次 unit `fluid-control-project-policy-h1-h5-conversion-20261006.service` / invocation `45e422939f5a477a85f359cd60c5047e` 在第一个采样视图因 Linux `protected_hardlinks=1` 拒绝对 root:root 0644 VTU 建 hardlink 而 exit1；未产生成功 packet、未加载模型、未运行 CFD，失败输出和 journal 保留。R2 只把 `os.link` 换为 `shutil.copy2`，并在采样前后核对字节 SHA，无 chmod/chown/sysctl 或数值协议变化。

R2 unit `fluid-control-project-policy-h1-h5-conversion-r2-20261006.service` / invocation `60aaab28df8d46508bdcb483a2c63074` 已 PID0/exit0。结果 `artifacts/projected_policy_h1_h5_conversion_20261006_r2/result.json` SHA `a22c3aa67509e9b3a342071398ae85da2ce4e07c74a3cbbd87e2a493c2b248bf`；独立报告 `docs/PROJECTED_POLICY_H1_H5_CONVERSION_R2_REVIEW_20261006.md` SHA `4172fb77b3e0afe774e3595537f3217e211102ef4fb7e53a3d5ae5e08b89781e`。固定 b00 投影策略轨迹的 mpc/zero 两分支、8 个机械起点、每个 6 帧，共 96 packet、16 个 official HDF5Reader mini-HDF 和 80 个未来端点；216 个源清单文件、动作/四力时钟、共同 mask/x/y、16 个 HDF 哈希均通过复核。VTK 时间保留 float32 表示，最大名义网格偏差 `6.103515630684342e-6`，不得声称十进制时刻精确相等。

两个导出容器 exit0/OOMfalse 后均已删除；1192 条资源记录最低 MemAvailable `122120433664` 字节。结果明确 `model_loaded=false`、`optimizer_steps=0`、`cfd_executed=false`、`scientific_admission=false`。下一步只能在独立批准下用冻结 K1 做“给定已实现未来动作”的回顾性 H1–H5 推理；在线起点并不知道未来策略动作，转换成功不改变 K1 H100 FAIL，也不构成新的闭环收益。

## FC-E061 启动记录（已被上方终态结论更新）：固定 b03 的第三次物理策略确认

实际 unit `fluid-control-exploratory-projected-32768-ppo-b03-long-cfd-20261006.service`，invocation `47612677a9f64dfc968917fada5e9ba8`，PID4013554，于2026-10-06 09:30:16UTC启动，独立查询active/running。仅初相位改为预声明b03/restart144，800周期至224，primary(164,224]；同一冻结32768策略、反射投影、一次actionfilter和原2%/1.05/10%标准。当前没有终态物理结果，不得将启动或此前b00/b01成功写成本次成功。

批准SHA `3ca5531815c48cff59fd1ca0d96e40e2305402435cb2e28d2adb26eeaf9328a6`；不可变driverSHA `6516456f07d765728055f58036bed97a5e37036f205c4d1ee2bf2456be8cc3b0`；输出 `artifacts/exploratory_projected_32768_ppo_b03_long_cfd_20261006`。b03固定动作H5 payload已经在FC-E060打开，故这是新的物理policy trial而非普遍未见相位；不声称统计独立。没有新训练/调参/MPC，H100 FAIL继续保留。8GiB controller、2×8GiB noSwap solver、Available50/22与3600/3750/120秒边界不变；不自动重试。

## 前阶段结论：两相位 primary 物理约束通过，短时预测已测量，H100 FAIL仍保留

Lead已确认投影策略b00与b01两次固定80D/U配对复验的primary原物理约束均通过；FC-E059 b01独立3200原始文件hash、六窗口重算、800动作投影/filter及容器清理已完成。Primary (150,210]减阻3.92363725%、rearCl centered RMS比0.815785507、均值偏置比0.027300222；早期first6.2偏置0.127807798仍未过10%。不能外推所有窗口、所有相位或Re；历史validation b01也不是新独立统计样本。当前训练和这两次CFD都已完成，以下running标题保留为历史。

b01结果SHA `961e1bc3ccb7a9f9dae4b54e9f8233c906507c794cff9a497d806391e0fc5c37`；报告 `docs/EXPLORATORY_PROJECTED_32768_PPO_B01_LONG_CFD_TERMINAL_REVIEW_20261006.md` SHA `1b59fdfdb9d698fd2c0622085c19bf4d59a670070ea7434cb469ac19c72a297b`。同一9ef43959e065431490bd4725fa8fb7fe已PID0/exit0，800周期1125.32秒（包含配对CFD/IO/清理，平均1.40665秒/周期）；是因果在线模拟闭环，不是已证物理实时控制。最低Available117067710464字节。原遗漏--execute的首个CLI失败保留；不放宽物理阈值，K1 H100 FAIL不变。

FC-E060前瞻冻结测试H1–H5已完成：同一固定K1、b03/b07十条sealed轨迹、32起点/条，1600端点，无失败或非有限数。H1→H5 pooled速度relativeL2为0.22427%→0.95415%，ROI中心化压力0.64234%→2.68274%，rearCl MAE0.01918→0.02212，总Cd MAE0.00708→0.01063。官方Reader实际使用、post-load highest/noTF32明确记录；独立复算已通过。这是短时固定动作预测证据，不覆盖K1 H100正式FAIL，不是新的训练或准入阈值放宽。

实际unit `fluid-control-short-horizon-frozen-confirmation-20261006` / `7a887ed792ec4d60a429f4a7a3660b3a` exit0。结果SHA `77ab4fb85f238d1e76b6e5a20c18f8992182d24a82b38d7b4b5a52483f9fce20`；报告 `docs/SHORT_HORIZON_FROZEN_CONFIRMATION_TERMINAL_REVIEW_20261006.md`。原seal/失败记录不改，不据最终测试自动调参；物理10%与20%敏感性也不改变预测误差。

## FC-E059 正在执行：固定 b01 相位的同策略、同投影复验

FC-E058 的 b00 primary 三约束经独立原始数据复核通过后，Lead 批准固定 b01 validation 相位复验；这不是按表现挑选相位，也不是新的独立最终测试。实际 r2 unit `fluid-control-exploratory-projected-32768-ppo-b01-long-cfd-r2-20261006.service`，invocation `9ef43959e065431490bd4725fa8fb7fe`，初始 PID `3509955`，已核 active/running。输出 `artifacts/exploratory_projected_32768_ppo_b01_long_cfd_20261006`。

冻结策略、反射投影、单次 action filter、800 周期和六个相对窗口不变；唯一预声明差异是实际 b01 restart 130，推进至 210，primary `(150,210]`。批准 SHA `790bb12fae2f5df729efda98ef59e5d99f75521d220cb9b39e8caf04808e4c59`；不可变 driver SHA `8b653f43bd1ffc69b6285dd10523199898d88d74aebe4279f65a66fb4807c741`。首次 unit invocation `dfa8ba1412e34522a1ed1385e28b2df7` 因启动命令漏传 `--execute` 在执行体、模型和 CFD 前 exit1；无输出生成。Root 核验后只批准以新 r2 unit 增加该必需参数的工程重试，失败日志保留。

当前只有运行证据，没有 b01 终态物理结论、跨相位成功或科学准入。FC-E058 b00 已核 primary 数值仍为减阻 3.89197994%、rearCl centered RMS 比 0.815695748、偏置比 0.011385207；其早期 first6.2 偏置 0.135462 仍未通过 10%，不得写成全部窗口通过。

## FC-E058 已核验：b00主窗物理三约束通过，跨相位仍待验证

同一afa5cde4daec474eb52b61c08f86746f已PID0/exit0，800配对周期完成。唯一控制变化为冻结32768策略的反射投影 `.5*(pi(o)-pi(Ro))` 后一次原actionfilter。Primary (168,228]12000点减阻 **3.89197994%**、rearCl centeredRMS比 **0.815695748**、平均升力偏置比 **0.011385207**，满足原2%/1.05/10%标准，无阈值变更。全80D/U为3.75168096%/0.824355527/0.013698861；早期first6.2偏置0.135462仍未过10%，不能隐去。

独立3200rawhash/all6windows/800projection-filter等式复算通过，新zero16000点全部原始列与FC-E055一致，两容器已清理，最低Available121917501440字节。结果SHA `199127979c6cb43e6304c60fc3373a2b1a8465476ffdd265d30c108dfffd0ca6`；报告 `docs/EXPLORATORY_PROJECTED_32768_PPO_LONG_CFD_TERMINAL_REVIEW_20261006.md` SHA `44ef122bae110d22b8046b98ced195e9c4f7548441bc22f863015106bfbe7ff4`。

这是一个b00训练相位真实长窗闭环的实测成功，不是全部项目完成：K1 H100正式surrogate仍FAIL，PPO外层显式投影不是新训练策略，跨相位鲁棒性尚未成立。下一项固定b01/restart130同策略同投影复验仍须单独Root执行审批；历史validation相位也不是新的独立最终测试。以下running/失败条目为历史保留，不代表当前FC-E058仍运行。

## FC-E058 已实际启动：反射投影32768策略的匹配800周期真实CFD

唯一科学变化为请求动作 `0.5*(pi(o)-pi(Ro))`，随后仅调用一次既有幅值/变化率过滤器；策略、VecNormalize、restart148、配对zero、800周期、六个窗口、CFD数值与物理标准不变。实际unit `fluid-control-exploratory-projected-32768-ppo-long-cfd-20261006.service`，invocation `afa5cde4daec474eb52b61c08f86746f`，启动时PID3059882、active/running。批准SHA `87944e807a68caab6ce7a46e01207e1d7ba432b86ccd639b6f47424de481c64c`，不可变driver SHA `5c3f40728cd383913a256a2f46b6bfaf0b02cc7d91586e198c354a007fcb9e76`。

这是FC-E055后的单因素探索，不是重训、正式准入或成功结论。输出 `artifacts/exploratory_projected_32768_ppo_long_cfd_20261006`；尚无终态物理数字。资源合同保持CPU控制器8GiB、两个8GiB/no-swap solver、Available50/22GiB、3600/3750/120秒；未经诊断不得重试。

## FC-E057 已完成：固定680周期上的策略反射缺陷只读审计

unit `fluid-control-policy-reflection-defect-audit-20261006.service` / invocation `dff10f4049dc4b4da84d7d05818d1c9b` exit0。32768策略的 `pi(o)+pi(Ro)` 均值−0.6702486725962338、RMS0.8046740447610694；4096策略分别−0.3195020545493154/0.3197158563363626。该信号支持单因素投影值得实测，但不能证明偏置成因或闭环收益。

结果SHA `b0c48354f85a2f3e0b6ccf9f41079e2eded7e33fe4a422f7f3a0c61310fc6809`；报告 `docs/EXPLORATORY_POLICY_REFLECTION_DEFECT_AUDIT_TERMINAL_REVIEW_20261006.md`。投影过滤统计以原轨迹先前omega为条件，不是反事实轨迹；无训练、CFD或科学准入。

## FC-E055 已完成真实80D/U：减阻与波动改善，但平均升力约束未完成

范围仍只有一个训练相位b00/初始状态，不能称跨相位鲁棒性。即使后续单因素投影满足primary约束，也需另行批准第二匹配相位b01或另一真实初始状态验证；当前不启动。K1 H100正式评估仍FAIL，探索性闭环许可不等于论文级通用性证明。

同一 `285bea88ff234cd5acfb9cb03c2b3cf3` 已PID0/exit0，800个配对周期完成，实际策略来自32768步全新PPO训练。独立核验3200份raw文件及六个预声明窗口；primary (168,228] 12000点减阻 **2.37796490%**、rear-Cl centered RMS比 **0.795798378**、平均升力偏置比 **0.322286722**。10%与20%均失败；不能称全部物理约束完成，更不能覆盖K1既有正式surrogate FAIL。

结果SHA `b425bd28ea6e1ca6786ee6ea38dd3a09e13191849a5778270b987a830584e827`，报告 `docs/EXPLORATORY_DIVERSE_32768_PPO_LONG_CFD_TERMINAL_REVIEW_20261006.md`。inclusive [168,228] 12001点作为独立companion保留，不替换primary。全80D/U减阻2.44101914%、RMS比0.803539927、偏置0.343010009；404/800动作端点饱和。两容器已清理，1600 solver segments正常，最低MemAvailable120469553152字节。当前没有此trial仍运行的计算。

已真正完成FNO→HydroGym/SB3 PPO训练→直接policy真实CFD长窗闭环；已测得同起点配对收益，但持续mean-bias仍是失败约束。FC-E056同24训练reset确定性诊断只显示微小、混合的surrogate回报变化，不支持自动继续增加预算或改阈值。下一步须单独审阅针对偏置/模型与观测局限的假设；保留原CFD-only成功baseline和所有失败。以下running条目均为历史。

## FC-E056已完成：同24起点确定性H5回报仅微小改善，长CFD继续

独立核验诊断unit `fluid-control-diverse-policy-h5-comparison-20261006` / `f4ba411c1bcf4cae9ebd178ad0c30a1f` exit0，无优化/CFD/模型修改。4096与32768最终策略同24真实reset各5步、等权宏回报−3.6902918374→−3.6875228003，delta+0.0027690371；7起点改善、10恶化、7相同。drag惩罚改善，但mean-bias/actuation/rate惩罚恶化；不能据此宣称普遍收益或已收敛。H5相对62历史的奖励问题仍是假说，不是原因证明。

结果SHA `3f8c6f7e5b03877601a3b25b26409e9d6f943fcbaec62600fa51343995922b9f`，报告 `docs/DIVERSE_POLICY_H5_COMPARISON_REVIEW_20261006.md`。独立复算全部24×5行/组件，47source/192runtime哈希一致；无独立模型重载。最低Available120472039424字节，GPUpeakallocated755589120字节，12GiB/noSwap/240s监督。实际评估协议不是继承元数据中的32768训练协议。FC-E055原800周期配对CFD同285bea继续，原物理标准不变。

## FC-E055已实际运行：最终32768策略的800周期配对CFD

Root启动 `fluid-control-exploratory-diverse-32768-ppo-long-cfd-20261006`，invocation `285bea88ff234cd5acfb9cb03c2b3cf3`，初始PID2588512。批准 `e103288a0558c10784a43a199a3c4d731ffc0e6509753646da7bb6930cb4dc12`，不可变driver `17060dda570ead4fdc8e33920fcc559b5bb8ad8d640a7e154579f8a795507afa`。唯一最终策略直接CPU驱动148→228真实CFD，无在线FNO/MPC。固定早期12.4及其两半窗、主要(168,228]12000点、历史[168,228]12001点伴随窗和全80窗，不择优统计；10%物理参考不变，20%仅标注敏感性。未执行另外124周期试验。3600inner/3750outer/120stop，8GiB控制器及两个8GiBsolver/noSwap、Available50/22不变。尚无终态物理结论。

FC-E054 R2已独立核验32768/256epoch/512optimizersteps及所有24起点；前4096转移数值与原4096训练完全一致，支持仅预算变化。最终策略 `5ab92ebe04459419bc724b48c6e20bde2464d7b6d880396e504406aa08806d4a`；结果 `ff3532a604b6816fb3ad4c7a11edfcd579bcb924445abca52a2fdab8ea4dcf20`。训练575.563秒、最低Available119260291072字节，FNO冻结检查通过；不推断收敛或真实收益。报告 `docs/EXPLORATORY_DIVERSE_H5_32768_PPO_TERMINAL_REVIEW_20261006.md` SHA `7bbb772513337bbd67588aa59454fcb1272014d81b52b1a7e3785f4fe7701d19`。首个审批类型错误失败保留，以下running状态均为历史。

## FC-E054 R2已实际运行

Root在最终落盘审批上执行冻结validate_spec及全部protocol类型比较通过后，启动 `fluid-control-exploratory-diverse-h5-32768-ppo-r2-20261006`，invocation `a19900b2bfa64d8d8372b67bc0564139`，初始PID2560906。输出独占 `artifacts/exploratory_diverse_h5_32768_ppo_training_20261006_r2`；同一算法/source/资源限制，未自动覆盖或重启首个失败unit。此后所有执行审批必须在最终序列化落盘文件上运行真实consumer校验，而非只检查draft。

## FC-E054首次启动失败：审批JSON类型错误，未开始训练

Root首次启动 `fluid-control-exploratory-diverse-h5-32768-ppo-20261006`，invocation `c406bd4af9a84219840027961119e6ab`。审批5f648549…将ent_coef的JSON浮点0.0序列化为整数0，触发严格protocol比较；在模型/训练前失败，原失败unit/output/approval保留。新R2审批 `1cd1d5182fd7e7a3eed11060e7a6ffe9fad840c776f021f239f059525a515132` 仅修复0.0及独占_r2输出，实际冻结trainer.validate_spec及所有protocol类型复核通过，不修改算法或资源门槛。

不可变trainer `4d681771736b63b628712d3b62fcdde831601e80221aef6f1fd78a4b6840ff01`。预定唯一变化4096→32768，从同seed全新初始化；K1/24reset/H5/奖励/所有PPO超参数不变，最终唯一策略，预期256epoch/512steps及每相位1639reset。12GiB/noSwap/1CPU、Available50/22、1800inner/1950outer保持；尚无本预算科学结果。

## FC-E053真实CFD终态：改善行为但物理目标仍未完成

同一464de68e…已exit0完成124周期。独立审查496份raw哈希，配对zero全部原始数组与FC-E051完全相同。全/前/后窗减阻+0.505760%/+0.687266%/+0.324248%，RMS比0.973707/0.960522/0.986439，平均升力偏置比0.123721/0.107197/0.140244。无饱和，最大|omega|0.202720；较旧策略更平衡，但减阻均低于2%，10%偏置参考失败；20%仅敏感性通过，不改变标准。12.4D/U不是80D/U正式准入。

结果 `8c909aa4bd0b73e3cf570dd55cb2a1abd7346a9c424695a5e0056b4e5e833bdc`，报告 `docs/EXPLORATORY_DIVERSE_PPO_CFD_TERMINAL_REVIEW_20261006.md` SHA `31a338bfc81e4ece0adeee943c074686e7065e57048d5783697d062856aa4e53`。资源最低Available122083807232字节、两owned容器均清理。下面running条目保留为历史。

## FC-E053实际运行：多真实起点训练的唯一PPO策略直接验证CFD

Root已启动 `fluid-control-exploratory-diverse-ppo-cfd-20261006`，invocation `464de68ee1114eea8e8ae214d18dc045`，初始PID2474346。批准SHA `67fda1a404f844d89b986442a4a9000561b02757417d5a8d366fdf9f9e6033db`；不可变driver `89e0d8bea92440babd3d647eed31758db9cfc2a43e31ed6e9d9bdf5047b77b6e`。固定148→160.4、124周期配对zero，CPU策略推理，无在线FNO/MPC；原69观察、单次动作变化率限制、全/前/后窗统计与10%物理参考不变。尚无终态物理结论，不是80D/U准入；先前FC-E051负结果保留。

FC-E052训练独立终态审查已完成：4096转移、32epoch/64optimizersteps、816完整H5episodes；24起点全部覆盖，每相位reset `[35,34,34,34,34,34]`。6项结果artifact逐字节复核，172次内存记录最低MemAvailable119470489600字节；策略改变、冻结FNO不变由受审执行代码的tensor校验记录支持。报告 `docs/EXPLORATORY_DIVERSE_H5_PPO_TERMINAL_REVIEW_20261006.md` SHA `c93b1e6b7d0127c204a5dd8795b080e11068fb990c19e559314920f26c80964e`。训练诊断不是真实减阻。以下时间点状态均保留作历史。

## FC-E052已结束：24真实reset的4096步PPO训练完成，待真实CFD验证

同一 `3a34c4d621244e4bbacdf1816b5b1374` 已exit0/PID0；实际结果SHA `cd5775e4647280b77803de9a5ced6abdf6378cded4f676f935bd9836350c3640`。4096转移/32PPO epoch/64优化器step，四相位reset次数各 `[35,34,34,34,34,34]`，没有选取表现好的起点或策略。唯一终态策略 `8dc8cabf2104654345f270e3fb86edca7752cf4c883112c0a4cbd3a181acea9b`，identity VecNormalize `6988d4d161bc69c8bbd89d477e9320ad9ef264d35c9dee0bbf63954d4cdfce70`，训练83.807秒。官方FNO冻结不变；独立终态资源/日志审计由非训练实现者完成后归档。

新直接策略CFD适配器只替换真实训练身份，原124周期/69观察/动作限幅变化率/配对零参照/全部窗口统计保持，7CPU测试及实际训练JSON消费检查通过；尚不代表已执行CFD或取得收益。以下running条目保留为已核历史。

## FC-E052已实际启动：仅改变真实reset分布的4096步探索PPO

2026-10-06 07:30:12UTC，user unit `fluid-control-exploratory-diverse-h5-ppo-20261006` / invocation `3a34c4d621244e4bbacdf1816b5b1374` 已核active/running、PID2463028。固定四相位各六个起点：原zero frame0，加五种控制train轨迹frame62，共24真实状态/当前omega/原始62点受力历史。仅reset分布改变；K1/H5、69观察、canonical奖励、seed、PPO超参数与4096预算均不变，仍只保存唯一终态策略。未有此次训练或CFD收益结论。

实际24包CPU收据 `f85f84a4b82e0c21eaf011281e0b98b570bfaa083805c604e1fbe04aaa14583b`；批准 `docs/EXPLORATORY_DIVERSE_H5_PPO_APPROVAL_20261006.json` SHA `760e1f9e81494bdd8c3742cd0ce77e168b0df2bf288bd04546412096721f41e2`。新四文件不可变manifest `2f9a1e5de2c99fa153cd8316fd2ecdbf96c70a735c0c7fc6a4a07621f8ec21ca`，复用旧39依赖，43source/192runtime逐哈希核验。12GiB无swap/1CPU，Available启动50/运行22GiB保护20，GPU allocator .06；不修改正在执行源、不自动重启。FC-E051负面lift/bias结果及隐藏奖励历史风险保持，当前无CFD运行。

## FC-E051终态：真实PPO闭环已跑通，受约束物理目标未完成

同一 `fd6d92f7ea9946b49c91c07e21f1d74b` 于07:10:18UTC正常结束，124周期/12.4D/U，最终策略直接驱动真实配对CFD，无MPC代选。独立重哈希496份生成受力文件并复算全部窗口：全/前/后窗减阻 **+0.411755% / −1.635856% / +2.459428%**；rearCl波动RMS比 **1.105863 / 1.199832 / 0.990986**；同窗zero归一化平均升力偏置 **52.7311% / 41.7266% / 63.7353%**。三个窗口均不满足10%或20%敏感性参考；原长期train-b00参考下也全部失败。不是阈值放宽可解决的问题，不是80D/U正式成功。

124个请求动作全为+0.75，经单次变化率限制后117个端点饱和；策略确实执行但尚未证明有用的状态响应。四个零起点H5训练与长期部署状态分布不同，但不能归因为唯一原因：69观察不包含完整62点受力奖励历史，另有短时价值自举、奖励稀释和模型偏差风险。下一项仅批准准备/CPU核验固定24真实训练起点，保持模型、H5、奖励、4096预算不变；尚无新GPU训练批准。

结果SHA `4007493f22de5855cbd0574e0ec006ca715941b8396f4e48af6527dc11e03d47`；报告 `docs/EXPLORATORY_FINAL_PPO_CFD_TERMINAL_REVIEW_20261006.md`。最低采样MemAvailable122930147328字节，所属两容器均已清理，无OOM。保留K1原formal FAIL及既有CFD-only PPO原80D/U成功的独立身份；以下running条目只作历史。

## FC-E051实际运行：唯一终态PPO直接驱动真实配对CFD，尚无收益结论

2026-10-06 07:07:51UTC启动的user unit `fluid-control-exploratory-final-ppo-cfd-20261006.service`，invocation `fd6d92f7ea9946b49c91c07e21f1d74b`，已独立观察active/running、MainPID2346139。唯一终态SB3策略 `3af2b2863f7fffa3579832c10dd2e7053caf80fc2719ed72ad842858f3da9fe1` 在CPU上确定性推理，从真实69通道CFD探针/双圆柱受力/实际omega选择请求动作；保持原±0.75幅值、每周期±0.1变化率及线性边界ramp。不调用在线FNO，也不以MPC代选动作。

固定同148起点，PPO与zero配对124周期到160.4；全12.4D/U及前/后6.2D/U均按 `(begin,end]` 报告，保留原10%物理均值参考。这是明确批准的探索性直接策略验证，不是原80D/U准入。当前没有终态收益数字；五步代理训练向124步真实CFD的分布差异仍在。训练网格插值探针与CFD原始探针仅声明坐标、通道顺序和物理单位相同，不假定数值完全相等。

批准 `docs/EXPLORATORY_FINAL_PPO_CFD_APPROVAL_20261006.json` SHA `7ace192519a08795fe9217473fae33941fc5edbb1075daeeb3701e672c521cb3`；不可变driver `44b488a97a2882e1325da8871d3ac4905cdae2a6f2cbb17202ced91afc58b91a`，输出 `artifacts/exploratory_final_ppo_real_cfd_20261006`。CPU控制器8GiB无swap、两个求解器各8GiB；MemAvailable启动50/运行22GiB保护至少20GiB余量。保留K1原formal失败；不自动重启。

## FC-E050实际终态：真正FNO→HydroGym→SB3 PPO训练已完成，尚无该策略CFD收益

同一user unit `fluid-control-exploratory-h5-ppo-20261006.service` / invocation `21cb82da66214924b38f120eb30723e5` 于06:53:11UTC正常结束，PID0/exit0。四个真实train-zero起点、H5回合完成4096条转移、32个PPO epoch更新、64次实际优化器step；policy tensor SHA确实改变，官方K1双FNO冻结且权重不变。816个完成回合均为5步，真实HydroGym/SB3截断自举已由CPU生命周期验证。不是MPC替代PPO，也不是原100步formal准入。

结果SHA `138a7b192eef1a6454cefa47cda7803c9b362937641a645c00889ac5a5d7a0c4`；唯一终态策略SHA `3af2b2863f7fffa3579832c10dd2e7053caf80fc2719ed72ad842858f3da9fe1`，identity VecNormalize SHA `54a08a438501aac0663e50da931f41aabb63cdeb8255aa80051e7af1b4eaaba2`。运行80.902秒、最低采样MemAvailable119542509568字节，无守卫失败。详见 `docs/EXPLORATORY_H5_PPO_TERMINAL_REVIEW_20261006.md`。

短回合仍有62点成本中预测贡献稀释、价值自举外推和模型偏差风险；93.04%训练动作被变化率限制，训练损失不证明减阻。当前这次训练已停止；下一步仅准备经单独批准的终态PPO直接控制真实配对CFD，不用MPC代选动作。K1原长AR失败与以下MPC负收益全部保留。

## FC-E049实际终态：124周期CFD完成，汇总失败后离线恢复，整体减阻为负

同一invocation `a601eec2da7649b4af6f9354a4deb470` 于06:40:19UTC以exit1结束：124周期全部到160.4，但旧inclusive受力reader令trailing窗口1241点触发汇总计数错误。原failed unit与缺失result.json保留；未重跑CFD。经Lead读审批准，独立离线按原定 `(begin,end]` 完整/前6.2/末6.2D/U恢复2480/1240/1240点。

真实paired drag reduction：全12.4D/U **−0.6505988%**、前半 **+4.1150446%**、后半 **−5.4163845%**；rearCl波动RMS比分别0.832241/0.913581/0.700322。全窗meanCl近零掩盖两半+0.203434/−0.205213偏移；不能挑前半报成功，放宽10%均值标准也不能修复全/后窗阻力为负。不是原80D/U物理准入或新PPO成果。

离线恢复 `artifacts/exploratory_accelerated_long_h5_real_cfd_20261006/recovered_metrics.json` SHA `1605604dc27f106acd05e6a721f26c4ba24527ac53996d6e65fbc70c601fa2b1`；详见 `docs/EXPLORATORY_ACCELERATED_LONG_H5_TERMINAL_REVIEW_20261006.md`。原restart完整重验不变，两容器已清理；第一10周期动作及4路200点原始受力与先前CPU H5完全一致。以下running状态只作历史。下一步保持探索路线，但不得自动重跑、扫阈值或以lift降低替代减阻目标；另行评审后决定干预。

## 当前实际运行：加速H5真实配对反馈124周期（2026-10-06）

实际user unit `fluid-control-accelerated-long-h5-20261006.service`，invocation `a601eec2da7649b4af6f9354a4deb470` 已核active/running、PID2131269，阶段观测5/124。输出 `artifacts/exploratory_accelerated_long_h5_real_cfd_20261006`；批准SHA `03e2bac8f55c4bbd09e377b60bfef849515b53a6d418d490ec682b2cde95bc75`，不可变driver `4cca28757f44e80f693d2d4a33c15cea0ea5bc74368eda669aead292b37464bf`。没有终态控制收益结论，也没有正在进行模型/PPO训练。

保持已审K1/H5代价、五候选、动作约束、初态148；实际场每步重观测，以persistent Curator和加载后显式highest/no-TF32 GPU推理加速。Curator20帧加重复帧的105组数组逐字节一致；GPU十状态实际回放10/10动作和排序一致，最大受力差2.2649765e-6。证据见 `docs/PERSISTENT_CURATOR_TERMINAL_REVIEW_20261006.md` 和 `docs/EXPLORATORY_CAUSAL_HISTORY_H5_GPU_REPLAY_REVIEW_20261006.md`（SHA `95dca2e1941b2aff46d6c7510cce0ce74ad49d3fcdc439cd1fdb62064eb69524`）。这些是工程证据，不是科学准入或任意GPU精度等价。

新窗口12.4D/U分别报告完整、前6.2和末6.2；仍短于原80D/U。MemAvailable启动50/运行22GiB，CUDA空闲仅观察；控制器12GiB无swap、两求解器各8GiB，内部1800秒/外部1950秒。K1原formal FAIL保留；以下旧准备/运行条目仅为历史。

## 当前实际终态：FC-E048 H5反馈完成，尚无减阻收益（2026-10-06）

同一 invocation `6adc59fae65344d2b49b57cbe5b30f70` 已于06:12:13UTC结束，PID0/exit0；以下旧running条目仅保留历史。10个真实反馈周期、每分支200个原始CFD受力样本独立复算：总Cd均值MPC `2.4137825099145`、zero `2.413592168615`，paired drag reduction `−0.0000788622460641264`（阻力差0.0078862%更差）；后圆柱Cl波动RMS比 `0.9836105589246837`。H5产生非零动作而非H2全HOLD，但仅1D/U，不能判定长时物理目标完成。250个候选阶段成本复算一致，mean-bias惩罚全零，原10%不是这次动作选择的阻断。

执行代码 `8f5afb3`，driver `0917cd5c62e43fc3f7b2cdc23900aa9d0932dff9ef4842a155bcf4288524b14d`；批准 `f927f6b956f847766d899745ebc0e679a7638db63f27cfb1f9a489eb69fb6c0e`。实际结果 `artifacts/exploratory_causal_history_h5_real_cfd_20261006/result.json` SHA `d4c3ad8198f69199606c0fa7a6c1a668c9a0f581e3b52bbeca99e2b0bd902c5e`；独立报告 `docs/EXPLORATORY_CAUSAL_HISTORY_H5_TERMINAL_REVIEW_20261006.md` SHA `e6b7496b3a59b0bfe5a6b4365bbb9076d892ec651b3cf75fa212075b98dbe79f`。两个所属容器已清理，无OOM；CPU Available最低121418903552字节。

下一步仅准备：固定H5、124周期/12.4D/U配对试验，同时报告完整窗口和末6.2D/U；不扫权重/时域。加速采样与GPU精度需先完成工程等价性评估并显式审批，尚未执行该长试验。不作原80D/U准入、新PPO或新GPU训练声明。

## 正在执行：H5真实配对反馈（2026-10-06，阶段观测）

实际unit `fluid-control-exploratory-causal-h5-real-cfd-20261006.service`，invocation `6adc59fae65344d2b49b57cbe5b30f70`，PID2075648已核active/running。批准SHA `f927f6b956f847766d899745ebc0e679a7638db63f27cfb1f9a489eb69fb6c0e`，不可变driver SHA `0917cd5c62e43fc3f7b2cdc23900aa9d0932dff9ef4842a155bcf4288524b14d`，输出 `artifacts/exploratory_causal_history_h5_real_cfd_20261006`。15项CPU测试及独立全源检查通过，代码同步GitLab8f5afb3。

最近已确认3/10周期，前两次转速为+0.1/+0.2，双分支CFD正常推进；仍无最终减阻结论。与FC-E047相比只将H2改H5，CPU/模型/代价/候选/初态/动作约束不变。原长窗口失败保留，未新增PPO训练。终态必须按同一unit与原始力样本核验，不能把此阶段运行记录当最终结果。

并行完成no-TF32 GPU工程对照：同一帧后Cl相对CPU最大差由.013054降至1.87755e-6，热推理约.14秒；最低采样Available120607776768bytes。仅单帧证据，报告 `docs/K1_UMA_GPU_NO_TF32_TERMINAL_REVIEW_20261006.md`，不自动改变当前H5设备或宣称全局等价。以下为历史。

## 最新工程事实：真实GPU推理完成，H5反馈实现中

统一内存探针 `74f94b14bb81495daed3568042bed798` 正常结束。原官方K1在同一已有帧上完成3次五候选H2推理，首次1.150291秒、热运行0.139241/0.137661秒；最低采样MemAvailable113.4714GiB。CUDAfree约1.78GiB仅作观测，不能等同实际可分配余量。模型权重不变，没有训练/CFD动作；报告 `docs/K1_UMA_GPU_INFERENCE_REVIEW_20261006.md` 已同步GitLab6c27470。不外推长时资源安全或CPU/GPU数值等价。

H5单因素真实反馈尚在实现和独立检查，未启动。看板已绑定FC-E047真实终态（10次零动作/零收益），不是运行中。后续仍需真实减阻验证及兼容HydroGym/PPO流程；短MPC和GPU工程成功不能替代整体目标。以下为历史。

## 最新：第二轮真实反馈完成；准备 H5 单因素对照

FC-E047，同实例 `6f554f10e87e4b9f9d6b6ed8b555c548` 已exit0/PID0，10次动作全部为0；两分支各200个原始力样本相同，减阻收益0。结果 SHA `74a28d45dce9b84ec5044700fe470390cde899a2fcf40a0b893c1b28817d99ca`，独立报告 `docs/EXPLORATORY_CAUSAL_HISTORY_H2_TERMINAL_REVIEW_20261006.md` SHA `9ceb4d58a66b549faa86834d15d0444d57c8fdcc2f2635f18859440a679d2098`。没有新训练、PPO或正在运行的CFD计算。

100个候选阶段平均升力偏置惩罚全部为0；H2预测减阻收益不足以抵消动作/变化率代价。Lead批准准备H5代替H2的单因素反馈对照，其他模型、代价、动作、初态与10周期配对不变，尚未执行。探索性短反馈不再等待全部长AR指标通过；原科学失败与长期物理评价仍保留。

并行准备隔离有上限的GPU推理探针：官方说明Spark CUDAfree不包含可回收缓存，不能单独判断实际可用容量。统一物理MemAvailable至少20GiB仍是要求，尚未修改旧守卫或执行新GPU任务。以下为历史。

## 最新：实际FNO-MPC→CFD短反馈完成，阻力目标未达（2026-10-06 05:37 UTC）

FC-E046：同实例 `e3b9eb7b58724a1c9ec4e64d63ac7bbe` 正常终态，10次实际当前场重观测/官方K1双FNO H2选动作/两分支真实OpenFOAM推进完成；不是shadow-only，也不是新HydroGym/PPO任务。200个真实力样本/分支的全指标独立重算与结果完全一致。MPC总Cd均值2.4192140666805，配对零控制2.413592168615，阻力增加0.23293%；后Cl波动RMS降低5.6891%，是短时取舍而非总体目标成功。

实际1D/U窗口短于涡脱落周期，不能代替原80D/U评价或物理10%平均升力标准；K1原formal FAIL保留。结果SHA `45fcab568ed7456e521ed17c4469f44716d231ca4ec5c08820864803ae856fbb`，独立报告 `docs/EXPLORATORY_PAIRED_H2_TERMINAL_REVIEW_20261006.md`；同源双case与owned容器清理已核验，CPU可用内存保护满足。当前阶段是FNO辅助真实短闭环工程完成、长期减阻目标未达。

Lead已批准准备单一代价改动：复用canonical62点实际过去力历史，再分别加入H2预测并平均各阶段代价；实现/测试准备中，尚未批准或执行新试验。先确认因果时序和目标含义，不改变物理门槛，不把该短窗收益外推长期。以下记录按时间作为历史保留。

## 当前优先：两段真实CFD场桥接已完成，下一步配对10周期MPC试验（2026-10-06 05:21 UTC）

R4同invocation `99e5c019c08240668241f7ac036320f3` 已实际exit0/PID0：固定零动作148→148.1→148.2，各20个dt=.005求解步，两个新端点均由原Curator采样并经canonical适配器得到[1,6,128,256]当前场输入。原restart/source哈希独立核对不变，owned容器已清理；结果SHA `be3c57e00003d7092b116058604a47d2ea2b2c1f033551adb39188f1c91f7584`，报告 `docs/TWO_SEGMENT_CURRENT_FRAME_R4_TERMINAL_REVIEW_20261006.md`。

这是无模型、无策略、无控制收益声明的真实工程桥接，不是正式准入。CPU-only已单独批准按MemAvailable启动50GiB/运行22GiB保护，77次记录最低114.195GiB；MemFree仅记录，GPU20GiB要求未改。先前启动余量失败及R2环境PATH失败完整保留。依用户探索性闭环优先级，下一优先任务是单独批准的配对10周期MPC真实试验，预测与下个CFD端点对照包含在试验内，不另设shadow-only前置任务；不等待长预测全部通过，也不把旧科学FAIL改成PASS。以下运行/等待状态均为历史。

## P031第二次资源测试因内存余量退出（2026-10-06 04:50 UTC）

同48a1f83e实例已terminalfailed/exit1：启动CUDA33.6905GiB通过，但进入H25循环后内部guard退出；外部记录 `ppo_gpu_floor_violation` CUDA19.816570GiB。确有低于用户20GiB要求的观测，不能声称保护始终满足。没有完整反向结果、optimizer更新或候选模型；不是已证明的OOM，也不是科学精度失败。数据哈希/导入期间物理空闲从34.08降至约25.05GiB，随后计算时进一步下降；各组成的精确贡献尚未实测。

保留两次失败和全部源代码/日志。下一安全改动正在准备：提高启动余量至40GiB、提前在22GiB检查处停止，并在抛出异常前持久记录资源观测；原用户20GiB底线不变。检查已完成项目文件的定向缓存回收资格，不执行全局清缓存、不删除数据、不降低科学门槛。新尝试仍须实际余量与独立实现审查。下面运行中状态均为历史。

## P031资源探针恢复运行（2026-10-06 04:49 UTC）

首次实例42df9cff在CUDA预检30.675<31.301GiB时退出，未执行H25或产生模型。失败证据完整保留，审查 `docs/FC_P031_RESOURCE_PROBE_TERMINAL_REVIEW_20261006.md`。经单独批准，仅回收16个已完成验证文件的干净缓存，未修改或删除数据；恢复前MemFree36,647,903,232bytes。

实际恢复unit `fluid-control-fcp031-resource-probe-r2-20261006.service`，invocation `48a1f83e007a4bb59f1ea025af2e7dd7`，输出 `artifacts/fcp031_h25_resource_probe_20261006_r2`。原424项固定源码manifest495e4f54及批准3177f364保持；实际CUDA预检33.6905GiB通过。仅单窗口H25前向/反向，无optimizer或模型保存；内存保护、900秒限制不变，尚无终态资源结论。完整训练仍须资源结果与候选加载/验证兼容审查。以下状态按记录时间解读。

## 最新工程进展：真实单帧输入一致，H25资源检查准备中（2026-10-06 04:47 UTC）

现有b00_zero、t=148.0的单帧VTU经原官方Curator采样器处理，与官方HDF5Reader读取的既有frame0逐值一致：97020个有效物理/归一化数值和196608个模型输入数值的最大绝对差及RMSE均为0，mask/grid/time一致。CPU隔离单次任务正常完成，未加载模型、未训练、未运行CFD求解器、未执行控制动作；该结果只是一个已有帧的接口一致性证据，不代表在线闭环完成。输出 `artifacts/online_current_frame_cpu_20261006`，独立复核进行中。

P031拟将同一官方FNO的训练预测长度从10步增至25步，其他数据、父模型、损失尺度及验收不变。最小资源探针/训练入口已完成源码与合成CPU测试，正在独立审查和绑定执行来源；尚未运行H25 GPU探针或训练。当前GPU空闲，不能标记为训练中。计划及当前场适配器已同步GitLab `cc04220`。整体目标仍未完成。

## 最新结论：连续推演误差在训练轨迹内也恶化（2026-10-06 04:26 UTC）

P030 r2正常结束（同4b89d3cb实例、MainPID0、exit0/noOOM），44条起点相同轨迹的两模型100步预测及全部原始记录已保存。独立重算全汇总一致；额外NumPy重算最大差4.86e-17。结果SHA `b1042b94fde60aed135c60d348431aa1c6177b1b9b12b1ae8ba56bc9fab6ee7f`，报告 `docs/FC_P030_RECOVERY_TERMINAL_REVIEW_20261006.md` SHA `4e21fc05f543b5c90a74e318e9e7ae999b27b8350d7573817b49c4c6670dc5b4`。最低观测CUDA空闲21.3438GiB。

新P029相对K1：第1/10步速度场误差略低；第25/50/100步，44/44条轨迹速度误差均更高。第100步汇总速度相对L2 0.050850→0.062903，后Cl MAE 0.055621→0.066476，总Cd MAE 0.015652→0.018949。第10步气动力已经总体变差；不能把问题完全归于训练外工况。局部例外保留：train16第100步后Cl MAE下降，train8总Cd MAE下降。固定start0早段面板不是完整1368窗口或旧origin51面板，不能推断唯一原因。

下一步正在设计单一更长预测训练窗口的对照（P031，未批准训练），保持官方FNO、原数据及验收不变；并行准备复用现有官方Curator单帧采样器的当前场适配接口，仅CPU工程测试。总体模型准入、兼容策略及真实CFD在线闭环仍未完成。以下“运行中”均为历史。

## 最新：P030修复后实际重跑（2026-10-06 04:23 UTC）

重跑unit `fluid-control-fcp030-train-horizon-r2-20261006.service`，invocation `4b89d3cb85c540448eeebd8ef5c7c3b3`，初始MainPID1771775；输出 `artifacts/fcp030_train_horizon_diagnostic_20261006_r2`。批准SHA `6c4edae1e955268dcae718c4eb2f6b24a216b8961189d56a09f4d4da7c091378`，v3源码manifest `73ac42127e0ace741c675cb7a5a53a6339171565462d0771c11665124d0f08e6`，恢复独立审查 `docs/FC_P030_RECOVERY_REVIEW_20261006.md` SHA `a0d7865fdde4c10cd754fb2c975b995aaa8bd655b3b0eeb1ec0281b1b043b8df`。

仅修复元数据rows传参、汇总前保存raw_records.json及其SHA；原数值核心/模型/数据/资源保护不变。25项集成测试与15项核心测试通过，新增实际核心44对记录汇总与生产调用AST回归。首次失败和v2全部保留。一次只读精确44文件缓存提示完成，收据SHA `fecbbdb461f3d94e9b7cf15310c562905802cfc95aae7a8fe5770bc38994065a`，未写入HDF/模型，物理空闲恢复约32.1GiB。已观测origin_complete1至3及实际看板登记；尚无最终诊断结论，更不是训练或闭环完成。

## 最新：P030首次运行汇总接口失败（2026-10-06 04:19 UTC）

同一实例d7739a9bf4fe42f68a584e8ec5edc684已终止，MainPID0/exit1。44项origin_complete存在，但driver把tuple键的dict传给要求metadata行列表的grouped_and_paired，触发TypeError。没有result.json，也没有保存rawrecords，不能报告科学指标或成功；必须修复并重跑。观测CUDA最低21.538467GiB、MemAvailable最低111.160320GiB，属于软件汇总接口故障而非内存故障。已授权小范围修复：直接传原始rows、用实际core执行44记录汇总的CPU回归测试，并在汇总前保存原始预测记录。原失败输出和v2源码保留不变，新版本需独立检查及单独重跑批准。下方运行中均为历史。

## 最新：真实模型H100诊断已启动（2026-10-06 04:16 UTC）

P030固定44条真实训练轨迹、start0、K1/P029连续100步对照已实际启动。user unit `fluid-control-fcp030-train-horizon-20261006.service`，invocation `d7739a9bf4fe42f68a584e8ec5edc684`，启动观测MainPID1756637。批准 `docs/FC_P030_EXECUTION_APPROVAL_20261006.json` SHA `2f80dab06ca1916495272dac0adf155745969534ce3d5b1626e7a4295eadc98f`；v2源码17项manifest SHA `a14bd2c00ff5c905ccd6193843736a6ccc2806cd2c2d7a51827635cb09fdb6e3`。23项集成CPU测试与15项核心CPU测试通过，独立审查 `docs/FC_P030_INTEGRATION_REVIEW_20261006.md` SHA `fa72abdd5686feff177a45ba615744c10c8098a2b7f1b0f51a92ea3843f36e7f`。v1固定源码保留。

该任务不更新参数、不保存新模型、不授予策略训练或闭环准入。输出 `artifacts/fcp030_train_horizon_diagnostic_20261006`，先记录selection.json，再对照各预测时长的流场与气动力误差。900秒上限、原20GiB物理/CUDA守卫保持。实际CUDA预检31.7396GiB通过；启动后资源监控仍在持续。看板已经登记本次真实实例与44轨迹进度。整体闭环目标尚未完成；下方GPU空闲或准备中状态均为此前记录。

## 最新：P029完整计算已结束，科学判定未通过（2026-10-06 03:47 UTC）

同一正式实例 `85ae29a422fc48739136418317de8ca5` 已结束（MainPID0、exit0），不是资源故障。原始development gate为FAIL：六个窗口joint2/6、总Cd6/6、后Cl波动2/6、后Cl均值4/6。通过的是b01-minus及b05-plus；两个zero分支的波动预测现在均失败，不能只报旋转分支改善。原始收据SHA `96e207af491ef4abe0c9e9c85983672111d86d70fe88b2d88551b29d0739a334`，gate SHA `aa7dd557bc516e898339655517eba8bf16cf27b4579751c6b9f75a4de9153c53`。独立复核已完成：35项输出、411项源码哈希一致，8个容器exit0/noOOM；原审计器所有离散判断一致，浮点重算最大差4.44e-16；最低物理空闲21.200443GiB。报告 `docs/FC_P029_ORIGINAL_FORMAL_TERMINAL_REVIEW_20261006.md` SHA `020042bb6846e9be14ccb10e36035bba7c5d3fa4d6e164c81ee5d027f9a62527`。未授予PPO准入。

P029独立复核与FC-E043台账已同步GitLab（ae99908）。P030固定44条训练轨迹start0/H100诊断的CPU核心获独立审查通过，15项工程测试通过；审查SHA `cc4f3e969850890f0a0b7384c25c5202885ea2a2b031f1e6571d0a5fd1b12596`。03:56 UTC已批准执行入口、来源清单和隔离容器启动器的准备，文件 `docs/FC_P030_INTEGRATION_PREPARATION_APPROVAL_20261006.md`。尚无真实模型/数据诊断或新GPU训练，当前GPU空闲用于等待数值程序集成完成，不能称为训练中。下一步完成集成检查、绑定实际来源并单独批准诊断运行。看板如实显示评估结束、目标未完成。下方“正在运行”均为较早观测。

## 当前：P029训练和H10对照完成，原完整验证运行中（2026-10-06）

最终FNO辅助PPO/MPC真实CFD在线闭环目标未完成。P029恢复训练已正常完成171次更新/1368窗口、候选审计和官方独立CPU重载；实际训练结果SHA `39b246d07ff673de5b3d5fdc2e65be5e46bcb466e1290f5549f2e75d5cb81336`，重载 `2023daadf611e6b4fe30146f029d142b1c432c09b41e14fe1af6bc1e7f6d9f64`，恢复运行CUDA守卫最低20.9074GiB。首次资源失败完整保留，不能由恢复成功抹去。

同44训练轨迹origin51/H10对照（FC-E042）已完成：P029相对原K1场mean-case RMSE下降2.50%，rear-Cl MAE下降13.79%，总Cd MAE下降9.01%。但场误差仍比P028高13.10%，升力波动改善并不跨所有数据族/相位一致。这是train-only诊断，不是独立验证或闭环成功。结果SHA `aed040eb22766f8f47b1f6093f50bc1aa753dd748cd90b8d68ac9db1095b329d`；报告 `docs/FC_P029_H10_TERMINAL_REVIEW_20261006.md`。

**实际运行原完整正式验证**：user unit `fluid-control-fcp029-original-formal-20261006.service`，invocation `85ae29a422fc48739136418317de8ca5`，输出 `artifacts/fcp029_original_formal_20261006`，批准SHA `b339d1175ea17dfd110763c044b0d9a2a15ce088062b1834c3627969f279fec5`。7项数值步骤与标准保持不变，11:26观测当前dynamic6。validation10端点子项通过，动作Cd变化MAE 0.020434、符号8/8、排序20/20；但H100速度L2/后圆柱Cl MAE/总Cd MAE为0.056598/0.045117/0.018809，均高于K1的0.043591/0.040186/0.010928。短期训练收益未稳定转化为长期验证收益，尚无完整验收结论，不批准新PPO。两小时结论已按时发布 `docs/TWO_HOUR_CONCLUSION_20261006.md`（GitLab提交1c3230b），明确目标未完成；此前记录中的运行状态均为历史。

### 并行下一步：P030诊断程序的CPU实现（2026-10-06 03:41 UTC）

P029原完整评估仍在同一实例运行，尚无最终判定。P030设计与独立审查已完成（计划SHA `07febacc623df47923db15fc8e560770c7b34f3643aa61a133efd65b022dc0b8`；审查SHA `237d88ae7709b82bbe4a9a6d547f5447a0b05fcec17726f9963fb192f62555ff`）。已批准隔离CPU代码与合成工程测试，批准文件 `docs/FC_P030_CPU_IMPLEMENTATION_APPROVAL_20261006.md`；尚未执行真实数据/模型/GPU诊断。

待检验的是长时误差是否在训练轨迹内部已经出现：固定44轨迹各自最早start0窗口，K1/P029分别连续预测100步，比较相同lead的场与力误差；它只覆盖固定早段面板，不代表全部1368窗口分布。旧origin51/H10成绩不能填入新面板。既有官方模型、数据划分和全部验收要求不变；数值执行需程序审查、来源绑定及当前formal终止后的单独批准。模型精度、兼容策略和真实CFD闭环仍须实际验证。

## 训练资源故障已处置；实际第二次运行（2026-10-06 02:51 UTC）

第一次P029训练b343d13dffb6402fa91bec055cafad32在1次更新/10个窗口后因内部物理/CUDA内存守卫退出1，无候选。触发瞬间采样未持久化，不能用外层20.134GiB采样最低值声称全程保持20GiB。完整五个证据文件已移至 `artifacts/fcp029_control_aware_flow_training_20261006_failed_attempt1`，迁移前后SHA完全一致；未删除数据。

精确核验并建议回收已结束验证任务16个HDF缓存，收据 `acad2af0c12ea2b5178db97eeb5dc08a7231061ff75480b4e04cc3cbdf417264`；随后原44训练文件r3缓存收据 `a8eb03631d838acdc59036cb2a5340a2bdfd7387758a43c8e9ddfe557b991917`。均只读同FD哈希/状态检查，未改写数据。实际启动前物理空闲36.55GiB，超过新35.5GiB恢复要求；连续20GiB守卫不变。

经 `docs/FC_P029_TRAINING_RECOVERY_20261006.md` 单次恢复批准，从原父模型与新Adam状态完整重跑171更新，不接续失败更新。同一user unit的实际新invocation **7c8b277455444df58825338a6590d684**，观测PID1587416。该运行尚未产生终态结果；以下b343及其“正在运行”描述均为历史。截止11:27报告实测结论不变。

## 实际进行：P029控制相关流场训练（2026-10-06 02:47 UTC）

同一官方FNO、原父模型与1368训练窗口，固定171次参数更新。实际user unit `fluid-control-fcp029-flow-train-20261006.service`，invocation `b343d13dffb6402fa91bec055cafad32`；批准SHA `4412292ee5695ee86e6dca1facdc1436587e161199da6e5394cea582003e3566`。模型和科学验收均未完成；不宣称新的PPO或CFD闭环结果。

原1368窗口父模型尺度计算已完成，field/force固定尺度0.001456146538716282/0.003364271827125755，收据SHA `05c71e723a73457de3bc3bac6539455ff3057f58b861affd8dd7d8f7d55782ef`；十步反向资源检查完成且30组梯度有限非零，模型参数不变，收据SHA `48e356dcda21fb70b74129aa611744e693973ae09d2f6cb7929ea00d7066f161`。独立报告 `docs/FC_P029_SCALES_RESOURCE_REVIEW_20261006.md`。最低CUDA空闲21.0737GiB；无更新检查不包含Adam峰值。已考虑约0.35184GiB动量及额外临时量，启动前精确44训练文件缓存提示恢复约33.5GiB物理空闲，保持连续20GiB守卫，不能据此保证完整训练容量。缓存未改写或删除数据。

原P028完整正式评估失败结论保留。新训练结束后须独立重载、同协议H10对照和原完整评估；11:27前报告实测结论，无论目标是否达成。

## 最新结论（2026-10-06 02:33 UTC）

P028原完整正式评估已正常结束，但科学验收失败：六个长时窗口仅1/6同时达标，阻力、升力波动、平均升力预测各2/6；原K1为1/6、5/6、2/6、4/6。短期流场改善没有转化成长期气动力精度，不能据此进入新PPO。完整收据SHA `63fd75d4e90176dd94998f2844f2f70cb5a7e357bd59d5362591019ed8655154`，独立复核 `docs/FC_P028_ORIGINAL_FORMAL_TERMINAL_REVIEW_20261006.md`。35输出与411源码全部匹配，8个容器exit0/noOOM；最低物理空闲28.029GiB。

下一项P029保留官方FNO、父模型、训练数据及171次更新，只把固定50/50的流场与气动力误差共同用于流场模型训练。先计算父模型训练集损失尺度，再执行带气动力反向图的资源检查，通过后才训练。此处记录的是准备，不是已执行。物理平均升力10%约束不变，模型预测误差要求也不变。北京时间11:27前提供完整阶段结论；总体FNO辅助强化学习/真实CFD闭环目标尚未达成。

以下时间较早的运行状态为历史记录。

## 当前结论与运行状态（2026-10-06 01:57 UTC）

工程更新（02:15UTC）：P028仍在同一实例运行dynamic6。P029已完成独立CPU准备，现有官方FNO增加固定50/50流场/气动力训练损失；423项训练源码只读副本已核验，尚无P029 GPU执行。canonical新增68项CPU测试通过，旧兼容133通过/1跳过；一项需PhysicsNeMo的额外测试不能在无该包的host环境收集，未改环境。详见 `docs/FC_P029_CPU_PREPARATION_REVIEW_20261006.md`。只有原完整评估结束、实际尺度和新反向图资源检查之后才可批准下一训练。

- **目标未完成**：官方FNO代理、兼容的HydroGym/PPO策略、真实CFD在线反馈控制须一起验证；已有CFD-only PPO基线不能当FNO辅助闭环成果。
- P028已完成1368训练窗、171次参数更新及官方独立CPU重载。相同44条训练轨迹、相同起点/动作的H10对照已完成：平均单例全场RMSE从0.038909降到0.033544（改善13.79%），但后圆柱升力MAE从0.038693升到0.039507（恶化2.10%），总阻力MAE亦恶化。不能只报有利的流场指标。
- **实际正在运行原完整正式评估**：unit `fluid-control-fcp028-original-formal-20261006.service`，invocation `eb4e12507302498bb8944373e0717a25`，观测PID1436370、activating/start，当前validation10。7项数值步骤原样保留，输出 `artifacts/fcp028_original_formal_20261006`。没有新PPO或闭环成功结论。
- 下一步：完成并独立复核完整评估。若不合格，仅准备一个可检验的改进——在同一官方流场FNO上加入通过冻结受力模型回传的气动力训练损失；须先审查设计与资源，不扩大网络、不扫权重、不降低验收要求。
- 两小时完整阶段结论截止 **2026-10-06 11:26:54北京时间**。资源连续保留20GiB；本次H10最低主机/CUDA空闲23.1112/23.1102GiB。

证据：`docs/FC_P028_H10_TERMINAL_REVIEW_20261006.md`；H10结果SHA `6146ea9276570981cc72c949e3e6fac46737c43aa83c561a37eaa0e54a4ab793`。下方记录均为此前里程碑，不能用历史“运行中”代替上方最新状态。

## 最新完成：P028训练及官方独立CPU重载（2026-10-06）

同一训练实例c46c60f3c2634802b2646bb094f9d201已正常结束：1368窗、171次更新，官方容器exit0/noOOM。结果SHA `74bc0d491d82da8c3b897a330e1397ac7db2e92465221801ae1868a57114840d`，实际候选审计SHA `dd3390d0ac09f8f8e4673ed8eb48d2fbe47293a1dd1689a7abae29972ddddaea`。主机采样最低物理空闲20.8349GiB，CUDA守卫最低21.2368GiB。首尾训练窗口不同，不能据其损失变化宣称精度提升。

独立官方CPU重载unit `fluid-control-fcp028-official-cpu-reload-20261006.service`、invocation `5edc5a00bd14492c9203a4b7d4d676b3` 已exit0/noOOM、无GPU请求。重载收据SHA `685d55a9a2116d1554f14c26e54ce2d2913a4f9c47553c70407f3c4e25d3aecd`，双模型张量与训练终态一致。报告 `docs/FC_P028_TRAINING_TERMINAL_REVIEW_20261006.md`；尚无科学准入或新闭环结果。

下一步：原44轨迹origin51/H10的训练前后对照，然后原完整正式评估。正式审批SHA `c061e50dc1d868e80d1c858ea79b03e8fe5560a204bbb30d2ef9236373838d55` 已通过真实候选/审计/重载绑定的dry-run（7项数值阶段），尚未执行GPU正式评估。不晚于北京时间11:27给出完整阶段结论，目标未完成则明确说明。以下运行中记录为历史。

## 当前实际运行：P028流场多步训练（2026-10-06 01:30 UTC观测）

实际unit `fluid-control-fcp028-flow-train-20261006.service`、invocation `c46c60f3c2634802b2646bb094f9d201` 为activating/start且PID1371882；日志已完成37/171次更新、302/1368个窗口。该数字是一次观测，不是实时常量；看板已按同一真实实例显示更新与窗口数。训练配置SHA `655f4d924036ef1e23857c4bc1892b8f0bbce40af1a1b22f8bd4657eca097837`；官方FNO流场模型H10训练，K1受力模型冻结，目标和数据不变。仍未产生终态候选或科学准入。

训练前按限定22个旧模型文件及44个核验训练文件实施缓存提示，物理空闲达到35.4GiB；观测训练中约22.9GiB，持续守卫保留双20GiB/CUDA20GiB要求。没有删除/改写数据。实际训练结束后仍须模型重载、原完整正式评估，合格后才进入兼容HydroGym/PPO及真实CFD闭环。

正式评估源码已实际固定：`artifacts/fcp028_formal_source_20261006_immutable`，411个数值源码、独立外部执行器、9个CPU重载依赖，receipt SHA `fb5fd1ef87a09188d78453d0c5f93e49cf1a795dc7fa9fcee5fd77bf14cc910d`，独立核验进行中。用户要求两小时结论：不晚于2026-10-06 03:26:54UTC/北京时间11:26:54，详见 `docs/TWO_HOUR_REVIEW_20261006.md`；报告实测进度和未达目标，不保证按时制造成功。

## 最新实测：P028十步前向/反向资源检查完成，尚未训练（2026-10-06）

R3实际unit `fluid-control-fcp028-resource-r3-20261006.service` 已exit0、PID0，官方容器退出0且无OOM。结果 `artifacts/fcp028_flow_resource_probe_r3_20261006/payload/result.json` SHA `e808095f9c4de77f838c7132615427ba76985f78d3c0c7803b1d8528008a40f1`；独立报告 `docs/FC_P028_RESOURCE_TERMINAL_REVIEW_20261006.md`。原窗口816实际完成H10完整反向，30项梯度有限且非零，模型前后张量完全相同；零优化器更新、无新候选。这不是模型精度改善或闭环成功。

最低物理空闲20.9028GiB、CUDA空闲20.9051GiB、可用110.3150GiB。Adam两份动量至少另需0.35184GiB，尚不含临时量，故不能由资源检查直接证明完整训练容量。下一步对明确已结束且无写入的项目CFD文件评估定向缓存回收，保留20GiB连续守卫；完成原正式评估调用兼容后执行既定1368窗口/171更新流场训练。此前两次工程失败及原始证据保留，R3修复只涉及三项源码依赖和只读数据挂载路径。以下准备状态均为历史，最新科学结论仍为P027诊断与K1/K4正式失败。

## 当前工程进展：P028多步流场训练准备（2026-10-06）

目标函数、无更新资源启动程序、训练执行程序及显式P028模型加载支持已完成代码集成。Root canonical runner/loader与旧模型回归共90项CPU测试通过；目标函数10项、资源启动P027/P028回归24项也已通过。真实GPU资源检查尚未执行，当前无新的训练或科学准入。详情 `docs/FC_P028_RUNNER_LOADER_CPU_REVIEW_20261006.md`。

418文件资源检查源码副本已创建且只读；它保留原P026加载器以读取K1父模型，不能与后续训练/评估的新加载器混淆。下一步在保持20GiB余量下准备一次真实H10前向/反向资源检查；目前物理空闲约26GiB低于额外30GiB启动要求，必要时仅对经SHA核验的44个项目训练文件作一次可回收缓存建议，不修改数据。尚未签发此缓存维护或GPU执行批准。完整训练仍需实际容量证据，原完整formal及真实CFD闭环目标不变。以下P027完成情况仍为最新科学结果。

## 当前：P027真实误差诊断完成，尚无新的代理准入（2026-10-06）

实际unit `fluid-control-fcp027-diagnostic-20261006.service` 已active/exited、exit0；官方b40容器退出0且无OOM。44条训练轨迹各10步，共440次流场推进及1760个受力状态评估（由完整案例和已审循环核对）。结果 `artifacts/fcp027_short_horizon_diagnostic_20261006/result.json` SHA `7785ebb92ca932b4fb572175b4bd66497fc3b7495f6ecdb534f3b587a0096366`，独立复核报告 `docs/FC_P027_TERMINAL_REVIEW_20261006.md`。

后圆柱Cl整体MAE：K1真实流场条件0.026546、连续预测0.038693；K4分别0.026586、0.038588；保持初始力不变的基线0.633210。两模型各29/44轨迹连续预测更差；K4相对K1的连续预测MAE仅改善约0.27%。这支持预测流场输入误差/分布变化的贡献，不证明唯一原因，也不证明缩短控制预测长度即可满足闭环要求。原K1/K4正式FAIL不变，无新PPO或物理成功。

26次主机采样最低MemFree22.5692GiB、MemAvailable111.0762GiB；25次GPU守卫采样最低CUDAfree22.5718GiB，均满足20GiB要求。源码复核确认P026已经包含等权H1/H100连续预测训练，不能重复称为新增预测状态暴露。下一步设计冻结K1受力模型、只训练现有官方FNO流场模型的H10多步对照，检查是否降低预测状态带来的额外受力误差；仍须原完整formal，不能用短期改善代替闭环验收。尚未批准新的GPU训练。以下准备和运行中描述为历史。

## FC-P027真实诊断执行准备完成（2026-10-06）

诊断及资源启动程序已独立审查并集成，Root canonical联合32项CPU测试通过；只读414文件源码与来源配置完成独立核验。批准单 `docs/FC_P027_EXECUTION_APPROVAL_20261006.json` SHA `fe218527b6f85daf08999673b9525a2b93144e1722235f5769dc2ea055e567a5` 批准一次44训练轨迹、origin51、H10的只读误差分解，无优化器、无新模型、无validation/frozen/PPO。官方镜像b40，GPU0、allocator0.06、容器12GiB、900秒，启动free30/available50，运行双20与CUDA20GiB守卫。dry-run已通过但尚不能据此称为实际运行；实际unit/container观察后另记。K1/K4原正式FAIL保持有效。

## 最新终态：2026-10-06，K1/K4均未满足代理控制精度要求

K4原协议完整评估已实际结束：unit `fluid-control-fcp026-k4-formal-20261006.service`，invocation `d5d2201c8e2c4bf2ab40201cca0dcb1e`，PID0、active/exited、ExecMainStatus0。receipt SHA `729f9ce1f307d5462307470af20491806f6cfe31b5c81fc74ec284a2461841d9`；development gate SHA `ce60621723ce364e4f8cdc165268a64ca5fd1a0185b7529bc21c1f91ac8aab9e`，状态 `DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL`。程序正常完成不等于模型合格。

六个受力窗口仅1/6同时通过；总阻力5/6、升力波动2/6、平均升力4/6，与K1相同。四个旋转分支的升力波动误差为0.068236、0.122342、0.068306、0.081256，均超过约0.0294的预测误差限值。1126次资源记录最低MemFree20.770393GiB、MemAvailable110.100964GiB。未自动启动PPO，未使用冻结测试集；实际物理平均升力10%要求不变。

独立终态审查已完成：35项输出和411项源码哈希一致，8阶段正常退出，原审计程序在Python3.12完整重算一致。报告 `docs/FC_P026_K4_FORMAL_TERMINAL_REVIEW_20261006.md`，科学记录FC-E038。下一优先为FC-P027：复用44条真实训练轨迹，区分真实流场条件下的受力误差与H10连续预测误差。Root已批准隔离CPU实现与小型工程测试，尚未批准真实数据扫描、模型加载或GPU执行。不追加小幅参数扫描，不用混入52点真值后的成本误差替代10点预测本身的误差。下方运行中描述均为历史观测。

**最新实际状态（2026-10-06）：K4原协议formal已启动，仍在运行，无结果结论。** 独立观测时间2026-10-05 23:38:24 UTC；unit `fluid-control-fcp026-k4-formal-20261006.service`、invocation `d5d2201c8e2c4bf2ab40201cca0dcb1e`、PID1156613为activating/start。实际官方b40容器 `0be58547…f84b64` 正执行validation10 H1/10/50/100、stride25/batch4、显式p026_k4，绑定候选manifest9d1fb9bd…2ae1b5e。审批SHA `03f6880893a730307a6a6acf0ed19276ca8dc958fc3266ebf14e8e2a1323cbb8`；源码/候选/数据挂载只读。启动不等于阶段通过、整体准入或PPO授权。实际观测见 `docs/FC_P026_K4_FORMAL_RUNNING_OBSERVATION_20261006.json`；下方此前里程碑保留为历史。

**当前里程碑更新（2026-10-06）：FC-P026 K4训练1368窗/171更新已实际完成，终态完整性审计及官方CPU双模型重载已通过；尚无K4正式评估结果或科学准入。** 同一训练invocation `eee5a6fbad40411cac2f05e00520b079` 已active/exited、PID0、success/exit0。实际审计SHA `423ad58a3d441d26f174174bc68824a59ccd453b2e49f0577888530a81083b0b`；实际官方CPU重载SHA `491d6e4e8868edd0c0a222ceb1e1ed5cc1c5b2c3a8b88f0f4a053895aed1a729`，7项候选文件与416项运行源码及双模型tensor身份独立核对一致。内部最低MemFree20.803394GiB、host-watch21.006046GiB、guard CUDAfree21.071632GiB，均保留原20GiB下限。固定训练窗诊断仅有小幅改善，不是held-out或准入证据。下一步仍需独立审批并完成原协议formal；无PPO自动授权。详见 `docs/FC_P026_K4_TERMINAL_REVIEW_20261006.md`。以下此前运行记录保留为历史。

**当前里程碑：FC-P026 K1原始完整正式评估终态FAIL；匹配K4训练已实际启动。** K1正式unit `fluid-control-fcp026-k1-formal-20261006.service`、invocation `c039836ab63246ff8772dad66e1b46e5` 已于21:58:17 UTC成功退出。终态receipt SHA `f2f7a50a26177c65ee048b58fb20df0aee7f4cfa42aed0edd857f08d911ef948`，35项输出哈希全部独立重算一致，八个阶段容器均exit0且无OOM。validation10与dynamic6端点诊断通过，但原62点窗口仅1/6联合通过：总Cd 5/6、后Cl脉动RMS 2/6、后Cl均值4/6；四个旋转RMS误差为0.068339/0.122262/0.068218/0.081354，均高于约0.0294限值。因此 `DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL`，无科学准入、PPO或frozen访问。完整独立复核见 `docs/FC_P026_K1_FORMAL_TERMINAL_REVIEW_20261006.md`。

Root随后签发独立K4审批`ef2ddd3`（SHA `fca9c5a1106c85fb55a54590784453bd5246d55c21bbaa8ded1e3c72562b3e86`）。实际unit `fluid-control-fcp026-history-k4-20261006.service`、invocation `eee5a6fbad40411cac2f05e00520b079` 已于22:00:26 UTC启动；观测到官方容器 `79a39768…75e3`、`--history-k 4`、GPU96%，启动preflight CUDA free32.9703GiB、MemAvailable115.0164GiB。这里仅证明K4真实运行，不是完成或精度结论；仍须完成1368窗/171更新、终态审计/官方重载和独立原完整formal，且无PPO自动授权。

## 历史记录（以下为当时状态，不代表当前仍在运行）

**FC-P026 K1原始完整正式评估启动观察。** Main user unit `fluid-control-fcp026-k1-formal-20261006.service`、invocation `c039836ab63246ff8772dad66e1b46e5` 于21:21:31 UTC启动；观测时处于validation10，实际容器 `4a9cea6bd3a1…43d2` 使用官方镜像 `b40d5888…a22e` 和GPU0。审批SHA `2d15c323…0000`，411文件正式源链 `ff8b742a…fe24`，runner `03c5862e…c0f3`，终态运行时manifest `277ec97a…1f70`。这是历史运行中观察；终态见上文。

**当前里程碑：P026 K1已完成1368窗/171次更新，真实终态完整性检查及官方CPU双模型重载均已通过独立复核。** 同一训练invocation `b3759e7e1acc4de7a1aa9f6e8d38de9a` 已成功退出、PID0；实际检查收据SHA `fa26bf47b6eb448e36046973a479e771b2d37eb605d9630b9022b392ce30d944`，官方CPU重载收据SHA `980698fd335a7a536e358ced26b9f69236e8d101d1f038a46eaf6a0c84f67028`。七个候选文件、两个模型张量、协议、源码及实际容器身份一致。训练守卫最小CUDA空闲20.7337GiB，主机物理空闲最小20.9367GiB，内部采样最小20.7402GiB；均未低于20GiB。前两次检查失败及其原始unit证据均保留；兼容修复仅处理JSON `1`/`1.0`和诊断汇总最多2ULP的Python版本舍入差，不改训练、模型、数据或科学门槛。

HydroGym历史运行时集成已提交 `23711cc`，运行镜像准备已提交 `e542ff1`，不再是待提交状态。报告 `docs/FC_P026_K1_TERMINAL_REVIEW_20261006.md`；只读执行证据 `artifacts/fcp026_k1_terminal_review_20261006/`。**这只是完整性与可加载性通过：尚无P026正式评估结果、科学准入模型或新PPO。** 正式评估审批另行准备，未凭准备材料宣称已经启动；K4需独立批准并与主机其他重任务错开。尚未增加科学CSV结果行。

### 更早历史（以下为当时状态，不代表当前仍在训练或待提交）

**P026 HydroGym显式历史运行时与PPO身份链已完成Root的canonical内容与CPU证据复核，当前仅待最终提交。** 六个生产文件保持既有legacy/direct-CFD路径，并仅为P026增加显式K1/K4历史缓冲、原子reset/step/snapshot恢复、候选身份与正式终态证明绑定；MPC仍明确不在本次范围。Root重新分进程运行canonical fake-HydroGym 15项和readiness/legacy 60项，全部通过；另一个保留的受限host CPU unit用真实HydroGym `PDEBase`/`FlowEnv`、小型mock网络和合成数据通过1项生命周期测试。不可变证据位于 `artifacts/fcp026_hydrogym_actual_core_canonical_cpu_20261006/`，`SHA256SUMS` SHA为 `b0ff32a762b13736da9f22a5f2ebeb4b476fc7ac6dc01e5576a94bd29a49674d`。后者不是官方模型、真实CFD、PPO或科学证据。当前没有启动PPO、没有访问新模型/HDF，也没有改变运行中的K1训练、数值门槛或成功CFD-only基线。

**历史阶段记录（训练后检查工具入库时）：HydroGym历史输入接口当时仍在并行实现。** GitLab提交 `7d0a12c` 包含独立审查后的终态检查、官方CPU双模型重载工具及更新后的阶段计划。Root在主机重新运行38项小型CPU测试，全部通过（1.55秒）；这只是工具验证，尚未对真实新候选执行检查或重载。该段中的K1进度334/1368是当时观测值，不是当前实时状态；训练实例未因本次接口集成而重启或修改。训练后仍按原完整协议评价，K4另行批准运行。

当前控制接口实现保持现有CFD-only案例不变：每个HydroGym环境独立保存FNO历史，使用实际施加的动作推进，和62点受力奖励历史分开；K1/K4身份必须明确传递。代码和CPU证据已完成canonical复核但尚未最终提交，也未启动新PPO或证明闭环收益。训练后检查的不可变源码、检查结果与正式评估候选文件仍必须精确对应。所有工作仍指向合格代理辅助的真实CFD闭环，不能以工程测试代替目标完成。

**K1训练持续运行；P026正式评估程序已完成独立软件审查。** 同一训练实例日志持续增长，已观察到93窗完成，物理空闲约24GiB，双20GiB守卫保持运行。新的正式评估程序保留原validation10/dynamic6/force-window及全部门槛，只新增明确的K1/K4身份和历史输入处理；实际源文件、基准Git树与七个允许修改的文件逐个核对。Root现有目录34项评估程序/旧dev30回归测试通过，独立联合57项通过；报告 `docs/P026_FORMAL_RUNNER_INDEPENDENT_REVIEW_20261006.md`。尚未执行正式评估，必须等待真实训练终态、独立模型检查和官方重载。未改变运行中的训练副本或数据。

**P026单帧K1完整对照训练已在GPU实际运行。** 主机user unit `fluid-control-fcp026-history-k1-20261005.service`，invocation `b3759e7e1acc4de7a1aa9f6e8d38de9a`；官方容器 `ed4f0ad7a42621712b6689d3f694ed90067f154ed7bf78c72e0993bbad930f65` 于19:27:46 UTC启动。审批755cd99/SHA `1088285e4e13c7e3553009dd511436e9a5da976e93eed44d6bb0c28ab9515aa3`；44个真实训练文件已在启动时逐个核验并仅对这些只读文件做缓存建议。计划1368窗/171更新，原目标和预算不变。早期观察已实际完成5窗，GPU96%，MemFree约26.36GiB；这些是时间点观察，不是固定实时数字。双20GiB守卫持续运行。看板注册到同一真实任务并刷新现有Chrome页；当前进度以日志为准。尚未形成终态候选或新精度结论，K4及正式科学评估另行执行。以下“未启动”为历史。

**P026正式评估入口的历史输入测试已完成，正在进行训练执行前总检查。** 实际evaluate/force-window的CLI主函数在隔离软件测试中运行；K1指标与旧调用完全一致，K4未来真实状态修改不影响预测。Root六项补充测试1.66秒通过，独立六项1.70秒通过；报告 `docs/FC_P026_FORMAL_CALLER_REVIEW_20261005.md` 清楚区分模拟模型加载的软件测试与后续真实候选评估。测试曾新建60张合成图，已完整移至repo外engineering_quarantine，未进入看板或科研结果；最终测试使用独立临时工作目录。所有已通过工程条件正在汇总为K1训练执行检查，尚未启动GPU。正式科学评估仍需真实训练终态和独立审核后另行执行。

**P026双模型加载已接入并通过独立检查，正式评估入口补充测试中。** 官方FNO流场模型仍为6通道，受力模型按单帧/四帧分别为6/18通道；新模型身份、训练协议、两个模型来源及实际导入的历史处理代码均核对，旧模型路径保持不变。Root在现有目录运行33项加载/历史测试及23项旧P015/P018回归测试通过。报告 `docs/FC_P026_ROLE_LOADER_REVIEW_20261005.md`；这不代表新候选精度或HydroGym闭环已通过。训练源码固定在 `artifacts/fcp026_history_training_source_20261005_immutable`，正式GPU训练尚未启动。看板已支持真实窗口与更新事件，并正确识别正在执行的oneshot任务；目前仍显示已完成的资源试验，不冒充训练运行中。

**P026真实训练窗口和采样顺序预检查已实跑并独立复核。** 官方CPU容器 `2fa6d157760a8ee65befa76442aa745bc89c06493a2c690edaf68b28e642698c` 退出0、无OOM；实际1368个唯一窗口、1300完整历史/68首帧填充和原官方采样顺序均通过。收据 `artifacts/fcp026_training_inventory_cpu_20261005/inventory.json` SHA `76c84cae2e08595d5326159926396ed8b1bc31a6c1fdd09a92bbae5b9ef2a526`。报告 `docs/FC_P026_TRAINING_INVENTORY_CPU_REVIEW_20261005.md` 包含资源守卫启动脚本的独立检查；Root与独立各18项准备测试通过。这里只遍历真实HDF元数据和官方sampler，没有训练或全文件字节检查；启动时仍须逐个核验44个训练文件。正式模型加载/历史评估接口正在隔离测试，完整GPU训练尚未启动。

**P026完整训练程序已通过独立CPU审查，尚未启动训练。** 最终程序SHA `562d268545ba5cd2559374f4bd8bd34bf59a2e49f2e886e4e286bb12aac5d49e`，独立相关41项测试通过，Root与历史推理模块联合44项测试通过。单帧与四帧均使用原1368窗口、171更新，明确区分两个模型来源，保留原目标函数与终态官方保存重载，并分别报告有完整历史和需要首帧填充的诊断窗口。报告 `docs/FC_P026_TRAINER_CPU_REVIEW_20261005.md`。仍在完成真实全数据窗口清单预检查、正式模型加载/评估接入及资源守卫启动脚本；没有新精度结果或PPO。物理平均升力10%暂不改，条件性15%方案不用于掩盖代理预测误差。

**P026保存重载独立执行审查完成。** 容器、固定官方镜像、只读来源与专用输出已核验；三个工程目录共六个文件逐个SHA复核一致。完整报告 `docs/FC_P026_CHECKPOINT_CPU_REVIEW_20261005.md`。工程文件仍为非候选，不会改名充当已训练模型。当前实际开发为完整训练程序与正式评估历史调用，尚无新的GPU训练或PPO；本地看板已显示P026单窗口实测而非P025旧任务。

**P026官方保存重载已实际完成，独立执行复核中。** 固定官方CPU容器 `db6cb7dde8455bec7ea9b33b00ae151dd5b867f4785af6484d3d4236a7ac3f6b` 退出0、无OOM，生成3个明确标识非候选的工程文件：flowK1、aeroK1、aeroK4。官方save/load重载后的完整张量、映射、epoch和元数据均按程序检查一致；receiptSHA `8b1b48930eeab818921a1a287d69b02769635faa912c643c219150af03af4665`，路径 `artifacts/fcp026_cpu_checkpoint_engineering_20261005/engineering_fixtures/engineering_receipt.json`。无优化器/训练/GPU/候选。只证明工程保存兼容，K4作为旧K1的拒绝检查仅覆盖测试用metadata校验器，不是所有生产调用。正式评估调用与完整1368窗训练程序正在分别隔离实现，尚未批准GPU训练。

**FC-E036：P026资源试验独立终态审查通过（仅工程结论）。** 报告 `docs/FC_P026_RESOURCE_TERMINAL_REVIEW_20261005.md` 核验实际实例/镜像/源码、K1/K4预测及目标完全一致、100次flow及各10次混合分块、非零历史梯度。外部14次采样最低MemFree29.161835GiB，内部最低29.121925GiB，guard退出0；无优化器或候选。后续官方保存重载脚本和正式评估调用接入正在隔离准备，未启动完整训练。Root还在固定官方镜像实际核验save/load_checkpoint来源SHA `0d26a62251c3724a1ceebfa1daa1bb5ba9dcdc5e73a0ded3ccb955355af2f78e`，不编造API。磁盘当前约300GB可用，足够本次少量工程重载文件；不据此启动新CFD或大批数据生成。以下“复核中”为历史。

**P026实际GPU无更新资源试验已正常结束，独立终态复核中。** 实例 `8667f3c9c82146d6ab861d634c460107` 已exited/success/PID0；结果SHA `8e1113efc903c6c95cd24755d8c9b081fb098a01ce60aa3c2477e932052f3589`。官方容器 `6c84aa2ab9551d80a2faef9c289259fd550be32a97c799f842b129517b8051aa`、审批SHA `77e9129e150cd383e38ac1051b33ed95c867d25d3f1f8a19fdd6f8048938277f`；源码ee13932，审批40abd5f。单个真实warm训练窗816，两臂各10个混合20样本分块前向/反向，共享100步冻结流场。K1/K4各3.277/3.047秒，峰值allocated3908428288/4004334592字节，reserved4651483136/4529848320字节；K4新增288系数梯度范数0.0462021，初始化预测与K1观测差0。内部最低MemFree29.121925GiB、MemAvailable107.817810GiB。无优化器/更新/候选/heldout/PPO；这不是完整训练或科学准入。结果位于 `artifacts/fcp026_history_resource_20261005/result.json`，实际容器身份保存于同目录runtime_container.json。下一步完成独立复核和官方保存重载/正式历史调用集成后才批准完整对照。以下“GPU尚未启动”为历史。

**P026显式推理适配器初版通过CPU测试；GPU试验仍待最后监控修复。** 新增项目代码 `scripts/p026_history_inference.py`，复用已审查历史打包/移位与旧raw输出合并，不修改官方网络。实现方和Root各10项CPU测试通过；K1旧输出一致，K4缺18通道输入/当前状态、mask、动作不一致会拒绝；显式历史clone/reset避免环境串扰。它尚未接入正式评估或HydroGym生产调用，未声称已完成在线控制。GPU资源程序数值路径27项测试通过，但监控进程退出时的容器启动竞争处理仍需修复及复核，当前未启动GPU任务。

**P026历史受力目标函数已通过20项CPU测试及独立审查。** K1在测试中与原P013预测/损失/梯度完全一致；K4因果帧序、100步目标、无未来AR输入和分块梯度一致性通过。实现与覆盖范围见 `docs/FC_P026_HISTORY_OBJECTIVE_CPU_REVIEW_20261005.md`。正在准备单个真实训练窗口、K1/K4各一次不更新权重的GPU资源程序；尚未批准GPU执行。正式评估和HydroGym推理中仍有旧6通道调用，必须明确适配后才允许完整训练与后续部署，不能直接给旧调用换K4权重。

**P026真实HDF接入已实跑并独立复核；完整对照设计已审查。** 官方CPU容器 `a8c735d1a2b7d1ca500ebface7e45488eb29b63ba0b2195a5e500a0c2669929c` 退出0，三类数据各两个窗口的目标/元数据/单帧输入完全一致，四帧历史和起点填充符合真实reader数据；只验证六个代表窗口，不是44文件全部字段审计。详见 `docs/FC_P026_REAL_HDF_CPU_REVIEW_20261005.md`。下一步正在准备共享历史受力目标函数及独立测试；完整设计见 `docs/FC_P026_HISTORY_COMPARISON_PLAN_20261005.md`，各臂1368窗口/171更新，原指标不变。GPU资源试验和训练尚未启动或批准。以下“真实HDF准备中”为历史。

**P026短历史工程进展：CPU适配器与官方FNO实跑验证通过，尚无新训练。** `scripts/p026_state_history.py`是项目适配代码，复用现有官方HDF5Reader，不声称是官方新增API。Root/独立各13项CPU测试通过；实际官方镜像容器 `9e4b793a92d5ef747f031a6e697821dcba173babd4c0ef0e4b9d34a2b7b26d84` 于18:31UTC退出0。小网格合成工程测试中，K1/K4零新增权重输出与旧模型差为0；非零历史权重有有限非零输入梯度，连续两步移位保留梯度。详见 `docs/FC_P026_OFFICIAL_CPU_EXECUTION_REVIEW_20261005.md`。这不是真实CFD精度、完整100步训练或GPU容量证据。真实HDF集成验证准备中；训练/正式评估/在线重置都必须使用一致历史约定。物理指标与最终闭环目标不变。

**最新状态：P025已结束并完成独立复核（FC-E035）；尚未达成代理辅助闭环目标。** 同一运行实例 `124d6521045a41cd9dcf5f35edff6712` 已 exited/success、PID0，结果SHA `7648df661439f530984504538bfaef3d1a602c1b960fd914fcba2f016fc4e74c`。16次更新、96窗口反向、36评估窗口完成，原参数保持固定；没有保存候选、没有新PPO。单步平均升力预测偏差平方比初态增加0.1473%，部分波动误差改善仍不足以满足原局部对照条件。外部最低物理空闲28.270GiB，内部最低28.239GiB，双20GiB约束满足。

按实验前约定，结束仅调整96个受力输入系数及统计损失的路线，不追加权重/尺度扫描。下一项是保留现有官方FNO、冻结流场预测器和44条真实训练轨迹，比较单帧与四帧历史输入；目前只授权隔离CPU适配器及测试，不是GPU训练或已证明的改进。1300个窗口有完整历史、68个需要明确标识初始帧填充；两组使用一致的动作时间语义，连续预测不得输入未来真实流场。真实平均升力10%暂不改，既有条件性15%方案仍保留。下方运行描述均为历史。

**2026-10-05 18:17 UTC：P025统计监督对照已在GPU运行。** Root及独立各20项相关CPU测试通过；源码4d12c75，审批SHA `6fb2f810e395fdc477339352d677616ad782b7edca6535f282710e2ae2573700`，协议SHA `7f6cc9c66df5a23ecb0b56a6a6bf9da5ffc1ec494acba9e586397f113f580598`。实际unit `fluid-control-fcp025-isolated-statistics-20261005.service`、invocation `124d6521045a41cd9dcf5f35edff6712`、PID448938、官方容器 `4aa9902af3c0dc4261493c8ebc10d6867ed567fb36b20a0fda300c874293b9c5` 正常运行；最近看板实际核验4/16更新。

仅新增96系数训练、旧权重固定，训练目标加入固定5/16的四项归一化均值/RMS误差；完整100步梯度，和原P023HIGH对照。开始更新证明此前精确初态重现检查已通过；尚无终态精度结论。双20GiB守卫和1200秒内限生效。详见`docs/FC_P025_RUNNING_EXECUTION_20261005.json`。若不支持，结束该局部统计损失路线，不扫权重/尺度。只读备选评审提出短历史条件化，但未批准实施或更改场景。下方早期“尚未启动”为历史。

**2026-10-05 18:06 UTC：P024响应诊断已完整结束并独立复核（FC-E034）。** 实际invocation `b85dcf9d70e443fe92ae4ef5d72d3c76`已exited/success/PID0；结果SHA `6eecbd75c5b18cd821d6cf814c6d9319f755bf8dc70f2eb0c0e7e9326bf8ba0b`。60窗口/6000配对前向、全部统计与系数重算一致，0/1精确重现P023原始预测；无优化器、候选或PPO。外部最低物理空闲30.915GiB，双20GiB满足。

放大8倍改善单步均值但恶化连续预测均值；64倍进一步恶化单步波形及原始目标，因此不能直接采用放大系数。详见`docs/FC_P024_TERMINAL_REVIEW_20261005.md`。下一项P025只批准CPU实现/审查：仍只训练96个输入系数，原目标增加此前明确的均值/RMS统计项，固定16更新，以原P023HIGH为精确可比对照；完整因果100步梯度。目标是测量是否能改变误差权衡方向，不是继续扫放大倍数。若仍失败则结束这条局部统计损失路线。GPU目前未运行；原准入和完整闭环目标不变。

**2026-10-05 17:53 UTC：P023完整结束并通过独立执行复核，但局部精度条件未全部满足（FC-E033）。** 同一invocation39aec740已exited/success/PID0。结果SHA `adfdd9a86cedf75019b655fe360b64b09d1aa51ce3de2f9166d0d80e296007cb`；32更新、192窗口反向、72诊断窗口及全部重复聚合一致。唯一失败条件是HIGH单步平均升力预测偏差平方比初态增加0.070729%；其余比较条件通过，但连续预测RMS绝对误差仅改善约0.0411%，不能称为完整修复。

两组关闭新增力输入后原始预测精确回到初态，说明测得变化来自新增96系数；参数变化幅度和输出变化不成简单比例。双20GiB满足，外部最低物理空闲28.611729GiB。详见`docs/FC_P023_TERMINAL_REVIEW_20261005.md`。当前无GPU训练；下一步只准备固定系数方向的有限尺度响应诊断，先重现原结果，不启动全数据训练/PPO或改变门槛。原P018正式FAIL与完整闭环未完成结论保持。以下运行描述为历史。

**2026-10-05 17:38 UTC：P023两学习率受力输入对照已在GPU运行。** Root与独立各25项相关CPU测试通过后批准f13a6a0；审批SHA `714db1f9d09b0ee7037953d6b807b3341cf7a736889d5c5fa98e5ae8c0116cf0`，协议SHA `4591677dc4c7515e40f13ce311a7c66ceec711b5083d98b397fe37ff138de3e1`。实际unit `fluid-control-fcp023-input-block-20261005.service`、invocation `39aec740a9914226bb1f74c2d29e7917`、MainPID387792；官方镜像容器 `847088c7789df80d028de330e60f1d193b7656b4a8110d002d710ff8c416e82d` 正在运行，非启动意图。

44训练HDF及274原始来源已核验；LOW组已完成4/16更新，正在第5次，HIGH尚未开始。17:38观察GPU96%、物理空闲约28.93GiB、可用108.73GiB；两项20GiB守卫保持。只学习新增96系数，旧官方FNO权重固定，两组均因果受力输入；共32更新/192窗口反向及72诊断窗口前向，无候选保存/heldout/PPO。输出`artifacts/fcp023_input_block_20261005`；结束后须独立复核，当前无精度改善结论。

17:32UTC一小时指标复评已完成并同步19244aa：现有真实CFD-only平均升力1.99%/3.87%已符合10%，放宽15%不能解决代理波动预测失败，因此本轮不改指标。后续真实CFD若仅平均载荷超限，再评估15%并列旧标准。以下准备记录为历史。

**2026-10-05 17:29 UTC：P023参数隔离CPU验证完成，真实数据对照程序正在准备。** 官方CPU容器实跑通过：零输入重现旧模型、非零输入与直接设置权重的输出/梯度一致、100步完整图与检查点重算一致、旧权重不变；Root和独立9项CPU测试通过。实现与真实运行记录已同步0f3ee83。这是工程验证，不是流场精度或闭环改善。

已委派实现和独立评估职责，按`docs/FC_P023_INPUT_BLOCK_COMPARISON_PLAN_20261005.md`准备只学习96个新输入系数的两步长对照；真实训练尚未启动，GPU执行仍需最终代码审查。17:28UTC实查GPU0%，P022已exited/PID0，物理空闲约35.91GiB、可用115.69GiB。当前不是GPU训练中，也没有新PPO。17:32UTC物理平均升力指标复评保留，10%目前不变；以下状态为历史。

**P022独立终态复核完成（FC-E032）**：全部32次更新/192反向、48个端点窗口、八个panel聚合及判据重算一致；local_support=false。外部490个资源样本最低free26.807617GiB/available106.579044GiB，双20GiB满足。详见`docs/FC_P022_TERMINAL_REVIEW_20261005.md`，主结果SHA仍为69e5d4a8。当前无GPU任务；只批准P023设计准备：固定旧权重、仅学习新增96个受力输入系数的两个固定步长对照。尚未批准实现或GPU，必须先确认参数隔离及无旧权重衰减/状态改变；不追加相同P022训练，不改物理或代理指标。17:32UTC指标复评仍待到时。

**2026-10-05 17:05 UTC：P022两组训练均已完成，记录的局部支持条件未满足，独立终态复核中。** 同一invocation `f1f3f7b31e70440693da2661a12cfc04`已active/exited、success、PID0、code1/status0；结果SHA `69e5d4a8a93b8036187a18a5c40cb270aec53462a49ddab94417fa0b908096d8`，耗时979.19秒。记录32次优化更新、192个窗口反向、48个端点评估窗口；不是新的完整模型准入或闭环成功。

加入当前受力的B组相对零输入A组四项均值/幅值平方误差较小；但相对共同初态，H1均值平方误差增加1.849754e-6、去均值波形MSE增加0.0002539900，AR均值平方误差增加5.336563e-6，且AR原始目标略劣于A，因此local_support=false。不能只按训练loss下降放行PPO。内层守卫492样本、最低CUDAfree26.804GiB、exit0；外部资源与完整端点统计正独立复核。未保存候选、未访问heldout；下一步仅分析新输入学习尺度与既有权重更新的影响，不自动追加同一训练或降低门槛。以下运行记录为历史。

**2026-10-05 16:49 UTC：P022受力输入对照任务已实际启动，尚无精度结论。** 计划`docs/FC_P022_CAUSAL_CONDITIONING_PLAN_20261005.md`；Root和独立复核均19项CPU测试通过。代码`074d979`，审批SHA `273fe63f097049fe28f9d3f6f241308a95eefb6fab23ca2dbe0d23559043cf78`，协议SHA `490ca6fe5334621d0fbade9552b8bb6d884f4561a6d99202f8ef68b22d35e73d`。实际unit `fluid-control-fcp022-causal-conditioning-20261005.service`、invocation `f1f3f7b31e70440693da2661a12cfc04`、容器 `ab4530e9fd0772df87ab1570258290ece1961d4241a5a289cd0add1d94e5c298`，官方镜像b40d5888；日志确认44训练HDF与274原始力来源核验及限定文件缓存建议完成。

六个既有训练窗、两组各16次更新，六窗梯度平均后更新，原始目标与优化器相同；唯一对照是新增四个输入为零或严格同刻受力。完整H100梯度，不输入未来真实力。任务先缓存冻结流场历史，再评估初态并训练，不能把启动等同于已完成优化。双20GiB守卫及30分钟内层时限不变。输出`artifacts/fcp022_causal_conditioning_20261005`，无模型候选保存/heldout/PPO；终态需独立复核。以下准备与P021阶段记录保留为历史。

**2026-10-05 16:38 UTC：P021真实尺寸资源检查r2完成，工程复核通过。** 两个对照各100次前向和100次反向重算，28项梯度有限、原模型与扩展初态参数不变；总29.05秒，单组前后向约4.60–4.86秒，最高CUDA reserved3.779GiB。外部物理空闲最低28.596GiB，可用108.325GiB，双20GiB满足。结果SHA `975fc40bbb88d5d0ab3d239ee0ce994bc0635aabcad9b7d74568bf73f1a8ce7a`，Root与独立审查核对实际终态和记录。详见`docs/FC_P021_RESOURCE_TERMINAL_REVIEW_20261005.md`。

尚未训练、无新候选或PPO；此次新增输入权重仍全零，不能推断学习后的反馈稳定性或精度。下一步已授权暂存CPU准备六窗、每组16次更新的零输入/当前受力输入对照，原目标和优化器相同；GPU训练须实现审查后另行批准。以下“正在运行”为此前观察。

**2026-10-05 16:37 UTC：P021真实尺寸资源检查r2已启动，尚无结论。** Root与独立审查各15项CPU测试通过；代码877911e、恢复518684f。首轮invocation `f2b325db3b8b49c2ad1647ad41ec7e15`在模型创建前因Python导入后的物理空闲约29.4GiB低于额外30GiB启动要求退出，无20GiB违规；failure SHA `30c1d17c7692a85ff60f62e11eb5700d1fdda1a49163e543253f44cf828317b5`，原证据保留。

r2只对44个同描述符重验SHA的只读训练HDF文件发送干净缓存释放建议，不写数据、不全局清缓存、不改模型/协议/内存门槛。恢复审批SHA `3bec0fb8f853efaf308f1fa0a70aef16059a5f65ca99d8aa751377940960f6a1`。实际unit `fluid-control-fcp021-causal-resource-r2-20261005.service`、invocation `abb778c24a494eaa881eaa2669c5b57f`、容器 `69ccfb0be99f24872432a6cf069712d008f51cc89089aee8c3c22b3b2833a63d`，官方image b40d5888。启动CUDAfree36.589GiB、MemAvailable115.087GiB仅是当时观测。

本次固定训练窗816、两个输入对照各一次完整H100前后向，不创建优化器/更新或保存模型，无heldout/PPO/准入。双20GiB守卫持续有效。结果位置 `artifacts/fcp021_causal_resource_probe_r2_20261005`；后续需真实终态与独立复核才能安排有限训练。用户新一小时复评时间17:32UTC（北京时间01:32），物理平均升力10%目前不变，详情见DECISIONS。以下较早“仅准备”描述为历史。

**当前阶段：P021 CPU工程复核通过，真实尺寸GPU资源检查仅在准备**。模块SHA `bcefcad2fa622e5b69133725aa8d39db5a1b0a41a1b5e6c7dcb1f6b7d99a4962`、测试SHA `a3dfb4b88d70eabb72d2df560444f828a6041b812e7cba9d06d1785139f9b5eb`，Root/实现/独立审查均24项CPU测试通过。Root另在固定官方CPU-only容器实跑微型FNO H100，完整图与checkpoint输出及28项梯度最大差0，10项谱虚部分量梯度非零，未用GPU或优化器。代码与证据已同步`1cc12fe`，详见`docs/FC_P021_CPU_REVIEW_20261005.md`；这不证明全尺寸精度或资源可行性。

当前仅按`docs/FC_P021_RESOURCE_PROBE_PLAN_20261005.md`准备一个既有真实训练窗口816的前向/反向检查程序及CPU测试；GPU执行尚未批准。计划使用原P018权重和精确同刻受力、完整100步递推，不更新/保存模型；两项20GiB要求不变。没有新的模型训练、代理准入或PPO/CFD闭环结果。以下CPU实现之前的描述为历史。

**当前阶段：六窗因果力输入审计完成（FC-E031），P021仅CPU工程获批**。独立从实际raw系数与配置重算完整时间审计，274个源文件SHA和完整结果一致；六窗606个名义端点、每圆柱均有唯一同时间raw力。base/train8的505帧中200帧HDF力实际含下一solver样本的微小插值贡献；train16的101帧使用精确端点，不含该依赖。六个初始力均精确，因此旧AR initial persistence不受影响；旧HDF-lag H1 persistence只能称描述性滞后一帧基线，不能称严格在线因果。精确raw sidecar只覆盖这六窗，不外推到44轨迹/heldout，HDF目标与归一化不改。

审计主结果`artifacts/causal_force_input_audit_20261005/timestamp_audit.json` SHA `72d9117922ef5dbbd3b9f9a5ae193d01c19ac44a39aea30b4dbe5eaf3189d4e2`；详见`docs/CAUSAL_FORCE_TIMESTAMP_AUDIT_20261005.md`。Root提交`225d99f`仅授权P021隔离暂存CPU适配器/梯度与warm-start工程测试，不授权GPU、训练或架构部署。P020局部条件未满足和P018正式FAIL保持有效，无新候选/准入/PPO；所有指标不变。以下早期状态保留为历史。

**当前阶段：P020两组有限步数对照完整结束，但预声明局部支持条件未满足（FC-E030，2026-10-05）**。同一invocation `4567f6d393414bba8baf2239d16960a7`已active/exited、success、MainPID0、code1/status0；guard exit0。独立复核32次更新、192个唯一六窗反向事件、全部有限数值、源/候选/审批身份、44HDF启动核验，以及八次端点panel的聚合与判据重算。两臂初态及所有端点重复逐窗数值完全一致，但重复差为零不是严格误差上界。结果SHA `a7c0d0c41b35391e22d07fb223a5ed243891ccdd4759815e9bf08b82670b5042`。

B统计监督组相对A原目标组的H1/AR均值平方误差与RMS幅值平方误差四项均改善；相对共同初态，B的H1均值平方误差增加`4.8015515e-6`，H1 centered residual MSE增加`0.0002363738182`，所以`LOCAL_CONDITIONS_NOT_MET`、local_support=false。两域原目标均下降仍不能覆盖这些失败。382个host样本最低available106.712757/free24.620411GiB，内层CUDAfree最低24.622280GiB，双20GiB满足。无候选保存、无heldout/PPO或科学准入；P018原正式FAIL继续有效，所有门槛未改。下一科学方向由Root分析决定，当前力条件化仅待分析，不是已批准执行。详见`docs/FC_P020_TERMINAL_REVIEW_20261005.md`。以下运行与准备描述保留为历史。

**下一步授权范围更新**：仅准备当前力条件化的数据/因果可用性与persistence baseline调查；未批准GPU、架构实现或新训练。以下旧“下一方向待分析”保留其时间语境，不扩大本准备授权。

**当前阶段：P020 两组有限步数训练对照正在运行（2026-10-05 15:37 UTC 观察）**。原损失与新增对称 H1/AR 升力均值、波动幅值监督，均从 P018 终态开始，固定六个真实训练窗口、每组六窗平均后更新16次。Root/实现/独立审查各17项CPU测试通过后批准执行，代码 c315cbc，审批SHA `ab69dfc9559a7e5458aba08c85b57d8d4aa41529ac7980e28f739ef667fab30a`。实际 unit `fluid-control-fcp020-symmetric-statistics-20261005.service`，invocation `4567f6d393414bba8baf2239d16960a7`，容器 `7d64bb2ade48984720b91f02bed5ed3cb4857109ac6f9142670ee02b84454824`；启动证据见 `docs/FC_P020_RUNNING_EXECUTION_20261005.json`。

15:37的实时接口确认原损失组完成7/16更新，统计监督组尚未开始；该进度仅是有时间戳的观察，当前状态以实际进程和日志为准。仍须独立复核完整结果；无候选保存、无heldout、无PPO或新增CFD闭环成功。原0.10物理平均升力限制和全部代理精度要求不变。双20GiB守卫与30分钟诊断时限持续生效。页面显示两组实际更新数，已通过20项相关UI数据测试。以下历史“P020未批准”被本段更新，不修改历史结果。

**当前阶段：P019只读梯度诊断完成，支持局部跨窗梯度干扰（FC-E029，2026-10-05）**。同一invocation `275365360254440aba18ed96aac58630`已active/exited、success、MainPID0；60个唯一梯度计算与精确journal一致，guard exit0。结果SHA `1bd66e3cbf7c1200ff0af96d23bd62803d59422129eec5c5dddf7615fbcb183f`。每窗原目标两次均精确重现P018终态；五非零窗AR-RMS平方误差沿负原始聚合梯度的方向导数为+0.859184，沿六窗原目标梯度为+0.468014，重复符号稳定，而各窗自己的AR-RMS方向导数六个均为负。解释限于聚合造成的局部跨窗干扰，不是AdamW实际方向、全局病因或loss改动有效性的证明。其余三项聚合统计量方向为负。

108个host资源样本最低available102.286327GiB/free20.551254GiB，内层CUDAfree最低20.544643GiB；双20GiB守卫满足但余量很窄。无optimizer/update/savecandidate/heldout/PPO，模型不变。完整出处和局限见`docs/FC_P019_TERMINAL_REVIEW_20261005.md`；实际启动证据`docs/FC_P019_RUNNING_EXECUTION_20261005.json`保留。下一步仅准备P020两臂16更新、固定六个训练窗且全部六窗梯度平均的原目标/对称尾窗统计loss对照；统计摘要另聚焦五个非零窗，尚未批准GPU或执行。P018原正式FAIL继续有效，物理平均载荷0.10与代理误差指标保持区分，本记录不修改门槛。以下P018/P019旧计划状态为历史，不覆盖本段。

**当前阶段：FC-P018原完整正式评估结束，独立复核FAIL（FC-E028，2026-10-05）**。同一formal invocation `ef589f7dbeff4fa0ab064309409971ad`已保留为active/exited、success、MainPID0、ExecMainCode1/Status0；不是运行中的默认exit0。18项receipt文件SHA与三阶段身份全部核对，冻结原审计器从raw force-window重算得到完全相同字典。完整receipt SHA `d4d3f85a79e31d50866bb8dbd23ec90e0453cde39ef6b314e3db80344d33869c`；原联合准入 **1/6，Cd5/6、Cl′RMS2/6、meanCl4/6，FAIL**。仅b01 zero联合通过；四个旋转分支RMS误差0.0683247/0.122662/0.0685903/0.0813667仍超过约0.0294限值。validation10与dynamic6的端点组件通过不能覆盖窗口失败。formal守卫1017个样本最低MemAvailable110.045368GiB/MemFree27.460377GiB，均高于20GiB。

同dynamic6协议P009/P015/P018的H100 pooled Cd NRMSE分别0.01813790/0.01929335/0.01791093，macro分别0.01562564/0.01834680/0.01569608，后Cl MAE分别0.08429627/0.08544903/0.08507009；原flow指标全部不变。P018相对P015部分指标改善，但不构成准入或完整波动幅值修复。详见`docs/FC_P018_TERMINAL_REVIEW_20261005.md`。没有新PPO、frozen-test或代理辅助真实CFD成功，项目未完成。下一步仅准备P019 objective/statistic gradient诊断，尚未批准或执行；不盲目追加训练。

物理平均升力0.10条件与代理均值预测误差门槛保持区分，最早15:20 UTC才进入用户授权的条件复评，本记录不修改任何门槛。以下14:30及更早状态按历史保留，不覆盖上述终态。

**当前阶段：P018训练终态与官方CPU双模型重载已核验，原完整正式评估运行中（2026-10-05 14:30 UTC附近核验）**。训练同一invocation `1ca4654aab074278bb2efdfff8dbc1eb`成功退出，171次AdamW更新/1368窗口、44条真实训练轨迹、固定学习率`1.5625e-7`与协议均通过完整性审计；冻结flow和两项lifting bias保持不变。候选审计SHA `03153fa5…2323b9`、completion SHA `bce5fb46…205bf`，不是科学准入。实际官方镜像CPU双模型重载收据SHA `d3625877…91bdde`；外部容器`79e72220…1a6407`退出0，镜像/命令证据见`docs/FC_P018_CPU_DUAL_RELOAD_EXECUTION_20261005.json`。前两次只读默认缓存目录导致的失败保留，成功尝试只将缓存重定向到容器临时目录。

正式评估于14:20 UTC启动，当前精确service `fluid-control-fcp018-posteval-20261005.service`、invocation `ef589f7dbeff4fa0ab064309409971ad`、MainPID `56992`为`activating/start`：这是oneshot任务正在运行，不是空闲或已完成。审批SHA `7b137d46…ed275b`，冻结评估链SHA `6a97e0b4…0efe3f`，不可变外层守卫SHA `4bb55e02…24a0c`；原validation10→dynamic6→force-window→联合审计协议不变，同时守护MemAvailable和MemFree至少20GiB。输出在`artifacts/fcp018_reduced_rate_training_20261005/posteval_fc_p018`，运行日志/资源在`formal_supervision_r1`。尚无完整正式精度结论；固定六个训练窗H1目标改善3.53%、自回归目标恶化0.126%，不等于收敛或准入。下一步完成并独立复核原完整门槛；只有合格候选才进入兼容新PPO与配对真实CFD反馈。P015联合1/6失败仍保留；没有新的代理辅助真实CFD成功，项目未完成。

**用户条件授权，当前不改指标**：按`DECISIONS.md`顶部2026-10-05 14:20 UTC记录，继续尝试一小时后、最早15:20 UTC（北京时间23:20），才可基于证据复评真实CFD的`|后圆柱平均Cl|/基准Cl′RMS <=0.10`要求。这是实际平均横向载荷限制，不是代理预测均值误差门槛；不会自动放宽升力脉动RMS比、减阻、动作或资源限制，也不改变正在运行的P018原正式评估。若随后修订，须登记新版本、并列保留旧10%结果，不回写历史PASS，且仍须真实CFD闭环验证。

以下按时间记录保留为历史；“训练正在运行”等旧描述不覆盖上述当前状态。

**P018全量低学习率对照已启动（2026-10-05 12:45 UTC核验）**：审批`9b44d9e`，Root及独立各58项回归通过。service `fluid-control-fcp018-reduced-rate-20261005.service`，invocation `1ca4654aab074278bb2efdfff8dbc1eb`，实际官方容器`afd47995f5153b1107dee041becbdb4e23c33b7281de7af69d07a621c174c31f`正在运行。相对P015仅优化参数lr由1e-5降为1.5625e-7；同44真实训练轨迹、同P009初态、同1368窗口/171更新与原目标，flow冻结。协议SHA `310f0bdf…04d2d`贯穿训练和保存；终态才评估，不挑中间模型。当前空闲内存约25GiB，双20GiB守卫开启。输出`artifacts/fcp018_reduced_rate_training_20261005/candidate`，外部执行证据见`docs/FC_P018_RUNNING_EXECUTION_20261005.json`。完整终态审计/CPU双模型重载/原formal衔接同步准备。P017独立复核支持局部首步过冲，不代表全数据稳定或准入；最终代理辅助CFD闭环仍未达成。

**P017首步诊断已运行（2026-10-05 12:33 UTC）**：审批`5f5f2c0`，实施/Root/独立各27项CPU测试通过。实际service `fluid-control-fcp017-first-step-20261005.service`，invocation `cfa568f1f4d5412a93628273d93a95a1`，容器`604af989be7d3dae1667c37d7a43afd50d8ec7e5b76798d0fca59bf6f4d5fd29`运行在固定官方镜像。重放P009同六窗首个AdamW更新，比较实际方向和完整/正负1/64位移；不保存模型、不访问heldout/PPO。启动空闲约33GiB，双20GiB守卫与15分钟内限开启。输出`artifacts/fcp017_first_step_diagnostic_20261005`。P016独立终态审查已完成：运行完整但同时改善假设不获支持（FC-E026）；原科学准入不变，项目仍未完成。

**P016检验结束，未支持同时改善（2026-10-05 12:24 UTC）**：同一invocation已保留为exited/0，内部GPU守卫exit0；result SHA `f760d2e7…54248`，记录32次更新/192窗口计算，完整独立终态复核进行中。固定训练面板H1/AR objective分别变化+0.175%/-3.803%；五个非零动作窗口的tail62均值偏差平方分别增加420.873%/208.108%，波形与波动幅值误差虽下降但不满足预声明同时改善要求。没有保存候选，没有PPO或新CFD闭环。下一步检查同批数据上的loss/梯度/更新实现及实际优化轨迹，再决定针对性改进，不盲目增加全量训练。看板`2c4f3cc`已部署，原私有API实测显示P016结束待独立复核及P015联合1/6失败。详见`docs/FC_P016_TERMINAL_OBSERVATION_20261005.md`。

**P016固定六窗检验已启动（2026-10-05 12:13 UTC）**：service `fluid-control-fcp016-fixed-panel-fit-20261005.service`，实际invocation `a95370a65c2b47e0b0e2926261937e33`。代码/方案/数据依赖审批`e40bca9`，实施、Root与独立各15项CPU测试通过；启动重新校验44个真实训练HDF。固定P009初始化、冻结flow，原mixed20目标，六窗梯度平均后更新，共32次；只解释终态，不保存候选，不访问validation/frozen，不运行PPO。官方镜像实际容器已运行、GPU约96%，当时MemFree约29.9GiB；两项内存守卫>=20GiB，内部30分钟时限。外部镜像/命令/挂载证据见`docs/FC_P016_RUNNING_EXECUTION_20261005.json`。这只是正在执行的局部可拟合性检验，不是新精度或控制成功。

**P015完整评估完成、未通过（2026-10-05 12:03 UTC）**：实际formal服务同一invocation成功退出，外部资源守卫完成；18项产物SHA经独立复核一致，原数值审计器重算与存档完全一致。receipt SHA `353004af…95a5b`，gate SHA `1d55e344…954e8`。六个时间窗口联合通过1/6：平均总Cd5/6、后Cl脉动RMS2/6、后Cl均值2/6。四个旋转工况RMS绝对误差为0.067566/0.118587/0.065684/0.080875，仍超过约0.0294的限值。动态H100后Cl MAE为0.085449，略差于P009的0.084296，优于P013的0.127625；pooled总Cd NRMSE为0.019293，差于两者，不能混用macro值0.018347。流场指标仍与冻结亲本完全相同。拒绝本轮代理进入PPO，保留负结果FC-E025；下一步优先审查既有训练误差与采样/表征，提出有明确判别条件的单项改进，不盲目加轮或仅做均值校正。当前没有新PPO、没有新代理辅助CFD闭环，最终目标仍未达成。

**P015正式精度评估已启动（2026-10-05 11:26 UTC）**：service `fluid-control-fcp015-posteval-20261005.service`，实际invocation `c41e61fcaa5f49e3be9e0092c1e19bc3`；原完整数值协议与冻结评估源码不变。外部资源守卫`d4b4d6a`经实现、Root与独立各25项CPU测试通过，实际启动前检查通过；同时监测MemAvailable和MemFree至少20GiB，3小时内部上限，systemd保留180秒清理时间，异常只清理精确匹配本评估的容器。运行记录在候选目录`formal_supervision_r1`，科学结果在`posteval_fc_p015`。刚启动时正在来源检查，尚无正式精度结论、PPO或新闭环结果。完成后核对原联合要求；失败则继续基于实测误差改进，不降低门槛。

**P015训练终态已核验（2026-10-05 11:17 UTC）**：171次更新/1368窗口完成，实际服务同一invocation成功退出。Root与独立审查重算候选完整性一致；completion SHA `9c27e5eb…05fdf`。官方镜像实际CPU双模型重载通过，receipt SHA `925a7dc0…2b18c`，外部Docker退出/镜像/命令证据已单独保存。固定六个train片段的平均objective从0.00624614增至0.00730327；去均值升力波形部分改善但平均偏差多数恶化，不能宣称精度或控制通过。原完整formal已独立核验执行审批，尚未启动；正在补齐外部MemFree守卫，不改变冻结数值评估代码。下一步完整formal→按实际结果决定后续干预或兼容PPO→真实CFD反馈。详见`docs/FC_P015_TERMINAL_STATUS_20261005.md`。最终目标仍未达成。

**P015控制接口准备完成（2026-10-05）**：`20d7170`已推送GitLab，补齐共享dual绑定、candidate readiness、PPO launcher、CFD readiness导出、底层训练入口及真实反馈入口的显式P015身份支持；缺失dual证据不能落入旧单模型路径。Root111项、独立196项及最终增量42项CPU回归通过，原数值审计器、reward、动作与验收标准不变。详见`docs/FC_P015_CONTROL_INTERFACE_READINESS_20261005.md`；这是工程准备，不是PPO/CFD执行或模型准入。当前训练同一invocation已完成57次更新后的六个固定train窗诊断，并继续至58/171（464/1368窗），诊断数值随预定终态保存，尚不判断精度改善。下一步仍为训练终态核验、实际双模型重载和原完整formal，不自动跳过任何失败条件。

**评估衔接准备（2026-10-05 10:06 UTC）**：P015仍在同一invocation训练，10:03浏览器API实测35/171更新、280/1368窗口、日志约11秒新鲜；不是终态。原始数值协议不变的P015正式评估profile、独立CPU双FNO重载验证及保留服务终态finalizer已完成并推送`41d34b5`。Root76项、独立94项CPU测试通过，隔离冻结依赖导入通过；尚未实际重载P015终态或执行formal，不能写成科学通过。finalizer对真实running服务只报告运行、不生成completion。下一步是训练终态→完整候选审计→实际官方镜像双模型CPU重载及外部容器证据→单独批准原完整formal。

**在线接口现状补充**：并行只读检查确认当前真实CFD反馈为真实probes/受力→PPO→动作约束→OpenFOAM，没有在线Curator/FNO调用；FNO现用于代理环境。P015正式结果进入控制绑定仍需显式profile兼容，正在准备，不改变科学门槛。69维策略观测为当前真实状态，不能未经定义直接替换成未来预测。影子预测展示不算FNO对控制决策的贡献；最终必须如实区分代理辅助策略与实际在线预测/控制。

**当前训练（2026-10-05 09:48 UTC）**：FC-P015已按审批`b0c326a`启动，真实unit `fluid-control-fcp015-window-accumulation-20261005.service`，invocation `7842742926284d0c94b0383163d5dc0b`。同P009初始模型、同1368窗口顺序、同官方FNO/loss/AdamW数值；仅改为8窗平均梯度后一次clip/update，共171更新。四个固定诊断时点不参与选择终态。实际running evidence SHA `320dda25…16836`核验完整容器命令、GPU0、90GiB、禁网/只读根目录及train-only挂载。当前实测4/171更新、32/1368窗口、GPU96%、MemAvailable109GiB/MemFree29GiB；这是当时观测。HTML已改为实际P015身份和两种进度计数，原Chrome页已刷新；尚无新精度、PPO或闭环结果。终态审计、P015双模型加载和原formal衔接并行准备，不能以训练运行代替最终准入。

**最新证据（2026-10-05 09:29 UTC）**：P014只读诊断已完成，result SHA `5550140b…e95a7`、unit exited/0、模型张量前后不变，44HDF及残差分解已独立复核。六窗实际H1、AR及总训练目标都变差；H100的AR centered residual MSE六窗均增加，H1三增三降，终态十二域平均残差均为正。这否定“仅纯常数偏移”作为完整解释，不能声称训练原因已查明。Lead不批准额外全训练集仅bias标量校准：它无法改变仍失败的Cl′ RMS，不足以推进整体准入。下一项优先是能改善波形、保持原架构/loss/数据/门槛的单因素优化干预，正在依据实际训练代码设计，不盲目继续原训练或扫参。P013正式FAIL继续有效，尚无新PPO或代理辅助真实CFD成功。

**最新阶段（2026-10-05 09:27 UTC）**：P013完整正式评估已完成，原联合准入FAIL：六个受力窗口joint0/6，平均总Cd5/6、后Cl脉动RMS2/6、后Cl均值0/6。独立重算原审计器得到完全相同结论，18项receipt文件SHA全部一致；receipt SHA `2733c3cb…aa2e7a`。P013不准入新PPO。固定六个train窗口的P014只读目标/升力残差分解已按独立审查方案启动，service `fluid-control-fcp014-train-objective-20261005.service`，invocation `6bba81dba45b46638643f441ba21082f`，审批`e26473d`；原数据与模型不变、无optimizer/backward/save/validation/frozen/PPO。启动前重算44个真实训练HDF，容器保留至少20GiB MemAvailable和MemFree。下一步依据分解结果决定单因素改进；不能以诊断完成代替合格代理与真实CFD闭环。

**当前摘要（2026-10-05，正式评估进行中）**：P013训练和终态完整性已完成，不是科学准入。validation10四份报告SHA与step receipt已独立核对；原端点组件PASS，但同协议P009→P013的H100后Cl MAE从0.0387410增至0.103710，pooled总Cd NRMSE从0.00558671增至0.00963676；start0动作差Cd MAE从0.01920627降至0.01554142。局部动作响应改善不能覆盖升力退步。dynamic6实际服务仍运行，后续force-window及联合准入尚待完成。原看板已更新并重载（8cc8f12），路线图已更新（4ed0c5f）；历史流场图片未冒充新模型结果。当前并行任务：P014固定训练窗口目标/均值与波动误差分解的实现和审查（GPU未批准），以及旧巡检误报的增量修复（尚未部署）。主线仍是合格代理→兼容PPO→真实CFD在线反馈→原减阻/升力联合验收；尚无新PPO或FNO辅助CFD成功。

**最新科学进展（2026-10-05 08:49 UTC）**：固定六个训练窗口诊断完成（result SHA `8e0255c9…ec873`），同窗对比P009：P013的H1后圆柱Cl MAE在6/6变差，自回归MAE在5/6变差；尾62点Cl′ RMS误差H1在2/6变差、AR在4/6变差。流场u/v/p指标全部逐值不变，模型tensor未被诊断修改。这不是精度改善；保留负结果FC-E022，不能据此启动PPO。预定完整正式评估已按单独批准`4ad097c`启动：service `fluid-control-fcp013-posteval-r2-20261005.service`，invocation `7235b2f06282435a89b84964e384c60f`，使用不可变f95048c链；正在validation10阶段，尚无正式验收结论。目标不变，下一步依据完整结果分析误差，禁止降低门槛或盲目加轮数。

**当前阶段（2026-10-05 08:45 UTC）**：FC-P013 r2已完成全部1368次更新并保存终态双FNO；Docker精确容器退出码0，fresh reload通过，冻结流场tensor SHA前后相同。完整候选审计SHA为`1c280b29…704a7`，completion receipt SHA为`3c53a7fb…e2d90`；这只是训练完整性完成，不是精度或控制准入。原终态检查器因临时systemd服务被回收而失败，记录保留；独立复核Docker退出事件、精确invocation日志与12项文件SHA后，专用恢复脚本`7bf45cf`生成如实记录服务已回收的completion，不伪造systemd成功状态。训练guard最低MemAvailable107.423GiB、CUDAfree27.595GiB。

固定六窗口只读诊断已启动：`fluid-control-fcp013-fixed-six-r2-20261005.service`，invocation `ef401d302bf54b468ec217a42492b968`，输出`artifacts/fcp013_independent_force_fno_training_r2_20261005/fixed_six_diagnostics`。批准`c46fa31`，不可变launcher SHA `cf0bc8d5…62bef`。比较P009亲本和P013终态的原H1/free-AR物理误差，禁止优化、选模型、validation/frozen/PPO。下一步查看诊断并执行另行批准的完整正式评估；只有原全部准入通过后才训练兼容PPO并开展真实CFD闭环。最终目标仍未完成。

**当前观测（2026-10-05 08:05 UTC）**：FC-P013 r2仍在运行，已记录888/1368次更新；本轮最近资源检查GPU约95%、MemAvailable约108GiB、MemFree约28GiB。训练终态核验服务也在等待；尚无终态候选、正式评估或新PPO结果。HTML已在`e11756c`改为读取r2实际服务和日志，下面07:05的“UI尚未更新”是历史状态。目标及物理验收不变：合格代理、兼容新策略、真实CFD配对减阻与升力约束共同成立才完成。

并行工程工作：候选PPO入口、双FNO身份传递、原始评估配置分离及兼容文件相对路径的108项CPU回归已通过；这些是软件测试，不是模型精度通过。独立复核发现正式validation诊断仍写死旧单模型容器路径，canonical入口也需要对原报告做可追溯的临时路径视图；正在以P013专用身份检查修复，不改原报告或数值门槛。新正式评估源码快照将在复核后重新冻结；运行中的训练和终态核验源码不变。下一步仍是训练终态核验→固定六个训练窗口诊断→原完整正式评估→仅在准入通过后新PPO及真实CFD反馈，不以工程测试代替科学结果。

**最新运行恢复（2026-10-05 07:05 UTC）**：原FC-P013训练在592步后无进展；内核记录NVIDIA NV_ERR_NO_MEMORY，进程卡在CUDA设备到主机复制。证据保留在原输出`operational_failure/`，没有终态候选，不是科学失败或成功。仅停止该容器后，同一镜像CUDA计算/复制检查通过。已按原初始模型、数据顺序、1368步、优化器和不可变训练源码启动r2，service `fluid-control-fcp013-training-r2-20261005.service`，invocation `d0138175399a40f487493919a674e1a5`，输出`artifacts/fcp013_independent_force_fno_training_r2_20261005`。恢复批准`a72afd1`/`docs/FC_P013_RECOVERY_APPROVAL_20261005.md`；增加MemFree保护和300秒无进展停止检查，禁止训练时并行大镜像传输，不改变科学验收门槛。启动MemAvailable114.64GiB，CUDAfree37.81GiB，仅代表启动观测。原HTML训练卡仍绑定第一次运行，尚需更新至r2，不能拿旧卡判断新训练。副节点镜像同步已完成并校验同一image SHA，无数据迁移、无PPO执行。下一步确认r2持续更新、更新UI及评估来源绑定，然后按原固定六窗/正式评估/新PPO/真实CFD反馈链推进。最终目标未完成。

**最新执行状态（2026-10-05 06:13 UTC）**：FC-P013正式train-only训练已经启动，Main service `fluid-control-fcp013-training-20261005.service`，MainPID `3503971`，当前invocation `b15752ea472d407ab3ebef57c850cb19`。实测8/1368更新、GPU利用率96%、统一可用内存约107 GiB；这些是该时刻的观测，不是实时常量。源码`1634c05`，执行批准`a6ca463`，批准文件SHA `1bdcfcf7…1a120`。本次固定一个1368-window训练遍历，独立官方FNO仅训练气动力，原P009流场FNO完全冻结，H1/AR各占一半，保持原数据/参数/数值协议；无validation/frozen/PPO。24项CPU训练及dual契约测试通过，工程探针已通过。模型服务容量不足使额外独立代理复核不可用；Root接管最终保存/重载检查，科学验收仍须独立完成。原六窗物理诊断改为只读独立进程，窗口和指标不变，不参与选模型。看板`16d2b27`已部署并经API确认实际服务代次和训练步数。下一步完成双模型正式评估链，训练终态做来源/保存重载/固定六窗核查，然后在不改门槛的前提下评估；最终闭环目标尚未达成。

下文各阶段记录保留作为历史；出现“尚未训练”时以本段最新观测为准。

最后重建：2026-10-04 13:40 UTC（北京时间 21:40）；以实际文件/日志为准。本文是科学状态，不是实时资源看板。

**当前实施状态（2026-10-05）**：FC-P011两臂的1368步固定顺序train-only训练及原formal suite均已完成，18项receipt SHA分别独立复核一致。Arm A head-only receipt/gate SHA为`256a65c7…52a07`/`8112c69b…155d3`，Arm B decoder-tail为`e6c0a171…cebcc`/`3bec0755…85f44`；两者均为development admission FAIL，窗口都只有两个zero分支通过（2/6），frozen未访问且PPO未执行。A四个旋转分支rear-Cl′ RMS误差为`0.070312/0.122247/0.068076/0.083214`；B为`0.062584/0.119208/0.075209/0.076236`，仍显著高于约`0.0294`的固定限值。A的validation10 start0 delta-Cd MAE `0.019206<0.023`通过；B为`0.024318>0.023`，端点也失败。B虽mean-Cl 6/6通过，但Cd仅3/6通过；局部变化不能覆盖联合失败。FC-P011不准入PPO，下一优先仅为固定既有六个train窗口、P009亲本与P011B终态的train-only gradient decomposition诊断设计；不训练新模型、不改门槛，项目未完成。

**FC-P012梯度诊断终态（2026-10-05）**：固定六个train-only H100窗口在P009亲本和P011B终态上各完成一次，共12行；result/completion SHA为`4142cdc5…d814d`/`86f9d931…b26c`。按预声明仅统计五个非zero-action-history窗口：两模型的hidden组`field/(0.2×balanced-force)`范数比大于10均为`0/5`，cosine小于`-0.2`也均为`0/5`，因此不支持强尺度失衡或方向冲突假设，也不支持据此扫描loss权重。zero窗单列：P009比值/余弦为`10.4045/-0.0983`，P011B为`4.5051/-0.1605`，不进入预声明计数。direct-total与组件和的relative residual观测范围为`3.09e-5–9.84e-5`，无预声明容差且不作等价PASS；两模型参数/buffer SHA前后相同，optimizer=0、无save、无validation/frozen/PPO。下一representation-capacity假设须另行审批。

**FC-P013工程状态（2026-10-05）**：独立受力FNO的单个真实train H100资源探针v2已通过，result SHA为`cba0ff7c…bccd`。固定`b04_m075/start180`窗口同时执行冻结flow的100步递推、true-state H1与free-AR两域的受力forward/backward；耗时`4.877 s`，CUDA peak allocated/reserved为`4.085/4.326 GB`，外层guard最低`MemAvailable=107.673 GiB`。官方两项冻结lifting bias名称/shape精确匹配，其余28个可训练parameter tensor梯度均存在且有限；flow/aerodynamic tensor SHA前后同为`89ce3b37…a8bb`。optimizer未创建、step=0、无candidate/save/validation/frozen/PPO。该结果只证明工程与资源可行；正式1368步训练尚未执行，仍待最终trainer保存/fresh-reload、六窗诊断与双模型契约审查，不是科学PASS。

**控制接口兼容状态**：canonical surrogate与direct-CFD启动reward history的不一致已在commit `962c165`修复。canonical路径现在绑定HDF/config/split manifest/HDF及raw-force SHA，以绝对restart时钟原子恢复62个raw-float64 causal samples，并在重复reset时重建fresh copy；legacy Stage-C/direct语义保持。扩展32项CPU回归、Root独立24项和SOTA独立17项均PASS，真实b00构造probe在`141.9–148.0`立即window-ready。该软件修复没有执行PPO，也不是FNO或闭环科学PASS。

**最新FC-P010结论（2026-10-05）**：纯CPU尾窗幅值监督诊断已完成（result SHA `d69033fd…8f94`）。它只优化rear-Cl仿射行；free-AR四个留出相位的尾62步RMS MAE均小幅改善`1.45%–7.34%`，但H1在三个相位恶化`13.26%–13.58%`，仅b02改善`2.08%`。全量fit同样是free-AR改善`9.02%`、H1恶化`10.51%`。五次LBFGS均达到固定200次上限且最终梯度未达声明容差，故这是固定预算下的多域权衡证据，不是已收敛最优解。FC-P010不支持构建候选或启动formal/PPO；下一步转向现有官方FNO表征内的有界训练干预，并继续要求完整field与原formal门槛。

**独立接口工程注记**：三个既有train-only帧的官方Curator持久进程检查已完成（comparison SHA `ab189aac…b87f5`，timings SHA `01e512…529a2`）。状态最大差`2.384e-7`，mask/coords/time一致，持久采样约`1.1–1.5 s/frame`。该检查没有运行solver、FNO或controller，也未计入在线`foamToVTK`导出成本，只说明单帧提取接口具备工程可行性；不能写成完整在线延迟、闭环成功或FNO控制贡献。

**当前优先级（2026-10-05）**：FC-P009的cache-only、固定50/50共享头、受限epoch0候选构建及原formal suite均已完成。候选completion/model/state/result SHA为`26ac241f…b9b5`/`dc41fc91…2e31`/`4998e534…771e`/`131428d4…da60`；formal receipt SHA `ac5c0dd0…e231c`的18项文件SHA已独立重算一致，未访问frozen且未运行PPO。validation10与dynamic6端点组件通过，字段指标与C逐值相同；但6.15 D/U force-window仍只有两个zero分支通过（2/6），四个旋转分支rear-Cl′ RMS误差为`0.06750/0.12501/0.07045/0.07987`，均高于约`0.0294`的固定上限，development admission FAIL。该共同头恢复了P008丢失的Cd/多数mean-Cl准确度并保留明显低于C的RMS误差，却仍没有解决旋转分支的时间窗升力幅值，因此FC-P009被拒绝进入PPO。纯CPU train-cache窗口诊断（SHA `f6c122a6…b626`）显示失败并非只在validation出现：joint free-AR尾62步rear-Cl′ RMS MAE在base20/train8/train16分别为`0.01002/0.02936/0.02035`，最差train PRBS/PPO case为`0.046–0.058`；phase OOF仅将总体`0.01760`小幅增至`0.01985`。随后4个固定训练窗的native-vs-ideal小检查（SHA `2dea49fa…b1ef`）确认runtime/cache features逐值相同、wiring误差为0，native与同保存头pooled-affine的尾窗RMS差最多`0.00194`，而三个旋转窗对真值误差为`0.12561/0.08839/0.06647`。因此当前数值执行差异不足以解释主失败，该诊断分支结束；下一干预应直接针对现有train-only时间窗幅值监督，不做无依据的相位补样或架构扩张。项目最终目标仍是合格代理、兼容控制器和真实CFD闭环共同通过。

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
