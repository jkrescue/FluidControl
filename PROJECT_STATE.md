# PROJECT_STATE — 串联双圆柱主动流动控制

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
