# PROJECT_STATE — 串联双圆柱主动流动控制

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
