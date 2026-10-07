# 串联双圆柱控制：技术架构与分阶段实施路线

日期：2026-10-07。本文为现有实现的技术说明及下一阶段设计建议，不是新实验批准。固定二维Re=100、L/D=5，仅后圆柱旋转；不扩大几何或Re，不降低验收标准。

## 1. 总体目标与当前交付边界

目标由三部分组成：可接受的受控流动代理、与代理兼容的控制器、真实CFD中的约束闭环效果。三者分别验证，不能用训练loss或代理reward替代真实控制效果。

- **已验证路线：FNO环境训练PPO→冻结策略→CPU PPO控制真实OpenFOAM。** 默认B保留；E114新增80 D/U达原三门，累计160 D/U经过已验证restart，不是单进程不间断或独立工况。
- **未证明收益路线：真实当前流场→冻结B FNO预测→H5 MPC→真实CFD。** 十周期只通过工程审计，动作与旧K1一致、预测误差更差；没有有效MPC控制收益结论。
- 完整FNO预测精度尚未准入。Representative256训练面板改善但原fixed-six恶化，候选拒绝；不能把最新checkpoint当默认模型。
- 基本PPO闭环可以独立交付；这不等于完整研究目标完成，也不要求伪称在线部署使用FNO。

## 2. 技术蓝图与责任边界

```text
OpenFOAM真实轨迹 → VTK → 官方Curator/Mesh采样 → HDF5/训练集normalization
                                           ↓ 官方Reader＋项目DataPipe
                                 官方PhysicsNeMo双FNO训练/保存
                                      ↓ 冻结代理环境
                             HydroGym接口＋项目reward/约束
                                      ↓ 开源SB3 PPO
                       冻结policy＋Vec＋canonical适配＋单filter
                                      ↓
真实69观测 → CPU策略动作 → OpenFOAM推进0.1 D/U → 下一真实69观测

探索分支：真实当前网格场 → 双FNO候选H5 rollout → MPC选首动作
          （不替换上述已验证PPO；未实现自动混合fallback）
```

官方库负责模型、数据读取/采样及训练基础设施；HydroGym提供环境接口，SB3实现PPO。坐标映射、数据组织、force pooling、损失、奖励、OpenFOAM传输、MPC选择器、执行审批与审计均是项目代码，不称官方内置控制方案。

当前数据链路入口：`scripts/curate_tandem_cfd.py`、`src/fluid_control/tandem_datapipe.py`。项目DataPipe围绕官方DatasetBase/HDF5Reader，不能把项目adapter本身称官方组件。模型身份、norm、split、配置和源码均以每次approval绑定为准。

## 3. 物理量与在线接口

无量纲时间`t*=tU∞/D`；D=U∞=1时数值等于case时间，但D/U不能自动解释为现实秒。CFD dt=.005，控制间隔=.1 D/U，每反馈20个solver step。

动作`ω*=ΩD/U∞`；表面速度比为`α=ΩD/(2U∞)=ω*/2`。原限制`|ω*|≤.75`、每反馈`|Δω*|≤.1`，后者等价最大速率1/(D/U)。控制表在求解区间内线性插值，不是把策略请求无约束直接交给旋转边界。

物理69观测严格按序：

| 索引 | 内容 | 实际来源 |
|---|---|---|
| 0–63 | 32探针，每探针u、v交错 | 训练为预测网格场双线性插值；部署为真实CFD probes |
| 64–67 | front Cd、front Cl、rear Cd、rear Cl | 训练为代理力；部署为真实force coefficients |
| 68 | 当前已施加ω | 控制状态；物理filter使用保留的double上一动作 |

训练/部署位置与顺序一致不意味着两种采样数值逐位相同。字段接口见`src/fluid_control/tandem_hydrogym.py:get_observations`及`scripts/run_p064_symmetry_canonical_b_ppo_long_cfd.py`。

canonical反射见`src/fluid_control/symmetry_canonical_wrapper.py`：反排y探针、v/Cl/ω变号、u/Cd不变；策略只调用一次，按当前orientation恢复物理动作，再执行一次幅值/速率filter。不能重复滤波，也不能用下一时刻orientation反解当前动作。

## 4. 代理模型训练与预测时序

双FNO为两个官方FNO实例：flow分支提供三个归一化场增量，aerodynamic分支提供四个力通道的网格读出。官方raw输出为7通道，并非新造4输出网络；气动力读出取后四通道，经流体mask空间平均与原force mean/std还原物理系数。

当前K1历史长度下，输入包含当前归一化`(u,v,p)`、mask、当前与下一施加ω/action_scale，共6通道。flow更新为`q_next=(q+delta_q)*mask`；不是将前三输出直接当下一场。气动力对应从当前状态和该区间动作预测下一端点力。

原B训练入口`scripts/train_fcp064_controlled_aero_ab.py`：fresh K1父、原192＋受控b00 64个窗口、256窗/32更新、8窗梯度累积；flow和两个lifting升维层bias冻结，只训练28个气动力参数张量。不是“冻结两个升力网络bias”。

原目标为`.5 L_H1 + .5 L_AR100`：H1使用真实当前场；AR使用冻结flow产生的连续预测场。force误差按训练集std归一并采用既定四通道权重；分块反传、累积/clip/AdamW及固定顺序都属于实验协议，不能悄然修改后仍称同一B。

训练内拟合、独立开发误差、动作响应、场误差和控制收益必须分开。固定小面板充分拟合只检验该优化方案对该面板的拟合能力；改善不证明泛化，预算未达目标不证明网络容量不足。

## 5. PPO：代理环境训练与真实部署不是同一过程

实际E082协议见`scripts/train_p064_symmetry_canonical_b_32768_ppo.py:PROTOCOL`：24个固定reset，4环境、32768交互、seed20261007、每episode5步；n_steps128、batch256、4epochs、LR3e−4、gamma.99、GAE.95、clip.2、vf_coef.5、max_grad_norm.5。保存最终策略，不据开发效果选中间checkpoint。

代理reset从绑定真实初态载入场、力、当前动作与真实62点因果力历史；不能零填历史或偷看未来力。`Full40CanonicalSurrogateFlow.reset`检查历史末端与当前force/时间一致。每step依次进行物理动作约束、双FNO推理、场/力/history更新，再计算reward和下一69观测；非有限或场界异常显式终止。见`src/fluid_control/full40_canonical_hydrogym.py`。

PPO的5步episode只有.5 D/U；reset带入真实历史不代表学习过程已经经历完整长周期控制。后续真实800/1600反馈用于独立检验持续控制，不能由训练episode直接推出。

真实部署冻结policy与Vec状态，CPU执行canonical策略，不更新权重、不运行在线FNO。实际序列是`真实当前69→canonical→policy.predict→符号恢复→单filter→两支CFD推进→真实下一69`；配对zero是独立求解分支，不参与策略选动作。保留完整raw日志与反馈闭合证据。

## 6. 原物理目标与实际reward

令`D=1−mean(Cd_front+Cd_rear)/mean(Cd_total_zero)`，`R=RMS(Cl_rear−meanCl_rear)/RMS(Cl_zero−meanCl_zero)`，`Q=|meanCl_rear|/RMS(Cl_zero−meanCl_zero)`。原三门为`D≥.02、R≤1.05、Q≤.10`，联合AND。Q不是减去zero平均升力后的偏置。

现有canonical六项cost（不是本文新设计）为：

```text
c = −clip(D,−1,1)
    + max(0,(.02−D)/.02)^2
    + max(0,(R−1.05)/.05)^2
    + max(0,(Q−.10)/.10)^2
    + .01(ω/.75)^2 + .01(Δω/.1)^2
PPO即时reward = −.1 c；PPO回报另按gamma=.99折扣。
```

代码`src/fluid_control/canonical_joint_v1.py`与`Full40CanonicalRewardAudit.step`核reward分量和总值。窗口未ready时四个物理cost置零、动作项仍存在；当前正式reset要求真实历史就绪。动作平方只是代价代理，未核扭矩/功率前不称净节能。

## 7. H5 MPC：状态、历史与局部优化

当前薄入口`scripts/run_p064_b_causal_history_h5_feedback.py`复用原K1 CPU十周期传输，仅绑定当前B双FNO。实际CPU/high+TF32协议保留；GPU highest/noTF32仅另有十包推理排序一致性复放，不等价于GPU真实长闭环已验证。

状态是当前真实CFD网格场、mask、上一实际ω，以及截止当前的62×4真实力历史。实时网格桥接沿原代码，不从未来CFD拿输入；K1网络自身的状态历史长度1与reward的62点力历史是两种不同历史。

`src/fluid_control/exploratory_causal_history_horizon.py`生成五个候选：`clip(ω+[-.1,-.05,0,.05,.1],−.75,.75)`，每个候选保持5步；分别rollout flow和aero，总计25个预测阶段。比较五候选的既有六项cost在H5上的算术均值，**不是PPO折扣回报**；仅执行获选首动作，下一周期重新观测并规划。

候选场非有限或越过绑定normalized-state界不得成为有效选择；相同cost按更小动作变化量、再按候选索引稳定决胜。该实现是有限候选rolling-horizon，不宣称全局最优、连续优化或通用CEM。

H5=.5 D/U预测窗；62点每.1采样，时间戳跨度6.1、按区间覆盖6.2，满足原6.15 D/U统计窗。每个候选从同真实历史的副本开始，每预测一步仅在候选副本中删最旧点、加入该预测力；五步后最多5个预测点，不是预测完整6.2 D/U未来。

持久历史只由实际下一CFD力推进，见`exploratory_causal_history_mpc.append_actual_endpoint`。候选预测不能污染下一真实规划历史。短预测相对于长统计窗的贡献有限，这是需验证的设计限制，不是自动改horizon、reward或阈值的理由。

原B十周期采用fail-stop，无PPO fallback。非有限、无可行候选、资源/CFD故障均保留失败，不自动换策略继续并冒称纯B-MPC。独立可杀推理进程、超时后清理再切冻结PPO、混合控制标记与故障注入测试属于**未来工程建议，尚未实现或批准**。

## 8. 失败定位与证据层级

| 故障/现象 | 首先检查 | 不能推导 |
|---|---|---|
| loader/precision/路径失败 | manifest、实际import、runtime、SHA、unit日志 | 模型科学失败或训练无效 |
| H1真实场输入误差大 | 场/动作/目标时钟、norm、读出、训练拟合与覆盖 | 唯一归因flow误差 |
| AR明显恶化 | 同初态H1/AR分开、冻结flow漂移、输入分布 | 用H1平均改善掩盖AR失败 |
| MPC排序/动作相同但预测变差 | 全候选cost、可行界、tie-break、真实下一力 | 工程可运行即控制收益 |
| 短窗物理好、长窗失败 | 固定过渡窗、均值偏置、持续反馈、baseline | 挑最好时间段宣称通过 |
| GPU/CPU数值改变 | 同真实包全候选排序和动作对比、precision绑定 | 速度更快即精度提高 |

当前B10与旧K1动作相同且B selected-next-force误差更大，不能仅凭GPU加速证据自动扩800CFD；需要单独说明待检验的模型/控制收益假设。

## 9. 下一阶段验收矩阵与顺序（建议，须另批）

| 次序 | 证据/验收 | 不通过时 |
|---|---|---|
| 1 固定交付 | B模型/policy/Vec/norm/config与历史审批可校验；安全入口默认只读 | 修身份/缺件，不覆盖默认B |
| 2 代理候选 | 工程保存与official reload；原six H1/AR双不退化；原16起点×H1–H5开发指标，比较precision一致 | 保留失败，不自动dev/PPO/CFD；执行范围由Lead逐项授权 |
| 3 兼容控制 | 若换surrogate，重新训练兼容PPO；不沿用旧策略冒充新代理策略 | 分开诊断代理与policy，不偷换reward |
| 4 在线工程 | 固定短段真实输入/预测/动作/下一端点配对；因果历史、资源清理、时钟通过 | fail-stop并诊断，不自动延长 |
| 5 真实效果 | 配对zero，预声明过渡/统计窗；原2%/1.05/10%与所有固定分窗，不事后挑窗口 | 点估计失败保留；不降低门限 |
| 6 论文强证据 | 最终策略dt/网格复核、公平恒转速/周期、冻结多seed/独立phase、扭矩代价 | 明确证据不足，不能由单条成功外推 |

训练候选、MPC工程与公平基线不应一次联合改动；先提出唯一可证伪问题、固定数据/预算/停止规则，再执行。达到训练误差停止准则只允许保存指定终态，不自动晋级；超时/OOM/非有限须区分工程失败与科学拒绝，不能拿未完成checkpoint补结果。

已有周期P10/P20失败，但幅值1、历史t80初态与窗口120–160不同于当前B，不能称公平优越；已有恒转速网格比较也不替代最终B策略验证。八个已选phase均已打开，E114是同case延长，不是独立holdout。详见[CLOSED_LOOP_PAPER_VALIDATION_GAPS_20261007.md](CLOSED_LOOP_PAPER_VALIDATION_GAPS_20261007.md)。下轮仍限定Re100/L/D5。

## 10. 复现与交接

安全入口：`.venv-curator-py312/bin/python scripts/reproduce_canonical_closed_loop.py`，默认只读`PREFLIGHT_PASS_NOT_RUNNING`。历史approval是证据，不是复跑许可证；实际执行需新的唯一output/unit/审批、输入SHA及资源检查。

详见[主报告](PROJECT_FINAL_REPORT_20261007.md)、[多阶段runbook](CANONICAL_MULTI_STAGE_RUNBOOK_20261007.md)、[正式模型清单](FINAL_MODEL_AND_REPRODUCTION_MANIFEST_20261007.md)和[拒绝候选清单](REJECTED_REPRESENTATIVE256_MODEL_MANIFEST_20261007.md)。本文引用当前repo接口解释实现；每次运行的不可变源码闭包/参数以对应approval为准，不能用以后修改的同名脚本替代历史运行身份。

限时本轮科学任务截止12:20:45 UTC、归档截止12:50:45 UTC；旧自动启动timer已停用。本文没有授权新的训练、MPC、CFD或超时后续跑。
