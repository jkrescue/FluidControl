# FC-E057：PPO 策略反射缺陷只读审计终态复核

状态：`EXPLORATORY_POLICY_REFLECTION_DEFECT_AUDIT_COMPLETE_NOT_ADMISSION`。这是固定已完成轨迹上的 CPU 只读诊断，不是反事实闭环、真实 CFD 干预或科学准入。

## 执行与身份

- unit `fluid-control-policy-reflection-defect-audit-20261006.service`，invocation `dff10f4049dc4b4da84d7d05818d1c9b`，PID0/exit0。
- 审批 `docs/EXPLORATORY_POLICY_REFLECTION_DEFECT_AUDIT_APPROVAL_20261006.json`，SHA `35c080b2979a5d1e151aa7b2943aa6c673ec503315e1e26242259afa0a88dfa7`。
- 不可变源 `artifacts/exploratory_policy_reflection_defect_source_20261006_immutable/audit_policy_reflection_defect.py`，SHA `bc4906e8c4c483ef16458819bd80fd210ff4046137fcae638ccd4f3d007aa384`。
- 固定680周期快照 `artifacts/exploratory_32768_reflection_prefix_20261006/prefix.json`，SHA `e2e4b0d6363d3ec36f530ad673d92f2d72638be993cf79fc30ce979958c405c5`，覆盖CFD时间148至216。
- 结果 `artifacts/exploratory_policy_reflection_defect_audit_20261006/result.json`，SHA `b0c48354f85a2f3e0b6ccf9f41079e2eded7e33fe4a422f7f3a0c61310fc6809`。

耗时2.424918秒，最低 `MemAvailable=123608264704` 字节。SB3加载重建优化器对象，但 `optimizer_steps=0`，无训练、模型写入或CFD执行。

## 固定变换与结果

69维物理观察按中线 `y=7.5` 反射：32个 `x=17,y=6..9` 探针逆序，`u/Cd` 偶，`v/Cl/omega` 奇。审计计算 `pi(o)`、`pi(Ro)`、同号缺陷 `pi(o)+pi(Ro)` 和投影请求 `0.5*(pi(o)-pi(Ro))`。

| 冻结策略 | 缺陷均值 | 缺陷RMS | 投影请求均值/RMS | 投影变化率受限/饱和 |
|---|---:|---:|---:|---:|
| 32768步 | -0.6702486725962338 | 0.8046740447610694 | -0.04383776928846012 / 0.47083816235608983 | 593 / 0 |
| 4096步 | -0.3195020545493154 | 0.3197158563363626 | -0.00454239776248441 / 0.03520572549740503 | 633 / 0 |

过滤统计以原始已完成轨迹记录的前一时刻omega为条件，不是投影控制的反事实轨迹。较大镜像缺陷是可复现诊断信号，但不能证明其导致FC-E055偏置，也不保证投影后有限时长升力均值为零或减阻保持。

FC-E058只检验一个单因素包装：相同32768策略、restart148、800周期、配对zero、六个窗口、资源与动作约束，请求替换为投影后再调用一次原过滤器。本审计不改变FC-E055、K1正式FAIL或原物理门槛。
