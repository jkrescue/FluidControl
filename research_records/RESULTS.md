# 评估结果

实验数据截至 2026-10-07。场景为 Re=100、L/D=5 的固定中心双圆柱，仅后柱旋转。受控与无旋转 CFD 使用同一评估窗口。编号是内部实验索引：E095 为基本复现，E109 为后续时段验证，E114 为延长时段验证。

## PPO 真实 CFD 闭环

| 实验 | 主评估窗口 t*=tU∞/D | 总减阻 | 后柱升力波动 RMS 比 | 平均升力偏置 |
|---|---|---:|---:|---:|
| 基本闭环复现（E095） | (150,210] | 4.0091% | 0.817190 | 3.6366% |
| 后续时段验证（E109） | (268,328] | 3.9949% | 0.816588 | 2.8533% |
| 延长时段验证（E114） | (328,408] | 4.1326% | 0.820771 | 1.2352% |

三项主窗口均通过原标准：减阻 ≥2%、RMS 比 ≤1.05、偏置 ≤10%。E114 波动降低 17.9229%，四个连续子窗口结果如下。

| E114 子窗口 | 减阻 | RMS 比 | 偏置 |
|---|---:|---:|---:|
| (328,348] | 4.2414% | 0.822530 | 4.8866% |
| (348,368] | 4.3018% | 0.827556 | 6.8816% |
| (368,388] | 4.0692% | 0.816828 | 0.9175% |
| (388,408] | 3.9176% | 0.817398 | 5.8391% |

![E114 受力与动作](assets/e114_closed_loop.png)

固定 CPU PPO 根据真实 CFD 观测输出动作，验证期间不运行 FNO 或 MPC。E109/E114 是同一配对轨迹的恢复前后时段，不是独立统计重复。[延长时段原始审查](https://github.com/jkrescue/FluidControl/blob/research-history-20261009/research_records/P064_B_CONTINUATION_328_408_TERMINAL_REVIEW_20261007.md)

## 代理预测

B 的 validation10 面板在 100 步末端：速度相对 L2 误差 0.0435911，后柱 Cl MAE 0.0431183，总 Cd MAE 0.0112626。受力统计的 6 个分支仅 2/6 同时达标，四个旋转分支均未通过升力波动预测标准。完整代理精度未达标。[原始评估](https://github.com/jkrescue/FluidControl/blob/research-history-20261009/research_records/P064_B_FORMAL_TERMINAL_REVIEW_20261006.md)

![B 五步预测与 CFD 对照](assets/retained_b_fields_h5.png)

图为一个保存样例，不代表总体精度。Representative256 在同精度固定六窗口的加权归一化误差为：

| 预测方式 | B | 候选 | 误差变化 |
|---|---:|---:|---:|
| 单步 H1 | 0.00418103 | 0.00484234 | +15.8169% |
| 连续递推 100 步 | 0.00900292 | 0.01028767 | +14.2703% |

候选未采用，未继续其 PPO/CFD 测试。训练损失下降不等于预测改善。[原始复算](https://github.com/jkrescue/FluidControl/blob/research-history-20261009/research_records/P064_REPRESENTATIVE256_FIXED_SIX_INDEPENDENT_REVIEW_20261007.md)

## MPC 短时测试

B-H5 MPC 完成 10 次反馈，覆盖 1 D/U∞：配对减阻 −0.007886%，波动 RMS 比 0.983611。流程可执行，未取得有效减阻收益；短窗口不能证明长期控制效果。[原始记录](https://github.com/jkrescue/FluidControl/blob/research-history-20261009/research_records/report_20261007/evidence/b_h5_mpc_result.json)

## 指标与结论

总阻力系数 CD=Cd前+Cd后；减阻率=100×(1−mean(CD受控)/mean(CD对照))%。升力波动为后柱 Cl 去均值后的 RMS，波动比为受控/对照。偏置=100×abs(mean(Cl受控))/RMS(Cl对照去均值)%，不是相对平均升力的变化。

当前结果支持限定工况的基本 PPO–CFD 闭环。尚未证明跨 Re 泛化、净节能、硬实时或实验迁移。早期窗口与部分 seed 的失败仍在[历史分支](https://github.com/jkrescue/FluidControl/tree/research-history-20261009/research_records)保留；完整[实验账本](../experiments/results.csv)没有改写。
