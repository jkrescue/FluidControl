# 实验索引

实验编号仅用于追溯记录，不是模型名称。主要实验如下。

| 实验 | 内容 | 结论 |
|---|---|---|
| E082，控制策略训练 | 冻结 B 代理，PPO 交互 32,768 次 | 得到默认 canonical PPO |
| E095，基本闭环复现 | 800 次真实 CFD 反馈，主窗 t*=150–210 | 减阻 4.0091% |
| E109，后续时段验证 | 800 次反馈，主窗 t*=268–328 | 减阻 3.9949% |
| E114，延长时段验证 | 800 次反馈，主窗 t*=328–408 | 减阻 4.1326%，波动降低 17.9229% |
| B-H5 MPC，短时控制测试 | FNO 五步预测，十次 CFD 反馈 | 未取得有效减阻收益 |
| Representative256，代理训练候选 | 固定六窗口同精度比较 | 单步/连续误差增加，未采用 |

[结果与指标定义](research_records/RESULTS.md) · [完整结构化账本](experiments/results.csv) · [历史实验说明](https://github.com/jkrescue/FluidControl/blob/research-history-20261009/EXPERIMENTS.md)
