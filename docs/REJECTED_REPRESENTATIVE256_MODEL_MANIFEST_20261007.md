# Representative256未采用候选模型清单

**未采用；不是默认B，不是“最新最佳模型”，不得替换默认policy或用于新的PPO/CFD。** Lead依据同精度fixed-six原AND规则拒绝该候选：H1综合误差比B增加15.81690317%，连续AR100增加14.27029177%。固定开发集未执行、未知，不能写成开发FAIL。训练内拟合改善不等于泛化改善。

本清单仅在训练与评估终态后只读核验文件大小及流式SHA256；未加载模型、未forward、未复制大payload。所有路径相对于Spark根目录 `/workspace/fluid_control`。默认正式交付仍见[正式B清单](FINAL_MODEL_AND_REPRODUCTION_MANIFEST_20261007.md)。

## 来源与状态

- 父B manifest SHA `92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891`；父气动力模型SHA `57d4634df22ce96c1c4467a2ed52412be452375129af05b89f10a690e363356e`。
- [唯一训练批准](P064_REPRESENTATIVE256_TRAINING_APPROVAL_20261007.json)，SHA `42e8c15d4835ea6696ecb12271246d9babba16b73f869b4802b57bbc8dabc38c`；训练invocation `6bc6aca4e1bd48d5938b273d319fc716`。
- [训练独审](P064_REPRESENTATIVE256_TERMINAL_REVIEW_20261007.md)：工程完成，300closures，状态 `BUDGET_STOP_NOT_FITTED`；非达到预设训练RMSE目标。
- result记录官方fresh reload通过，此清单不重复模型加载。气动力tensor digest `174d79e1387446bee15174ac9c2b9c4f97370bc96d6ebb3d551a2de9dfc7360a`，flow digest `ed47415392b2b466433b92dd002cb09ac252a20f8fdd16197b9fc4a8bf2d9b01`。
- [同精度六窗批准](P064_FIT256_FIXED_SIX_APPROVAL_20261007.json)、[工程报告](P064_REPRESENTATIVE256_FIXED_SIX_ENGINEERING_REVIEW_20261007.md)、[独立数组报告](P064_REPRESENTATIVE256_FIXED_SIX_INDEPENDENT_REVIEW_20261007.md)；六窗result `artifacts/p064_fit256_fixed_six_20261007/result.json` SHA `57add3a45f4cedcde94fbc242337e64cd6d54d51156e4d2be27c29cb8c309289`。比较双方均highest/noTF32，不混用历史B high/TF32数值。

## 完整候选文件（实际大小与SHA）

共同前缀 **C** = `artifacts/p064_representative256_training_20261007/candidate/`；表中C仅为缩写，不是另一个目录或环境变量。

| 路径 | 字节 | SHA256 |
|---|---:|---|
| C `dual_model_manifest.json` | 8699 | `793bbdab1da9fb26ebfa27a2efe607b1ccb24c10a4bd73dcc726d4253b696848` |
| C `training_protocol.json` | 391 | `af6dcaf2d4307360a529913a0124a05b381e315aa84ffd370fee2e34bf568325` |
| C `flow/FNO.0.0.mdlus` | 188903667 | `dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31` |
| C `flow/checkpoint.0.0.pt` | 1773 | `4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e` |
| C `aerodynamic/FNO.0.1.mdlus` | 188903667 | `f0c8ada71225b2d2015aad7805c91455eb9e9311a45abe41439cc3c2509162ac` |
| C `aerodynamic/checkpoint.0.1.pt` | 2266698142 | `b040cfd15ff0710abcfda5a1cab468874f889a9b814b1a3cc3c85b1dd5ca2cc8` |
| `artifacts/p064_representative256_training_20261007/result.json` | 6362894 | `50617c1c49cebcdc2198fb1cc427f7a7289dc1912846d3f882b0a257025b6d6a` |
| `artifacts/p064_representative256_training_20261007/supervisor_receipt.json` | 119 | `b50dce3cd0310a64950040beee71652a17b8ea4f48c2c0ca88038770006bf7f5` |
| `artifacts/b00_controlled_train_dataset_view_20261006/normalization.json` | 1248 | `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1` |
| `artifacts/fcp027_diagnostic_source_20261006_immutable/training_config.yaml` | 1764 | `07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9` |

`.mdlus`和`.pt`共同保留官方模型及训练状态；此处气动力`.pt`保存LBFGS状态，不能按旧AdamW状态大小或28个Adam条目解释。normalization/config复用原固定文件，没有为新候选重拟合归一化。

## 消费者和源身份

候选kind为 `P064_REPRESENTATIVE256_FORCE_FNO`；manifest中 `...MANIFEST_VERIFIED` 仅表示工程契约，不代表预测准入或采用。精度合同为highest/noTF32。官方FNO结构仍6输入/7输出；数据为原192+64权重的256真实H1点，255唯一点、1个重复保留。

冻结源目录 `artifacts/p064_representative256_source_20261007_immutable/`：

- `train_representative256.py`：`00e9921f1614a0a791d4883f08a3bc6e475a6160a5051e3232b588d1a7fa9e35`
- `panel256.py`：`ef230ec45fbd6f5f254d258b308b41674c761cdea6d34dba4d46b8c883549724`
- `dual_fno_representative256.py`：`484eb9e26889b4e9efac807447d131113d2b1dd8bd0cb6b77472250f8ba8fbc8`
- `PLAN.md`：`cf36533eaec670d1ba8cd68e04f2dbc42514e3064ea77bb8ee7b719605506fe1`
- 共用core `artifacts/p064_fixed_small_fit_r2_source_20261007_immutable/bounded_lbfgs.py`：`e0623157281a041b05257d9843ca46e8e1557ffca83c5872d9280fbacdd5a6c0`。

这些文件及批准绑定实际官方save/reload路径。不得将候选manifest冒充旧B kind、把候选aero混接B policy声称完成新闭环，或把未采用候选重命名为latest。保留全部失败/拒绝证据；无新部署命令，若未来研究需另行批准。
