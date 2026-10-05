# 在线闭环：现有实现、待完成项和结论范围

2026-10-05。基于现有源码检查；不替代实验结果，不改变验收指标。

## 当前四层职责

| 层 | 当前实现 | 尚不能声称的结果 |
|---|---|---|
| CFD与数据 | OpenFOAM真实流场及壁面forceCoeffs；Curator采样；项目DataPipe基于官方DatasetBase/HDF5Reader | 数据处理完成不等于代理精度合格 |
| 代理预测 | 官方FNO双模型分别给流场递推和气动力；P015正在训练 | 没有合格P015终态，也没有已验证的P015受力改进 |
| 策略训练 | HydroGym环境接口、项目流动适配器和SB3 PPO | 没有基于合格P015的新策略；旧CFD-only结果不能改名为代理辅助成果 |
| CFD在线反馈 | 每段读取真实探针/四力，PPO产生动作，约束动作后推进真实OpenFOAM，并更新下一次观测 | 当前真实反馈函数没有调用在线FNO或Curator；不能声称其已运行在线预测控制 |

## 源码核对

- `scripts/run_full40_canonical_ppo_openfoam_feedback.py::run_feedback`：真实
  `total_drag_observation_at` → `PolicyInferenceSession.request` →
  `apply_action_rate_limit` → `OpenFOAMPairSession` → 新真实观测。同初态零动作
  CFD同时推进，保留初态/观测/边界条件来源和逐段计时。
- `src/fluid_control/tandem_hydrogym.py::TandemFNOStepper`：代理环境内的FNO
  递推。这里调用模型不证明真实CFD反馈函数调用了模型。
- `src/fluid_control/tandem_datapipe.py::TandemWindowDataset._load`：读取
  current和following帧，后者提供训练目标。因此不能原样当作只读当前时刻的
  在线输入接口。官方DatasetBase/HDF5Reader与项目子类的职责必须分开说明。
- `scripts/sample_tandem_vtk_frame.py`：使用官方Curator的VTKSource和Mesh采样，
  再由项目代码处理固定ROI、有效mask、压力去均值及输出。这不是控制器。

## 后续按依赖推进

1. 完成P015预定训练及同协议正式评估；失败时不得用代理reward代替CFD验收。
2. 补齐P015模型/完整评估证据在控制入口的显式身份支持；旧P013路径保留，
   不接受交叉身份或缺失实际终态/重载证据。
3. 仅在完整准入后训练兼容的新PPO，保持现有69维观测、奖励和动作约束。
4. 冻结策略，完成真实CFD在线反馈与原配对减阻/升力验收。这证明的是
   “代理辅助训练的策略在真实CFD中闭环控制”，不是“FNO在每次反馈时规划动作”。
5. 在线流场预测展示若接入，应从当前真实状态和本次拟施加动作开始，预测
   下一时刻并在CFD到达该时刻后比较。只展示不参与动作决策的预测必须标记为
   影子预测；若要让预测影响动作，需要另行明确MPC或策略观测协议并独立验证，
   不能偷偷把未来预测替换现有当前状态观测。

## 时间与资源

控制间隔0.1D/U是无量纲物理时间，不是0.1秒墙钟deadline。已有Curator帧
采样计时不包含完整solver/export/预测/策略链。完整闭环应分别记录CFD推进、
状态提取、预测和策略耗时；未测量前不使用“实时硬件控制”表述。

当前训练保留20GiB以上统一内存余量。PPO启动还有独立CUDA与统一内存预检，
不能因为主训练资源合格就跳过PPO预检，也不为通过预检降低20GiB保护要求。
