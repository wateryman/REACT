# Stage-5 部署设计(中文版)

**状态:** 草稿,stage-3.6 后写的。**部分内容已经被 2026-05-23 新 sprint plan 取代** ——
本文档保留作历史记录,新计划见 [paper_outline_cn.md](paper_outline_cn.md) §A.3。
**目标(原):** 测两个 paper-target 数字(<10ms 延迟、≥85% 闭环 SR),不做任何
进一步训练改动,这样后续要做的监督升级是 data-driven 的。

staging 顺序背后的数学在 [REACT_MATH_Derivations/04_stage5_deployment_math_cn.tex](../REACT_MATH_Derivations/04_stage5_deployment_math_cn.tex) §4。

英文版:[stage_5_design.md](stage_5_design.md)

---

## 1. Scope 分解

Stage-5 拆成两个近独立的 sub-stage。任何一个可以先做,但 5.A 给出延迟答案更快
(不需要闭环基础设施),所以从那里开始。

### 5.A —— 推理延迟 profile(~3 天)

1. **5.A.0 —— Dev-box baseline。** 用当前 stage-3.1-ish 权重在合成 batch 上跑
   `policy.inference()`(用 v0.3.4-temporal HEAD model 在 `use_temporal=False`
   下,等价于 stage-3.1+3.2 默认)。测三种模式的 p50/p95/p99 延迟:
   - 单帧(`depth.shape = (1, 1, 96, 160)`)
   - K-frame stateless(`depth.shape = (1, 10, 1, 96, 160)`)
   - "stateful K=1" —— 同单帧,但 temporal aggregator 在图上(模拟推理时
     stateful 模式)

   在 RTX 3070 8GB(开发机)上做,CUDA 11.8。这是个**关联**数字,不是 Jetson
   数字 —— 但给 per-step 计算的上界。

2. **5.A.1 —— TorchScript / `torch.compile`** trace 网络。对比延迟。文档化任何
   trace 失败的 node(最常见:`if self.use_temporal:` 里动态 shape 分支 ——
   可能需要编译两个分开的图)。

3. **5.A.2 —— ONNX 导出**静态 shape 网络(每个模式一个)。验证输出跟 eager
   forward 误差 < 1e-3。存在 `deploy/onnx/`。

4. **5.A.3 —— TensorRT engine build**(先 FP16,FP16 不够再 INT8 + calibration)。
   先在 RTX 3070 上 build —— production Jetson engine 必须在 Jetson 上 build
   因为是设备特定的,但 FP16 RTX 数字是最近的信号。

5. **5.A.4 —— 延迟报告。** 把 eager vs TorchScript vs ONNX vs TRT-FP16
   (vs TRT-INT8)在三种输入模式下的表列出来。判定单帧 stage-3.1 在 Jetson
   <10ms 下是否够余量(大概率够),以及 K=10 stateless 是否够(marginal,
   按 stage_5_deployment_math §2.2)。

### 5.B —— 闭环成功率测量(~4 天)

1. **5.B.0 —— 场景集。** 生成 100 个匹配 v3 bake 分布的动态场景(同 `dyn_obs`
   参数:3-8 球,1-5 m/s,0.3-0.6m 半径,30×30m 场地)。复用
   `dataset_dynamic/v3/env_*/` 的静态森林点云。存为
   `tools/eval_scenarios/v3_dyn_100/*.yaml`。

2. **5.B.1 —— 闭环 driver。** spawn drone,30Hz 跑 controller + planner against
   sim,log:
   - per-frame state(`pos`、`vel`、planner endstate、score)
   - per-frame 最近障碍距离
   - 终止条件:`goal_reached` / `collision` / `timeout`(默认 10s)

   现有 `Simulator/` + `Controller/` ROS 包已经提供大部分;我们加一个薄 Python
   driver 循环场景 + 聚合结果。

3. **5.B.2 —— 指标计算。**
   - 成功率 SR = `goal_reached` / 总场景
   - 成功 case 的平均到 goal 时间
   - 障碍最小距离分布
   - 失败场景的碰撞细节报告(哪个球、哪一帧、预测是否已经知道错)

4. **5.B.3 —— 跨配置 A/B。** 同样 100 个场景跑两组配置:
   - **C1**:stage-3.1 单帧模型(`use_temporal=False`,`use_dca=False`)。
   - **C2**:stage-3.4 时序模型(`use_temporal=True`)。

   `04_stage5_deployment_math_cn.tex` §3 的 score-head selection factor `k`
   正是这里观察到的 SR/dyn_dyn 比率。结果写到 `results/stage5_closedloop_*.csv`。

---

## 2. Stage-5 结束时的决策规则

| 结果 | 行动 |
|---|---|
| C1 SR ≥ 85% | **Ship stage-3.1。** 围绕它写 paper。stage-3.2/3.4 留作 ablation 行。跳过监督升级。 |
| 70% ≤ C1 SR < 85% | 对比 C2 和 C1。如果 C2 明显更好,ship C2(stage-3.4)。否则上 Option B(多 waypoint head —— 见 `02_multi_waypoint_extension_cn.tex`)。 |
| C1 SR < 70% | Triage:是 planner(失败帧 dyn_dyn 高?)还是 controller(安全预测下跟踪误差高?)还是数据(大部分失败在 <22% FOV-presence 帧?)。Stage-3.7 计划基于此。 |

---

## 3. 风险 + 缓解

| 风险 | 缓解 |
|---|---|
| RTX 3070 延迟跟 Jetson 差很远(10× 快) | 两个数字都报;从公开 Jetson 同类网络 benchmark 给相对 scaling。 |
| 闭环 sim 有 bug 掩盖真实模型表现 | 5.B.0 的场景包含参考路径(直线 goal,stage-3.1 应该轻松通过);这些上失败说明 driver 有 bug,不是 planner。 |
| 开发机 GPU 内存压力(8GB)under TRT | 推理 benchmark batch=1(总是);训练式 batch 在 5.A 里不需要。 |
| `torch2trt` API 跟当前 torch 2.4 差异 | 现有 test_yopo_ros.py 用过;预期需要小修改。Fallback:直接 ONNX → TRT via `trtexec`。 |

---

## 4. 本计划**不做**的事情

- 真机硬件上的真飞行测试。**没有真无人机,out of scope**。
- 重训。Stage-5 整个 point 就是测我们已经有的。
- Camera-aware bake(数据 quality lever,在 `01_collision_loss_saturation_cn.tex`
  §4 最后一个 bullet)。如果 5.B 说需要,stage-3.7+ 再做。
- 多 waypoint head(Option B)。同上 —— 如果需要,stage-3.7 做。

> **更新 2026-05-23:** 本文档 §4 写的"真机 out of scope"已经被推翻 ——
> 用户确认硬件就位,真飞行验证是 sprint Week 3 的必做。
> 见 [paper_outline_cn.md](paper_outline_cn.md) §A.3。
> 同时 §2 的 85% 决策规则也已 reframe 为"beat baseline +15pp"。
> 见 [reflective-enchanting-steele.md](/home/wxs/.claude/plans/reflective-enchanting-steele.md) §2。

---

## 5. Phase + 交付

| Phase | 输出 |
|---|---|
| 5.A.0 —— eager baseline | `scripts/stage5_latency_baseline.py` + `results/stage5_latency_baseline.csv` |
| 5.A.1 —— TorchScript trace | vs eager 的 delta log |
| 5.A.2 —— ONNX 导出 | `deploy/onnx/yopo_stage_3_1.onnx`、`deploy/onnx/yopo_stage_3_4.onnx` |
| 5.A.3 —— TRT engine | `deploy/trt/*.engine` + build log |
| 5.A.4 —— 延迟报告 | `docs/stage_5_latency.md`(或 ARCHITECTURE.md 一节) |
| 5.B.0 —— 场景集 | `tools/eval_scenarios/v3_dyn_100/*.yaml`(100 个文件) |
| 5.B.1 —— 闭环 driver | `scripts/stage5_closedloop_eval.py` |
| 5.B.2 —— 指标实现 | 在 `stage5_closedloop_eval.py` 里 |
| 5.B.3 —— A/B run | `results/stage5_closedloop_C1.csv`、`results/stage5_closedloop_C2.csv`、`docs/stage_5_SR.md` |

总估算:**~7 个工作日**(3 天 5.A + 4 天 5.B)。Phase 5.A 可以独立 land;5.B 不依赖 5.A 结果。

---

## 后续(本文档外)

stage-5 完成后,2026-05-23 新 sprint plan 接管:fine-tune from baseline、
camera-aware spawn v4、real-flight Week 3。所有这些都跟本文档 §4 "不做的事情"
**冲突** —— 时序上 stage-5 设计在前,sprint 改向在后。**当前实际工作以
sprint plan 为准。** 本文档保留作 stage-5 内部组织参考 + 历史 audit。
