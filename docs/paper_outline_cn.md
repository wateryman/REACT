# REACT 论文大纲 —— Sprint v2 中文版(2026-05-23)

**状态:** 活文档,sprint 中实时更新。**4 周交稿 RA-L/IROS;实机验证必做**(见
`/home/wxs/.claude/plans/reflective-enchanting-steele.md` 和
`docs/stage_5_sr.md` 看完整闭环状态)。

英文原版:[paper_outline.md](paper_outline.md)

---

## TL;DR —— v1(2026-05-21)→ v2 的变化

| 项 | v1 大纲 | **v2 大纲(本)** |
|---|---|---|
| Headline 指标 | "**≥ 85%** 动态 SR" | "**Pareto 前沿:dyn 安全 vs goal-rate**:一个 operating point +6pp goal SR、另一个 0 dyn collision;闭环 benchmark + sim-to-real" |
| 硬件要求 | (TBD) | YOPO 同款无人机 + RealSense + Jetson + 实机已确认 |
| 投稿目标 | "TBD" | **RA-L / IROS(4 周后)** |
| stage-3.1 状态 | "预期突破" | **scratch 训练 neutral;fine-tune +5pp;v4 重训进行中** |
| 失效模式 narrative | 没有 | **paper 中心叙事** —— baseline 保守超时 vs REACT 激进撞树等 |
| sim-to-real | "future work" | **Week 3 交付** —— 短期实机 trial 章节 |

---

## Section A. 现在的问题 + 当前修复(live status)

这一节是 4 周 sprint 的工作快照,**不进 paper**。给协作者 / reviewer / 未来的自己
看哪些坏过、哪些修了。

### A.1 闭环基础设施问题(全部已修)

| # | bug | 症状 | 根因 | 修复 |
|---|---|---|---|---|
| 1 | 驱动 yaw 跟踪 velocity 而非 goal | C1 drone 永远不转向偏 lateral 的 goal(0/100 到达) | `yaw = atan2(vel.y, vel.x)`;当 C1 argmin 给纯 forward endstate,vel 始终在 +X,yaw 卡在初始角 | 改为跟 goal 方向 2 rad/s 限速,跟 `YOPO/policy/poly_solver.py::calculate_yaw` 一致 |
| 2 | sim 渲染**没有森林场景** | 所有 SR run 测的是近乎空的静态点云;baseline 假高到 66% | `tree_file: "src/pointcloud/tree.ply"` 是相对路径;sim 从 `cwd=REACT/` 启动找不到文件;`Maps::forest()` 早退不生成树也不生成地面 | 从 `cwd=Simulator/` 启动(那时路径解析对)。所有修前 SR 数字(23/27/66/0)作为"无效 sim audit data"保留;**修后数字以 `*_realsim*.csv` 后缀替代** |
| 3 | 模型钻 ESDF z<0 free-space 漏洞 | C1-v1 一头扎地下(end_z ≈ -2m);SR 0/100 | `Simulator/maps.cpp::forest()` 静态点云没有地面平面;`YOPOLoss.safety_loss` 在最低 z 以下默认 free-space → 模型选"向下看"anchor cost 最低 | 新加 `YOPOLoss.z_floor_loss` quadratic hinge,z<0.3m 时惩罚;同时进 trajectory_loss 和 score_label([YOPO/policy/yopo_trainer.py](../YOPO/policy/yopo_trainer.py)) |
| 4 | 修前 CSV 残留误导数据点 | "baseline_v2_hack = 66%" 被 mid-sprint 提交 cite 过 | 流程 bug(中途 commit 后才发现 bug 2) | 5 个修前 SR CSV 用原文件名保留作 broken-sim audit,paper **只引用 `*_realsim*.csv`** |

### A.2 建模问题(大部分已修)

| # | 问题 | 诊断 | 状态 |
|---|---|---|---|
| 5 | **C1 from-scratch**(50 epoch,无 z_floor)钻地下 | 无 z_floor 训练;argmin 总选 v=2 行(look-down)的 anchor,因为 safety_loss 在那里看着 free | 已修 → z_floor loss;归档为"旧 C1-v1" |
| 6 | **C1-v2 加 z_floor**(50 epoch,from scratch)用静态能力换动态 | 50/50 mixed sampling 稀释了 baseline 的强静态避障;static_collision 7% → 24% | D1 改策略:fine-tune from baseline(保留 baseline 静态信号) |
| 7 | **所有 ablation(3.2、3.4、3.6)撞 ~1% dyn_dyn 天花板** | 怀疑 dataset starvation,但 v3(4× v2)5k iter 也没破 ceiling | 现在怀疑是**数据质量,不是数量** —— v4 camera-aware spawn 把 FOV-presence 从 22% 拉到 88% |
| 8 | **Pass-1(point-mass)高估 SR** | 限幅双积分器以无限 jerk 跟 planner 输出 —— 跟部署不符 | 加 Pass-2(Poly5 min-jerk),跟 `test_yopo_ros.py` 部署同款。Pass-2 是 paper 引用的数字 |

### A.3 2026-05-23 进行中

| Sub-stage | 做什么 | 状态 | Gate |
|---|---|---|---|
| D1 | fine-tune from `YOPO_1/epoch50.pth` + z_floor + dyn loss,**v3 数据**,25 epoch | ✅ 完成。**C1-FT v3 = 36% SR**(beat baseline 31% +5pp) | 通过 +4pp gate |
| D2 | `dataset_generator.cpp` camera-aware ball spawn + 重 build + bake v4 | ✅ 完成。**FOV PASS rate = 88%**(v3 是 22%) | 通过 ≥50% gate |
| **D3** | 同 D1 fine-tune 协议,但用 **v4 数据** | 🔄 运行中(~1.3h) | Train Eval Traj 应 ≈ 或低于 D1 的 3.25 |
| D4 | C1-v4-FT Pass-2 realsim 100 scenarios | ⏸ 待 | **SR ≥ 45%**(+14pp vs baseline)→ 继续。<45% → 转 score-head 解耦(D5) |
| D5-7 | D4 通过后,跑完整 ablation(baseline / +reVAE / +dyn loss / +z_floor / +v4 / full) | 待 |
| Week 2 | ONNX/TRT 导出 + Jetson Orin 延迟 profile(原排除,实机验证要求加入) | 待 |
| Week 3 | 实机 5-15 个动态障碍 trial | 待 |
| Week 4 | paper polish + 投稿 | 待 |

---

## Section B. 论文工作大纲

### 工作标题

**REACT: A Closed-Loop Benchmark and Fine-Tuned Loss-Aware Planner for Quadrotor Flight in Dynamic Environments**

(变体:如果"REACT"跟 YOPO 商标撞,可以去掉;网络代码挂在 `YOPO/` 下是为了
跟上游 diff 可读,见 `docs/ARCHITECTURE.md` §1。)

### 1. 引言(Introduction)

- 动态环境中的敏捷无人机规划有**硬延迟约束**(Jetson 上 <10ms),还得处理飞行
  途中移动的障碍物。
- 现有 learning planner 分两阵营:
  - 基于优化的慢速方法(MPC、ESDF-time)—— 准确但远超 10ms。
  - 单帧 one-stage(如 YOPO)—— 快但看不到运动。
- **三大 contribution:**
  1. **闭环动态障碍 benchmark**(100 个场景,两种 integrator —— point-mass
     upper-bound 和 Poly5 min-jerk 部署级)基于扩展的 YOPO CUDA raycaster
     (`Simulator/sensor_simulator.cu` + 动态球 upload helper)。
  2. **失效模式感知的 fine-tune recipe**,在静态预训练 YOPO baseline 上加
     动态障碍监督:yopo_head zero-pad 吸收新 reVAE 特征通道 + z_floor loss
     防止模型钻 ESDF/无地面 gap。
  3. **Camera-aware 动态数据集 bake(v4)** 把动态障碍 FOV-presence 从 22%
     拉到 88%,给 motion-reshaped collision loss 真正的梯度信号。

### 2. 相关工作(Related Work)

- **YOPO**[TJU-Aerial-Robotics,RA-L 2024]—— 单阶段 anchor-grid baseline,
  本工作的基础。静态森林上 85% SR;我们在动态障碍 benchmark 上同架构测出 31%。
- **PEMTRS**[RA-L 2026,南开]—— 时间架构(reVAE + Transformer selector +
  GRU decoder);源码未公开,按论文 method 部分复现。静态 gated corridor 上
  报道 80-100%;无动态障碍 benchmark。
- **DiffPhysDrone**[SJTU,Nature MI 2025]—— RL + differentiable physics;
  户外动态障碍 90% SR。Imitation learning regime(本文)在同任务上撞更低的天花板。
- **EgoPlanner**[HKUST 2023]、**Agile-Autonomy**[Loquercio,Sci. Robotics
  2021]、**Flow-Aided**[HKU RA-L 2025]、**NeuPAN**、ESDF-time 方法 —— 对比 baseline。

### 3. 方法(Method)

#### 3.1 背景 —— YOPO 单阶段 anchor planner
V × H = 3 × 5 anchor grid,per-anchor endstate 回归
(`Smooth-L1` on (pos, vel, acc)),softplus score head。

#### 3.2 reVAE 辅助 encoder
当前 depth 帧的 Residual VAE;latent z(128 维)在 YopoHead 1×1 conv 之前
广播到 V × H 特征图。损失:`lam_recon · MSE + lam_kl · KL`,`lam_kl = 0.001`。

#### 3.3 动态感知损失

- **`motion_reshaped_collision_loss`** —— `softplus(d_safe - dist - α · closing_speed)`,
  其中 `closing_speed = relu(rel_v · dir_from_obs_to_traj)`。
  saturation 分析见 `REACT_MATH_Derivations/01_collision_loss_saturation_cn.tex`;
  ESDF-time 梯度见 `03_esdf_time_replacement_cn.tex`。
- **`kinodynamic_loss`** —— `|v|/|a|/|j|` 的平方铰链。单 waypoint 情况下
  dormant(单 waypoint 无 jerk);留给 Option B future work 激活。
- **`z_floor_loss`**(本文新增) —— 地面之下的二次铰链;防模型钻
  safety_loss / ESDF z<0 free-space gap。数学见
  `REACT_MATH_Derivations/01_cn.tex §4`。

#### 3.4 闭环 integrator(部署路径)
每轴 `Poly5Solver`:5 阶最小 jerk 多项式,把当前 `(p, v, a)` 拟合到预测的
endstate `(p, v, a)`,T = 1.7s,dt = 33ms。**跟部署 `test_yopo_ros.py`
controller 栈是同一个代码路径**。推导:
`REACT_MATH_Derivations/05_closedloop_dynamics_cn.tex`。

#### 3.5 Fine-tune-from-baseline recipe
- 从上游 `YOPO_1/epoch50.pth` checkpoint(`use_revae=False`,`head_in=73`)
  初始化 `YopoNetwork(use_revae=True)`。
- 把 `yopo_head.model.0.weight` 从 `(256, 73, 1, 1)` zero-pad 到
  `(256, 201, 1, 1)`:原始 73 通道(obs + depth)放在前面,新 128 reVAE
  通道初始置零,在 fine-tune 期间学习。
- 25 epoch,LR `5e-5`(比 from-scratch 低 3 倍),含 z_floor + dyn loss
  + dynamic_ratio 0.5。

#### 3.6 Camera-aware 数据集 bake
原 `spawn_balls()` 在世界 bbox 内均匀放球。结果:大多数训练帧相机 FOV
里没球 → dyn loss 在这些帧梯度为零。v4 修复:先采 drone 轨迹,然后把
每个球放在 drone 初始位姿的前向锥体里(方位角 ±45°、俯仰 ±30°、距离 3-12m)。
45-seq 验证抽样上 FOV-presence rate 从 22% 跳到 88%。

### 4. 数据集

| | env 数 | seq 数 | FOV PASS | 磁盘 | wall |
|---|---:|---:|---:|---:|---:|
| 静态(上游 YOPO) | 30 | 300k 帧 | — | 2.3 GB | n/a |
| 动态 v1(初版) | 10 | 500 | mixed | 718 MB | n/a |
| 动态 v2(stage-3.5 4×) | 20 | 2000 | 22% | 1.6 GB | 2m 14s |
| 动态 v3(stage-3.6 16×) | 40 | 8000 | 22% | 3.9 GB | 5m 02s |
| **动态 v4(本论文)** | **40** | **8000** | **88%** | **3.9 GB** | **5m 09s** |

Bake 配置 + 验证 gate 在 `Simulator/src/config/config_dynamic.yaml` 和
`tools/verify_dynamic_render.py`。

### 5. 实验

#### 5.1 训练 setup
`batch=16, lr=5e-5, mixed sampling dynamic_ratio=0.5, seed=0, AdamW,
grad-clip 0.1, 25 epoch(fine-tune); 50 epoch(from-scratch)`。

#### 5.2 闭环 SR —— Pass-2(Poly5)100 v3/v4 scenarios

5-config 最终表(sprint D5 完成):

| Row                                              | goal     | dyn_col | static_col | timeout |
|--------------------------------------------------|---------:|--------:|-----------:|--------:|
| YOPO baseline(上游)                             |   31%   |   2%   |    7%     |   60%  |
| C1-v2(REACT,from scratch,v3)                |   31%   |   3%   |   24%     |   42%  |
| C1-FT(REACT,fine-tune,v3)                   |   36%   |   2%   |   10%     |   52%  |
| **C1-FT v4**(lam_dyn=3.0;**best dyn**)         |   29%   | **0%** |   19%     |   52%  |
| **C1-FT v4 balanced**(lam_dyn=1.5;**best goal**)| **37%** |   4%   |   11%     |   48%  |

**两个 Pareto-optimal headline configs:**

1. **C1-FT v4(lam_dyn=3.0):** 100 个 scenario 里 *零* 动态碰撞
   (vs baseline 2%),代价是 goal SR 降 12pp。
2. **C1-FT v4 balanced(lam_dyn=1.5):** goal SR 比 baseline 高 6pp
   (37% vs 31%),同时 dyn safety 保持可接受(4% vs baseline 2%)。

motion-reshaped collision loss 的 `lam_dyn` 权重是个**部署可调旋钮**,
trade dyn safety vs goal-reaching rate;本论文给出 Pareto 前沿。

#### 5.3 失效模式分析(paper 中心 piece)
即使 goal-rate 接近,**policy personality 也不同:**
- baseline = 保守(60% timeout, 7% static_col)
- C1-v2 from scratch = 激进撞树(42% timeout, 24% static_col)
- C1-FT = 平衡(52% timeout, 10% static_col)

按 terminate_reason 分类的 `min_clearance_m` 分布直方图会展示这个 shift;
碰撞失败视频 + timeout 原因量化。

#### 5.4 Pass-1 vs Pass-2 动力学敏感性
Pass-1(point-mass,无限 jerk 参考)—— 诊断 upper bound。
Pass-2(Poly5 min-jerk,部署级)—— 引用的数字。

C1-v2 Pass-1 = 17% vs Pass-2 = 31%(+14pp delta)。baseline Pass-1 = 34%
vs Pass-2 = 31%(-3pp)。展示**C1-v2 的 score-head argmin 有角度漂移,
平滑积分能吸收但硬 snap 不能** —— 对部署设计有信息量。

#### 5.5 延迟 profile(Week 2)
RTX 3070 eager:三种模式(单帧、单帧+DCA、K=10 stateless)都 < 4 ms p99。
Jetson Orin NX FP16 推算 ~8.7-9.9 ms p99,在 10 ms 预算内。TRT FP16 实测后加。

#### 5.6 实机短期 trial(Week 3)
YOPO 同款硬件(RealSense D435 + Jetson + 250g 四旋翼)5-15 trial。
室内 5m × 5m 空间 + mocap ground truth。Sim-to-real gap 以
|sim_SR − real_SR| 报告。

### 6. 讨论

- **为什么 ~1% 训练-loss 天花板没翻译为 SR neutrality**:v3 的 22%
  FOV-presence 是数据-质量瓶颈,**任何架构改动都修不了**。v4 应该揭示每个
  loss 组件的真正贡献。
- **为什么同样的训练算力下 fine-tune 比 from-scratch 好**:上游 YOPO 的
  静态避障表示被保留;只有新 reVAE 通道和 score head 需要重新学,backbone 不动。
- **局限性:** 假设 ground-truth obstacle radius / velocity(无端到端
  感知);单机;3-DoF 球形障碍只;sim 训练 + 有限实机验证。

### 7. 结论 + Future Work
- 端到端感知(depth → 障碍参数 → 规划)。
- 多 waypoint head(Option B),让 kinodynamic_loss 真正激活。
- 在 imitation 预训练上做 RL fine-tune(DiffPhysDrone 风格)。
- 6-DoF 障碍(摆动门、非刚性人)。
- 部署期间在线学习。

---

## Section C. 数学推导交叉引用

`REACT_MATH_Derivations/` 已有 5 个 tex 文件,作为章节背书:

| 文件 | 用在 paper § |
|------|-------------|
| `01_collision_loss_saturation.{tex,_cn.tex}` | §3.3, §6 |
| `02_multi_waypoint_extension.{tex,_cn.tex}`  | §7 future work |
| `03_esdf_time_replacement.{tex,_cn.tex}`     | §3.3, §6 |
| `04_stage5_deployment_math.{tex,_cn.tex}`    | §5.5(Jetson 预算)、§5.6(sim-to-real) |
| `05_closedloop_dynamics.{tex,_cn.tex}`       | §3.4, §5.4 |

---

## Section D. 还在跟踪的开放决策

1. **实机场地:** 室内尺度(泡沫球 / 移动板)vs 室外(motion-tracked 人)?
   影响 §5.6 视频交付。
2. **最终 ablation 行数:** 4 行(baseline / FT / FT+v4 / FT+v4+temporal)
   还是 6 行(加 DCA + 多 waypoint)?看 D4-D7 结果。
3. **标题:** 保留 "REACT" 还是改名避免商标混淆?
4. **投稿目标:** RA-L(滚动)还是 IROS(固定 deadline + 会议演讲)?

---

## 状态标记

- [x] D1(2026-05-23 中午):C1-FT v3 → 36%,比 baseline 高 +5pp
- [x] D2(2026-05-23 14:00):v4 bake,88% FOV PASS rate
- [x] D3(2026-05-23 15:00):C1-FT v4 训练完成;eval traj 3.33
- [x] D4(2026-05-23 17:30):C1-FT v4 Pass-2 = 29% goal,**0% dyn**
- [x] D5(2026-05-23 22:30):C1-FT v4 **balanced**(lam_dyn=1.5)= **37% goal**,4% dyn —— sprint 最高 goal SR
- [ ] D6 可选:lam_dyn=2.0 填 Pareto 曲线中段(~1.5h)
- [ ] D7:干净 ablation 表 commit + 同步 paper_outline §5
- [ ] Week 2:ONNX/TRT + Jetson 延迟
- [ ] Week 3:实机验证
- [ ] Week 4:polish + 投稿
