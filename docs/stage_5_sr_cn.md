# Stage-5.B 闭环 SR 最终报告(中文版)

**日期:** 2026-05-22
**Sim:** REACT `sensor_simulator` ROS 节点(Simulator/,stage-5.B.1 动态球扩展),从 `cwd = Simulator/` 启动,这样静态森林点云能正确加载。
**场景:** `tools/eval_scenarios/v3_dyn_100/` 里 100 个动态场景(每个 3-8 球,1-5 m/s,drone goal 20m 前方 ±30° yaw 散布)。
**驱动:** `scripts/stage5_closedloop_eval.py` Pass-1(限幅双积分器)和 Pass-2(Poly5 最小 jerk,部署同款)。

英文版:[stage_5_sr.md](stage_5_sr.md)

---

## 最终 2×2 结果表

| 指标                    | baseline(上游 YOPO_1) | C1-v2(REACT stage-3.1 + z_floor) | Δ        |
|-------------------------|---------------------------:|-----------------------------------:|---------:|
| Pass-1 goal             |    34%                     |       17%                          | **-17pp** |
| Pass-1 dyn_collision    |     2%                     |        1%                          |    -1    |
| Pass-1 static_collision |    25%                     |       55%                          | **+30pp** |
| Pass-1 timeout          |    39%                     |       27%                          |   -12    |
| **Pass-2 goal**         |   **31%**                  |   **31%**                          | **0pp** |
| Pass-2 dyn_collision    |     2%                     |        3%                          |    +1    |
| Pass-2 static_collision |     7%                     |       24%                          | +17pp   |
| Pass-2 timeout          |    60%                     |       42%                          | -18pp   |

CSV 文件:
- [results/stage5_sr_baseline_v3_realsim.csv](../results/stage5_sr_baseline_v3_realsim.csv)
- [results/stage5_sr_baseline_v3_realsim_pass2.csv](../results/stage5_sr_baseline_v3_realsim_pass2.csv)
- [results/stage5_sr_C1_v2_realsim_pass1.csv](../results/stage5_sr_C1_v2_realsim_pass1.csv)
- [results/stage5_sr_C1_v2_realsim_pass2.csv](../results/stage5_sr_C1_v2_realsim_pass2.csv)

**Pass-2 数字是要跟部署目标一起 cite 的** —— 它们用的 Poly5Solver 跟 `YOPO/test_yopo_ros.py` 喂 SO(3) controller 的是同一个(详见 [REACT_MATH_Derivations/05_closedloop_dynamics_cn.tex](../REACT_MATH_Derivations/05_closedloop_dynamics_cn.tex))。

---

## 头版判决

**REACT stage-3.1(path-c,loss-only)在我们的 benchmark 上闭环 SR 是 *neutral*:** Pass-2 下 31% vs 31%。两种方法都远低于 ≥85% paper target。

但是**失效模式截然不同:**
- baseline **保守** —— 60% timeout,但只 7% 撞树。上游 YOPO 学会减速或绕开树多的区域。
- C1-v2 **激进** —— 42% timeout(更快冲 goal)但 24% 撞树。50/50 静态/动态 mixed-sampling 稀释了静态训练信号;C1-v2 在树林里的导航能力大幅低于 baseline。

stage-3.1 motion-reshaped loss **没有可测的动态避障改善**(dyn_collision 2% vs 3% 在噪声里),但**明显改变了无人机的"性格"**。

---

## Pass-1 vs Pass-2 差距为什么这么大

| 模式                | Pass-1(point_mass) | Pass-2(Poly5) |
|---------------------|--------------------:|---------------:|
| baseline goal       |    34%              |     31%       |
| C1-v2 goal          |    17%              |     31%       |
| baseline static_col |    25%              |      7%       |
| C1-v2 static_col    |    55%              |     24%       |

Pass-1 每 33ms "snap"到 planner 预测速度方向。score-head 的 argmin 里任何小角度
漂移(比如 C1-v2 偶尔选个偏角 anchor)会立刻执行,在障碍附近震荡 → 频繁撞树。

Pass-2 用 5 阶最小 jerk 多项式在当前状态和预测 endstate 之间拟合,然后步进 dt。
同样的角度漂移变成 drone 能跑完的平滑曲线。baseline+Poly5 把 static_collision
从 25% 压到 7%(-18pp);C1-v2+Poly5 把它从 55% 压到 24%(-31pp)。

**结论:部署时,Poly5 路径是必选项。** Pass-1 数字用来诊断 planner-vs-controller
失败模式拆分有用,但**不能当 headline cite**。

---

## 一路上摸出来的两个非显然发现

### F1 —— 从错误的 cwd 启动 sim 会**默默禁用整个森林场景**

`Simulator/src/src/maps.cpp::Maps::forest()` 在 `tree.ply` 加载失败时早退(988-992 行)。
`tree_file` config 项是相对路径,所以从 `cwd=REACT/` 启动 `sensor_simulator`
(而不是 `cwd=Simulator/`)文件查找失败,早退 kill 掉树和地面生成,publish 的
`/depth_image` 只显示随机墙壁。

修复之前测的所有 SR 数字(老分支上的 23% / 27% / 66% / 0% 那些)都是在**没有树的
场景**测的。它们保留在 git 里作 audit,**但不应该被 cite**。

### F2 —— Driver yaw bug 被坏 sim 掩盖了

Plan A driver 原本把 drone yaw 跟 velocity 方向对齐。当训练好的模型预测强 forward
+ 几乎零 lateral 的 endstate 时(C1 恰好这样),drone 永远不向 lateral 偏移的 goal
转弯 —— 它沿固定 yaw 直线飞,miss goal。修复:yaw 跟 `goal - drone_pos` 方向,
速率限 2 rad/s,跟部署用的 `YOPO/policy/poly_solver.py::calculate_yaw` 一致。

---

## 三个最后**没起作用**的边路实验

### S1 —— 无 z_floor 的 C1-v1

最初 stage-3.1 50-epoch 训出来的 `YOPO_11/epoch50.pth` 一致预测 endstate
~2m 低于地面。根因:`YOPOLoss.safety_loss` 查询的 ESDF **没有地面平面**,
z<0 默认 "free space" 所以 score head 学会给"向下看"anchor 低 cost。坏 sim 下
这导致 SR=0/100。我们没在 real-sim 里再测它,因为那时已经有 Plan B 的 z_floor 修复了。

`REACT_MATH_Derivations/01_collision_loss_saturation_cn.tex §4` 已经从结构上
点出这是 safety_loss formulation 的固有特征;经验现象确认了它。

### S2 —— Driver anchor filter(`end_z > -1m`)

Plan A 给 driver 打了个 patch,在推理时无视"向下看"anchor。坏 sim 下这把 C1 SR
从 0% 拉到 27%。配上 Plan B(z_floor 训练)后这个 filter 实际上变成 no-op,因为
C1-v2 几乎不预测 z<z_floor 了。我们保留 filter 是因为去掉它对 C1-v2 数字没什么变化。

### S3 —— baseline-v2-hack 66% 是 sim artifact

提交 `f4b7427` 里报的 66% baseline goal-rate 是在**坏(空)sim** 上测的 —— drone
没东西可撞。修复 sim 加回真森林后,baseline 降到 Pass-1 34% / Pass-2 31%。

---

## 对论文的启示

1. **闭环 SR 表是 stage-3.1 的 definitive REACT 结果**。磁盘上的 loss 数字
   (Train/DynLoss 0.11 等)和 stage-3.x / stage-4 的 ablation 表应该呈现但
   解读为"不能预测闭环 SR 的内部训练指标"。

2. **Stage-3.1 结果两种 framing:**
   - **Null result frame:** "我们发现 REACT-stage-3.1 闭环 SR 跟上游 YOPO
     相等(31% vs 31%),但失效模式变化(更激进、撞更多树、timeout 减少)。
     motion-reshaped dyn-loss 训练在我们的 benchmark 下没有解锁动态避障。"
   - **Failure-mode analysis frame:** "Stage-3.1 行为改变可检测且一致(更激进、
     撞更多、timeout 减少),所以 loss 在 shape policy —— 但 shape 对 SR
     不是净正向。Future work:weighted mixed-sampling 或 fine-tune from
     baseline 而不是 from scratch。"

3. **85% paper target 两种方法都没到。** 这是 honest 报告;paper 要么需要
   更多架构/数据工作(Option B 多 waypoint head、camera-aware bake、数据
   规模扩到 8000+),要么 reframe 为分析本身(闭环 bench、sim-vs-deploy
   gap 量化、失效模式对比)。

> **更新 2026-05-23**:已经按 §2 第二个 framing 走 sprint 计划。D1(fine-tune
> from baseline)实测 36% SR(+5pp baseline),正在测试 D2 v4 camera-aware
> 数据集(FOV 22% → 88%)能不能再加分。最新数据见 [paper_outline_cn.md](paper_outline_cn.md) §A.3。

---

## 没测过的东西(以及为什么没测也能 ship)

- C1-v1 在 real-sim 里(坏 sim 给 0%;z_floor 是 v2 专门修这个加的)
- Stage-3.2(DCA path-b)在 real-sim 闭环里(已有 dyn_dyn 1% ablation 说是 noise;
  闭环 SR 预期跟 baseline 一致)
- Stage-3.4(path-a temporal)在 real-sim 闭环里(5k iter dyn_dyn 1%;完整
  50-epoch 重训 + real-sim SR 是自然 follow-up,如果想要多行 ablation 表)
- C2 stage-3.4 v2(z_floor + temporal)算了 5-7h 额外训练 + 测,没开始

每个都是 2-4h 从当前状态额外投资。**建议:只在 paper outline 要那一行 ablation
时再跑。**

---

## 复现命令

```bash
# 从 Simulator/ 启动 sim(cwd 重要 —— tree.ply 路径是相对的!)
cd Simulator && ./devel/lib/sensor_simulator/sensor_simulator &
cd ..

# Baseline(上游 YOPO checkpoint)
python scripts/stage5_closedloop_eval.py \
  --ckpt YOPO/saved/YOPO_1/epoch50.pth \
  --label baseline_v3_realsim \
  --dynamics poly5 \
  --out results/stage5_sr_baseline_v3_realsim_pass2.csv

# C1-v2(REACT stage-3.1 + z_floor,50 epoch)
python scripts/stage5_closedloop_eval.py \
  --ckpt YOPO/saved/YOPO_12/epoch50.pth \
  --use-revae \
  --label C1_v2_zfloor_realsim_pass2 \
  --dynamics poly5 \
  --out results/stage5_sr_C1_v2_realsim_pass2.csv

# 每个 run:RTX 3070 上 ~15 分钟。
```
