# REACT 架构(中文版)

本文档解释仓库的三包布局、bake 时和训练时的数据流,以及 REACT 的改动相对于
上游 YOPO baseline 的位置。

英文原版:[ARCHITECTURE.md](ARCHITECTURE.md)

---

## 为什么保留 `YOPO/` 子目录的名字

外层目录布局(`Controller/ Simulator/ YOPO/` 三个并列子目录)从**结构层面**字节
对齐 [TJU-Aerial-Robotics/YOPO](https://github.com/TJU-Aerial-Robotics/YOPO)。
熟悉 YOPO 的人可以直接 `git diff` 两个树看 REACT 改了哪些文件。

改名比如改成 `react_planner/` 会丢掉这个性质,而且会带 200+ 行的 churn
(路径、import、CMake、文档、每条提到文件名的 commit message)。决定:
**改名推迟到 v1.0 release 再考虑**。README 的目录树有一行注释指向这里。

## 三个包

| 包 | 角色 | Build chain | REACT 改动(截至目前) |
|---|---|---|---|
| **Controller/** | ROS quadrotor 动力学 + so3 位置/姿态控制器 | `catkin_make` | 无(跟 YOPO 一致) |
| **Simulator/** | CUDA raycaster 打随机森林点云;ROS sensor publisher + 离线 dataset generator | `catkin_make --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5` | stage-2:`sensor_simulator.cu` 加 `DynSphere` struct + `ray_sphere_depth` device 函数;`dataset_generator.cpp` 加 `mode: "dynamic"` 分支,bake K-frame 序列 |
| **YOPO/** | Python 训练 + 推理(depth → 轨迹) | `pip install -r YOPO/requirements.txt`(系统 Python;preflight 自检) | stage-1:`policy/models/{revae,temporal_selector,gru_decoder}.py` + `policy/utils/frame_buffer.py` + `yopo_network.py` 和 `yopo_trainer.py` 集成 reVAE loss;stage-2:`policy/yopo_dataset.py` 加 `--dynamic` 开关 |

顶层另外两个小目录:

- `scripts/` —— preflight、smoke test、training runner、tfevents extractor。
  独立 Python 脚本,不是 package。
- `tools/` —— 用来验证 `Simulator/` bake 出来的数据的脚本。

## 数据流

### Bake 时(stage-2 D-3,mode=dynamic)

```
config_dynamic.yaml ──┐
                      ▼
 ┌─────────────── dataset_generator (C++, Simulator/) ────────────────┐
 │                                                                     │
 │   for each env:                                                     │
 │     mocka::Maps -> 静态点云(forest)                                │
 │     GridMap     -> CUDA voxel index                                 │
 │     pcl::KdTreeFLANN -> 查询 drone-spawn 的 safe_dist               │
 │                                                                     │
 │     for each sequence (K=10 帧):                                    │
 │       生成 3-8 个 DynamicBall(随机 pos/vel/radius)                │
 │       采样无人机轨迹:K 帧,body-+X 速度,任何一帧                  │
 │                       撞点云就重采样                                │
 │                                                                     │
 │       for each k in [0, K):                                         │
 │         球前进一帧(在 bbox 边界做 axis-aligned 反弹)              │
 │         uploadDynamicSpheres -> GPU                                 │
 │         renderDepthImage(GridMap, camera, T_wc, DynSphere*) ─┐      │
 │                                                              │      │
 │       depth_t{k}.png(uint16,以 max_depth_m=20m 归一化)      │      │
 │       state.json(每帧 pos, quat_wc, vel_world)               │      │
 │       dyn_obs.json(每帧的 ball pos/vel/radius)               │      │
 │       meta.json(K, dt, intrinsics, depth_encoding)            │      │
 └──────────────────────────────────────────────────────────────┴──────┘
                                                          dataset_dynamic/v2/
```

同一个 `renderDepthImage()` 函数(`n_dyn=0`)产出原始静态数据集
`dataset/` —— **只有一个 renderer**,静态和动态数据在任何动态球没碰过的
voxel 上像素级分布一致。

### 训练时

**Stage 1 / 3.1 / 3.2 单帧 forward 路径(回退默认):**

```
dataset/                                  YOPODataset (dynamic=False)
  env_X/depth_*.png    ────────────────►   __getitem__ 返回 5-tuple
  pose-X.csv                              (image, pos, rot, obs, map_id)
                                                       │
                                                       ▼
                                            YopoNetwork.forward(depth, obs)
                                              │
                                              ├─ ResNet18(depth 特征,V×H)
                                              ├─ ReVAE(depth → latent 128)              🟩 stage-1
                                              ├─ 广播 latent 到 V×H
                                              └─ cat → YopoHead → (9, V, H) + V×H score
                                                       │
                                                       ▼
                                          YOPOLoss = smooth + safety + goal + acc
                                          + score Huber                                  🟦 YOPO
                                          + 0.1 * (mse_recon + 0.001 * KL)               🟩 stage-1
                                          + motion_reshaped + kinodynamic(动态batch)    🟧 stage-3.1
```

**Stage 3.4 K-frame 路径(启用 `frame_buffer.enable_temporal=true` 后):**

```
dataset_dynamic/v2/                       DynamicYOPOWrapper(return_kframe=True)
  env_X/seq_Y/depth_t*.png  ────────────►  __getitem__ 返回 (K,1,H,W) depth_seq
  env_X/seq_Y/state.json                  + 之前的 6 个 tuple 元素
  env_X/seq_Y/dyn_obs.json                              │
                                                        ▼
                                          YopoNetwork.forward(depth_seq, obs)
                                              │
                                              ├─ ResNet18 跑 depth_seq[:, -1](最后一帧)
                                              ├─ ReVAE 在 batch reshape 上跑全部 K 帧 → (B, K, 128)  🟦 stage-3.4
                                              ├─ TemporalAggregator GRU → (B, 128)                   🟦 stage-3.4
                                              ├─ 广播到 V×H(替换 stage-1 的 z 槽位)
                                              └─ cat → YopoHead(不变)
                                                       │
                                                       ▼
                                          loss 不变(reVAE 重构监督换成最后一帧)
```

## 模块图

🟧 = REACT 新增;🟩 = stage-1 PEMTRS 移植;🟦 = stage-3.4 path-a;其他从上游 YOPO
继承或是 REACT 加的不动上游行为的工具。

```
YOPO/
├── train_yopo.py                   stage-1 入口;实例化 YopoTrainer
├── policy/
│   ├── yopo_network.py             🟩 stage-1: reVAE 接到 YopoNetwork;
│   │                               🟦 stage-3.4 phase 2: K-frame forward + TemporalAggregator
│   ├── yopo_trainer.py             🟩 stage-1: revae_loss 进总和;
│   │                               🟦 stage-3.4 phase 3: cfg flag + 自动 K-frame 扩展
│   ├── yopo_dataset.py             🟧 stage-2 2.e: --dynamic 开关;
│   │                               🟦 stage-3.4: DynamicYOPOWrapper.return_kframe
│   ├── poly_solver.py / primitive.py / state_transform.py     (YOPO 上游)
│   ├── models/
│   │   ├── backbone.py / head.py / resnet.py                  (YOPO 上游)
│   │   ├── revae.py                🟩 stage-1: residual VAE encoder
│   │   ├── temporal_selector.py    🟩 stage-1: Transformer ROI selector(未接图)
│   │   ├── gru_decoder.py          🟩 stage-1: GRU + cross-attn + 三头(未接图)
│   │   ├── dynamic_attention.py    🟧 stage-3.2: DynObsEncoder + DynamicCrossAttention
│   │   └── temporal_aggregator.py  🟦 stage-3.4: nn.GRU 薄包装(99K 参数)
│   └── utils/
│       └── frame_buffer.py         🟩 stage-1: K-frame 滑窗(未接图;数据加载走 dataset)
├── loss/
│   ├── loss_function.py            🟩 stage-1: revae_loss() 静态方法
│   │                               🟧 stage-3.1: dyn_collision_loss + kinodynamic_loss 包装
│   ├── motion_reshaped_esdf.py     🟧 stage-3.1: motion-reshaped collision loss
│   ├── kinodynamic_loss.py         🟧 stage-3.1: V/A/J 包络
│   └── smoothness_loss.py / safety_loss.py / guidance_loss.py (YOPO 上游)
└── config/
    └── traj_opt.yaml               🟩 stage-1: + revae/frame_buffer/selector/gru_decoder/loss_weights
                                    🟧 stage-2 2.e: + dataset_dynamic_path
                                    🟧 stage-3.1/3.2: + dynamic_attention + motion_reshaped + kinodynamic
                                    🟦 stage-3.4: + frame_buffer.enable_temporal

Simulator/src/
├── include/
│   ├── sensor_simulator.h          (YOPO 上游)
│   ├── sensor_simulator.cuh        🟧 stage-2 2.a: + DynSphere struct + 扩展的函数签名
│   └── maps.hpp / perlinnoise.hpp                              (YOPO 上游)
├── src/
│   ├── sensor_simulator.cu         🟧 stage-2 2.a: + ray_sphere_depth + kernel 动态循环 + upload/free helpers
│   ├── dataset_generator.cpp       🟧 stage-2 2.b: + 动态 mode orchestrator + DynamicBall + 内联 JSON
│   ├── test_dyn_sphere.cpp         🟧 stage-2 2.a: smoke(3/3 PASS @ <0.05m)
│   └── sensor_simulator.cpp / test_simulator*.cpp / perlinnoise.cpp / maps.cpp   (YOPO 上游)
└── config/
    ├── config.yaml                 (YOPO 上游;mode: static)
    └── config_dynamic.yaml         🟧 stage-2 2.b: mode: dynamic + sequence 参数

scripts/                            (REACT 工具,全部 🟧/🟦)
tools/                              (REACT 工具,全部 🟧/🟦)
docs/                               文档 + 可视化样本
```

## Path-c(stage-3.1)和 path-b(stage-3.2):实验发现

原先的"未接图模块"计划提供了三条把动态障碍通道接到 forward 图的路径
(c:loss-only;b:side-channel token;a:full K-frame)。c、b、a **三条
都已实现并在 `main` 上**;以下是结论。

### Stage-3.1(path c,`v0.3.1-loss-only`)

`motion_reshaped_collision_loss` 和 `kinodynamic_loss` 作为独立的加权项
加进 YOPOLoss 类;`forward` 没动。50% 静态 / 50% 动态的混合采样加这两个新
损失被 `scripts/run_stage3_1k.py` 验证 —— 五个过关 gate 全过。

### Stage-3.2(path b,`v0.3.2-side-channel`)

`YopoNetwork.forward` 加了可选的 `dyn_obs_tokens` / `dyn_obs_mask` 参数。
当 `cfg.dynamic_attention.enable=true` 时网络实例化 `DynObsEncoder`
(7 → 201 维 MLP)和 `DynamicCrossAttention`(n_heads=1,因为 head_in=201
除不尽 4)。trainer 从 bake 出的 dyn_obs payload 构造 token:
`[rel_pos, abs_vel, radius]`,其中 `rel_pos = obs_pos - drone_pos`
(世界系,不做 yaw 旋转)。

### 5k iter A/B 对比(DCA off vs DCA on,同数据集、同 lr、batch=16)

| 指标 | path c(DCA off) | path b(DCA on) | Δ |
|---|---|---|---|
| total head → tail | 5.52 → 4.20 | 5.52 → 4.17 | -0.03(0.8%) |
| static traj head → tail | 4.13 → 3.55 | 4.27 → 3.49 | -0.05(1.5%) |
| **dyn dyn head → tail** | **0.236 → 0.225** | **0.239 → 0.223** | **-0.002(0.9%)** |

DCA 在当前训练规模上给动态损失带来 ~1% 的边际改善 —— **在噪声里**。架构 plumbing
没问题(sub-A 到 sub-D smoke 测试全过;DCA 恰好在 `dyn_obs_tokens` 给的时候
触发,梯度回流到 token),但**侧通道在当前数据集规模下没拉开**(450 训练 seq,
50% 采样 → 每 epoch effective ~225 个动态 batch)。

### 关于差距没拉开的假设,按可能性排序

1. **Dataset starvation(数据饥饿)。** DCA 参数(205 K)只在动态 batch 上
   收梯度;那是 ~50% 的训练步。5k iter 里 DCA 从 random init 起只有
   ~2.5k 次有效更新 —— 不足以学到 obstacle-attention 模式。
2. **Token 跟 depth 信息冗余。** 深度图已经编码了障碍位置。再通过 token 喂
   同样的信息可能是冗余的,除非网络能跨帧解码"这是同一个球" ——
   stage-3.2 单帧 forward 做不到。
3. **世界系 token 不做 yaw 旋转。** 网络要隐式学这个 yaw 变换;
   能训但耗能力。
4. **静态 safety_loss 污染。** 动态 batch 用随机的静态 `map_idx` 让
   `YOPOLoss.safety_loss` 查一个跟当前动态场景不匹配的 ESDF —— 这是
   stage-3.1 故意走的捷径,但可能给梯度信号加噪。

### 决定:ship stage-3.2 现状,记录负结论,不再在这投入

架构是正确且可复用的。未来想测"path-b 真的有用吗"应该先解决 (1) —— 把动态
数据扩 4-8×,或者转向 **path a(完整 K-frame forward)**,后者理论上限更高,
因为网络可以从深度序列里直接学障碍运动。

论文里:stage-3.1 是 headline 架构;stage-3.2 在 ablation 表里写一行
"显式 GT token 侧通道在当前规模下加 <1%"。

## Stage-4 ablation(`v0.4-ablation`)

`scripts/run_stage4_ablation.py` 在跑之间内存里改 `cfg._data`,每行新开
一个 `YopoTrainer`,一次调用产出 head-to-head 的表。5 行是叠加的:

| 行 | reVAE | DCA | dyn_ratio | lam_dyn | lam_kino |
|---|---|---|---|---|---|
| A baseline-yopo | off | off | 0.0 | 0.0 | 0.0 |
| B +reVAE         | on  | off | 0.0 | 0.0 | 0.0 |
| C +dyn/kino loss | on  | off | 0.5 | 3.0 | 0.5 |
| D +DCA only      | on  | on  | 0.5 | 0.0 | 0.0 |
| E full           | on  | on  | 0.5 | 3.0 | 0.5 |

### 结果(batch=16,lr=1.5e-4,seed=42)

尾窗均值(最后 10% 步)。`nan` 意味着这行从来没产出过动态 batch。完整 CSV
(带 `dyn_kino_tail`、`n_dyn`/`n_stat`、wall-clock)在 `results/`。

**5k iter run**(`results/ablation_5k.csv`,~13 min):

| Row | total | stat_traj | dyn_traj | dyn_dyn | reVAE |
|---|---:|---:|---:|---:|---:|
| A baseline-yopo | 3.84 | 3.50 | nan  | nan  | 0.00  |
| B +reVAE         | 3.85 | 3.48 | nan  | nan  | 0.036 |
| C +dyn/kino loss | 4.18 | 3.54 | 3.71 | **0.224** | 0.036 |
| D +DCA only      | 4.09 | 3.55 | 3.72 | 0.00 | 0.035 |
| E full           | 4.20 | 3.55 | 3.72 | **0.226** | 0.036 |

**2k iter run**(`results/ablation.csv`,~5 min)—— 同样的趋势,作为快速
迭代复现目标保留:

| Row | total | stat_traj | dyn_traj | dyn_dyn | reVAE |
|---|---:|---:|---:|---:|---:|
| A baseline-yopo | 4.16 | 3.61 | nan  | nan  | 0.00 |
| B +reVAE         | 4.15 | 3.62 | nan  | nan  | 0.044 |
| C +dyn/kino loss | 4.49 | 3.74 | 3.82 | **0.23** | 0.047 |
| D +DCA only      | 4.33 | 3.64 | 3.83 | 0.00 | 0.044 |
| E full           | 4.43 | 3.64 | 3.83 | **0.22** | 0.045 |

### 发现(2k 和 5k iter 一致)

1. **reVAE 单独(B vs A)是 wash**:5k Δtotal = +0.003,Δstat_traj = -0.014
   —— 在噪声里。reVAE 只在 path-a 时间 forward 跨帧用它的后验时才发挥作用,
   现在还没做(path-a 用了但是只看最后一帧重构;时间信号被 GRU 抓走了)。
2. **dyn/kino loss(C vs B)** 把 total 拉高(5k:3.85 → 4.18),因为
   多了一项罚款,但**首次**引入有意义的动态碰撞信号(`dyn_dyn` 0.224 vs
   `nan`)。**这是 stage-3.1 真正的贡献**。
3. **DCA 没有 dyn-loss 信号(D vs B)** 在 `total` 上稍微差一点
   (5k:4.09 vs 3.85)—— 多参数,除了共享的静态 traj/score 监督外没有梯度
   方向。**确认 DCA 必须配损失才能学到东西**。
4. **DCA 加在 dyn-loss 之上(E vs C)** 让 `dyn_dyn` 从 0.224 变到 0.226
   (5k),从 0.23 变到 0.22(2k)—— ~1% 围绕噪声的震荡。和 stage-3.2 5k
   A/B(~1% 改善)一致。**5k 比 2k 没拉开,所以不是 DCA 在 2k 训得不够;
   是它在当前数据规模下真的没拉开**。
5. `dyn_kino_tail` 在每一行运行 kinodynamic loss 的地方基本都是 0
   (5k 时 0.0001,2k 时严格 0)—— 配置的包络(`v_max=8`、`a_max=10`、
   `j_max=30`)太宽松,单 waypoint 预测几乎打不到。损失是接对的但在这些
   默认值下不触发;**包络只对多 waypoint 轨迹(path a)有约束**(path-a
   实测仍然没触发,见下)。

### 论文含义

- Headline:stage-3.1 dyn/kino loss 加上可测量的动态碰撞信号,代价是小的
  静态 traj。
- Stage-3.2 DCA 是 ablation 表里的一行,记录"GT obstacle token 的 naive
  侧通道在当前数据规模下没用"。
- 复现:
  - 2k(~5 min):`python scripts/run_stage4_ablation.py --steps 2000`
  - 5k(~13 min):`python scripts/run_stage4_ablation.py --steps 5000 --out results/ablation_5k.csv`

## Stage-3.4 K-frame temporal forward(path a,`v0.3.4-temporal`)

按 `docs/stage_3_4_design_cn.md` Option A:depth 输入变成 (B, K, 1, H, W);
reVAE 一次编全部 K 帧;`TemporalAggregator`(nn.GRU,99K 参数)把 K 步
latent 序列压成单个 embedding,占据 stage-1 z 在广播槽里的位置。
**YopoHead anchor 网格和每一项 stage-3.1 损失都没动**。

### v2 数据集上的 A/B(5k iter,batch=16,seed=42)

| 指标 | F temporal off | G temporal on | Δ |
|---|---:|---:|---:|
| total tail | 4.261 | 4.187 | -0.074(-1.7%) |
| stat_traj tail | 3.544 | 3.559 | +0.015(+0.4%) |
| dyn_traj tail | 3.762 | 3.770 | +0.008(+0.2%) |
| **dyn_dyn tail** | **0.2366** | **0.2343** | **-0.0023(-1.0%)** |
| dyn_kino tail | 0.0001 | 0.0001 | 0 |
| reVAE tail | 0.0355 | 0.0427 | +0.0072(+20%) |
| wall_s | 145.4 | 219.1 | **+51%** |

完整 CSV 在 `results/ablation_stage_3_4.csv`。

### 判决:又一个 <1% 的负结论,但**失败的原因不同**

Path-a K-frame forward 把 `dyn_dyn` 改善 **1.0%** —— 跟 stage-3.2 DCA 侧通道
v1 上的 ~1% 一样,也跟它在 v2 上重测的 ~1%(F vs E_full)一样。
**三个独立的架构升级全部在同一个 ~1% 上限收敛**。

§1 的成功标准(≥5% dyn_dyn 下降)没达成。但 **path-a 失败的方式跟 path-b
不同**:

- Path-b(DCA)直接把 GT obstacle token 喂给网络;如果网络在 500-2000 seq
  数据规模下用不起来,更多架构帮不了。
- Path-a(temporal)让网络**自己从 K-frame 深度序列里找运动信号**。这是
  严格更灵活的 formulation,在信息层面包含了 path-b 能提供的所有内容。

两条独立机制都败在 ~1%,强烈指向**瓶颈是数据规模,不是架构**。
`docs/stage_3_4_research_cn.md` 调研的所有 temporal 论文都用了至少 10×
的数据(UCF101:13k clips;DiffPhysDrone:无限 RL)。

### 计算代价

K-frame temporal **每训练步 +51% wall**(10× reVAE encode,部分被 GPU 把
backbone+head 缓存温了的事实抵消)。推理时同样的 K=10 stateless 路径
会把每帧延迟乘 ≈10×。**stage-5 部署必须重新评估 DiffPhysDrone 的
stateful-GRU 模式**(单帧 encode + 跨帧保留 hidden),否则在 Jetson 上
profile 大概率打不到 <10 ms。

### 论文含义

- Path-a(temporal)进 ablation 表作为第三行,展示"架构升级在当前数据规模
  下也不能拉开 baseline"。和 path-b、path-c 组合,**清晰地指向"下一步
  实验":扩动态数据到 10k+ 个 seq**。
- Headline 贡献仍然是 stage-3.1(dyn/kino loss)。Stage-3.2 和 stage-3.4
  是两行 ablation。
- 复现:
  - `python scripts/run_stage4_ablation.py --steps 5000 --only F_v2_temporal_off,G_v2_temporal_on --out results/ablation_stage_3_4.csv`

## Build/run 命令速查

```bash
# 每次开发 session
bash scripts/preflight.sh

# 一次性建 ROS 包(每台机器一次)
cd Controller && catkin_make --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cd ..
cd Simulator  && catkin_make --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cd ..

# bake 静态数据集(rosrun pipeline,见 README Quick Start)
# bake 动态数据集
cd Simulator
./devel/lib/sensor_simulator/dataset_generator --config src/config/config_dynamic.yaml

# 训练(stage-1 默认;仅静态数据集)
cd YOPO && python train_yopo.py

# Smoke 检查
python scripts/smoke_stage1.py                  # 模块形状
python scripts/smoke_stage1_integration.py      # forward + backward
python scripts/smoke_stage1_train.py            # 100 iter on real data
python scripts/smoke_stage2_2e.py               # dataset --dynamic + 静态回归
python scripts/smoke_stage3_4a.py               # 🟦 TemporalAggregator 单元
python scripts/smoke_stage3_4b.py               # 🟦 YopoNetwork K-frame forward
python scripts/smoke_stage3_4c.py               # 🟦 trainer K-frame 端到端
python tools/verify_dynamic_render.py dataset_dynamic/v2/env_0008/seq_0007
```
