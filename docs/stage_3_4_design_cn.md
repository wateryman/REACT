# Stage-3.4 Path-a 设计文档(中文版)

**状态:** 草稿,等待评审。  §1-§5 定义要做的内容;§6 列出 stage-3.4 不做的变体;
§7 是 5 阶段实施计划。

英文版:[stage_3_4_design.md](stage_3_4_design.md)
对应文献调研:[stage_3_4_research.md](stage_3_4_research.md)

---

## 1. 目标 & 非目标

### 目标

把 `YopoNetwork.forward` 的输入从**单帧 depth** 换成 **K 帧 depth 序列**,
**不动 anchor-grid 的 `YopoHead`,也不动 stage-3.1 的任何损失**。
网络从时间维度学障碍物运动;stage-3.1 监督它的那一套损失
(`motion_reshaped_collision_loss`、`kinodynamic_loss`、trajectory、score、reVAE)
原封不动复用。

### 非目标(明确推迟到未来 stage 或永不实现)

- 用 `gru_decoder.GRUDecoder`(多 waypoint 输出)替换 `YopoHead`。
  详见 §6 "Option B" — 不同 stage,不同 head,不同 loss 路由。
- 在 forward 路径里激活 `TemporalRegionSelector` 的未来 ROI 预测。
  模块留在磁盘上但旁路。
- 端到端感知(depth → 障碍物参数)。仍然通过现有 `DynamicYOPOWrapper`
  通道消费 GT 障碍物 token。
- 实时推理的形状重整(TensorRT / ONNX 导出);放到 stage-5。

### 成功标准

1. **回归干净的退路。** 当 `cfg["frame_buffer"]["K"] = 1` 时,
   网络跟 stage-3.1 字节级一致(忽略 `mean()` reduction 跨一个 dummy 维
   的 fp32 数值噪声)。
2. **K=10 时训练能端到端跑通**,v2 数据集上无 NaN/Inf,batch_size=16
   能放下显存(本机大约 12-16 GB,phase 1 里确认)。
3. **5k iter A/B vs stage-3.1**:同一数据切片下,temporal forward 让
   `dyn_dyn` 相对 stage-3.1 降 ≥ 5 %。低于 5 % 也 ship,作为下一行
   ablation("当前规模下 temporal forward 加 X %")—— paper 仍受益。

---

## 2. 现有可复用模块

| 模块 | 路径 | 当前状态 | stage-3.4 怎么用 |
|---|---|---|---|
| `ReVAE` | [policy/models/revae.py](../YOPO/policy/models/revae.py) | stage-1;一次编码一帧 | 包一层批量调用,通过 batch reshape 一次性编 K 帧 |
| `YopoBackbone`(ResNet-18) | [policy/models/backbone.py](../YOPO/policy/models/backbone.py) | stage-0;一帧编码到 V×H 特征图 | **只对最后一帧跑**(Option A 保留单帧 depth 特征通路 — 是这个 Option 最关键的简化点) |
| `YopoHead`(1×1 conv) | [policy/models/head.py](../YOPO/policy/models/head.py) | stage-0;`(B, head_in, V, H) → (B, 10, V, H)` | **不动**。这是 Option A 的载荷决策 |
| `TemporalRegionSelector` | [policy/models/temporal_selector.py](../YOPO/policy/models/temporal_selector.py) | stage-1 写好但**未接图** | stage-3.4 **旁路**(推迟到 Option B / 未来) |
| `GRUDecoder` waypoint 输出头 | [policy/models/gru_decoder.py](../YOPO/policy/models/gru_decoder.py) | stage-1 写好但**未接图** | **旁路**。我们直接实例化 `nn.GRU`(§3.3),跳过多 waypoint 输出头 |
| `DynamicCrossAttention` | [policy/models/dynamic_attention.py](../YOPO/policy/models/dynamic_attention.py) | stage-3.2;侧通道接 anchor token | canonical 3.4 **旁路**(dyn_obs token 仍然流到,但 DCA 开关还看 `cfg["dynamic_attention"]["enable"]`;ablation 可以再叠加打开) |
| `_load_dynamic` | [policy/yopo_dataset.py:153](../YOPO/policy/yopo_dataset.py) | stage-2;**已经**返回 `(K, 1, H, W)` depth_seq + state_seq + dyn_obs + dt_seq | `DynamicYOPOWrapper` 当前在第 365 行扔掉 K−1 帧。stage-3.4 不再扔 |
| `motion_reshaped_collision_loss` + `kinodynamic_loss` | [loss/](../YOPO/loss/) | stage-3.1 | 不动。仍然作用在 `YopoHead` 预测的 endstate 上 |

---

## 3. 新 forward 图(canonical Option A)

```
                    ┌─────────────────────────────────────────────┐
        depth_seq   │                                             │
   (B, K, 1, H, W) ─┤  reVAE.encode(批量 B*K 一起)               │   stage-3.4 新增
                    │      → (B, K, latent_dim=128)               │
                    └─────────────┬───────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────────────────────┐
                    │  TemporalAggregator(nn.GRU,1 层)           │   stage-3.4 新增
                    │      → 取 h_K  (B, hidden=128)              │
                    └─────────────┬───────────────────────────────┘
                                  │
                                  ▼
                    z_temporal (B, 128) — 替换 stage-1 里的 "z[-1]"
                                  │
   depth_last (B,1,H,W)           │
        │                         │
        ▼                         │
   YopoBackbone(最后一帧)         │
   feature map (B,64,V,H) ◄───────┘
        │
        │ + obs (B,9,V,H)
        │ + broadcast(z_temporal) → (B,128,V,H)
        ▼
   concat → (B, head_in=201, V, H)
        │
        ▼
   YopoHead(1×1 conv) ───── 不动
        │
        ▼
   endstate (B,9,V,H) + score (B,V,H)
```

### 每一步的形状契约(B=16,K=10,V=3,H=5,图像 96×160)

| 张量 | 形状 | 说明 |
|---|---|---|
| `depth_seq`(输入) | (16, 10, 1, 96, 160) | 来自 `DynamicYOPOWrapper.__getitem__` |
| reVAE 用的 `depth_flat` | (160, 1, 96, 160) | `view(B*K, 1, H, W)` |
| reVAE 输出 `z_flat`, `mu_flat`, `logvar_flat` | 各 (160, 128) | `revae.encode` |
| `z_seq = z_flat.view(B, K, 128)` | (16, 10, 128) | reshape 回去 |
| GRU → `h_K` | (16, 128) | K 步展开后的最后一帧 hidden |
| `depth_last = depth_seq[:, -1]` | (16, 1, 96, 160) | 喂 `YopoBackbone` |
| backbone 输出 | (16, 64, 3, 5) | V×H 上的 `hidden_state` 通道 |
| `obs`(prepare_input 后) | (16, 9, 3, 5) | 不变 |
| `h_K` 广播到 V×H | (16, 128, 3, 5) | `h_K[:, :, None, None].expand(...)` |
| concat → head 输入 | (16, 201, 3, 5) | head_in = 64+9+128(跟 stage-1 一样) |
| `YopoHead` 输出 | (16, 10, 3, 5) | `endstate(9) + score(1)` |

**关键:** `head_in = 201` 跟 stage-1 字节一致,所以 `YopoHead` 的参数量
和初始化都不变。唯一不同的是那 128 通道 latent 块的**来源** — 从"单帧 reVAE z"
变成"K 帧 GRU 输出"。

### reVAE 重构损失在 K-frame 模式下

stage-1 重构输入那一帧;3.4 里我们继续用同一个损失,作用在**最后一帧** —
`recon_last = revae.decode(z_seq[:, -1])`。前面的帧编码但不解码 — 省 10×
decoder 计算,而且监督信号跟 stage-1 一致。

(备选方案:重构全部 K 帧。3.4 拒绝:多 10× decoder,而且对验证 temporal
forward 不必要。如果有动机,后续 phase 可以加一行 ablation 试。)

---

## 4. 代码改动点(touch points)

### 4.1 `policy/yopo_dataset.py` — `DynamicYOPOWrapper.__getitem__`

```diff
-        image = depth_seq[-1]              # (1, H, W)
+        # 🟦 stage-3.4: 返回完整 K 帧;per-anchor head 用最后一帧的
+        # V×H backbone 特征 + 所有 K 帧 GRU 聚合的 latent。
+        image_seq = depth_seq                 # (K, 1, H, W)
         ...
-        return image, pos, rot_wb, random_obs, map_idx, dyn_pad, dyn_mask
+        return image_seq, pos, rot_wb, random_obs, map_idx, dyn_pad, dyn_mask
```

**静态 `YOPODataset.__getitem__` 不动。** 静态路径保持单帧 `(1, H, W)`。
形状不对称由 trainer(4.3)在 batch 级处理:静态帧复制 K 次到 `(K, 1, H, W)`。

### 4.2 `policy/yopo_network.py` — `YopoNetwork`

- 构造函数加 `use_temporal: bool` + `temporal_hidden: int = 128`。
  当 True 时实例化 `nn.GRU(input_size=revae_latent, hidden_size=temporal_hidden, num_layers=1, batch_first=True)`。
- `forward` 签名加 `depth_seq`,行为变化:
  - 当 `use_temporal=False`(stage-3.1/3.2 回退):老的单帧路径原样跑 —
    depth 是 `(B, 1, H, W)`。
  - 当 `use_temporal=True`:期望 `(B, K, 1, H, W)`。reVAE 编 K 帧;
    GRU 聚合;最后一帧进 backbone;其他一切照 §3 流。
- DCA 侧通道(stage-3.2)是正交的:若 `use_dca=True` 且 `use_temporal=True`,
  temporal 聚合的 latent **先**进 concat,**后**进 DCA refinement,
  所以 DCA refine 一个带时间信息的 feature map。Phase 3 的 smoke test 6 验证。

### 4.3 `policy/yopo_trainer.py` — `train_one_epoch` + `forward_and_compute_loss`

- 读 `use_temporal = bool(cfg["frame_buffer"]["enable_temporal"])`
  (新 yaml 键,默认 False,老 run 继续工作)。
- 当 `use_temporal=True` 且 dyn_obs payload 是 None(静态 batch)时,
  把单帧 depth 复制成 (B, K, 1, H, W),用
  `depth.unsqueeze(1).expand(-1, K, -1, -1, -1)`。静态训练实际上把
  同一帧重复 K 次,所以 temporal 路径仍然在图上但收不到任何运动信号。
  这是"优雅的静态回退"。
- 当 `use_temporal=True` 且 dyn_obs payload 存在(动态 batch)时,
  wrapper 直接返回 (B, K, 1, H, W);不需要 shape 操作。

### 4.4 `config/traj_opt.yaml`

```diff
 frame_buffer:
-  K: 10                  # sliding window size (5~15 per PEMTRS); inference-side only in stage 1
+  K: 10                  # sliding window size; stage-3.4 temporal forward 消费这个
+  enable_temporal: false # 🟦 stage-3.4: YopoNetwork 是否走 K-frame forward
+  temporal_hidden: 128   # 🟦 stage-3.4: GRU hidden 大小;默认 == revae_latent
```

### 4.5 `policy/state_transform.py` 和 `YopoNetwork.inference`

- `inference()` 加 `depth_seq` 参数路径。部署时控制器喂最近 K 帧的
  环形缓冲区。stage-3.4 的 inference 测试只需要离线训练 shape;
  部署是 stage-5。

### 4.6 新模块:`policy/models/temporal_aggregator.py`

小包装:装 GRU + `forward(z_seq)` 返回 `(B, hidden)`。可以内联进
`YopoNetwork` 但单独一个文件让网络代码可读 + 给个干净的 unit-test 目标。

---

## 5. 开放设计问题 —— 推荐答案

每个问题有推荐,但允许覆盖。**所有 5 个推荐都经过文献调研验证**
(见 [stage_3_4_research.md](stage_3_4_research.md))。

### Q1. K-frame forward 下静态 batch 怎么处理?

**推荐:** 单帧复制 K 次(`expand`)。理由:
- 让 `cfg["dynamic_ratio"]` 仍然是混合采样的唯一开关。
- 静态路径仍然是 90% 的上游监督(trainer 的 step 时间被静态主导)。
- temporal aggregator 的 GRU 看到的是 K 步常量输入;
  输出有界且确定 — 不污染训练。
- 内存代价:16 × 10 × 1 × 96 × 160 × 4 字节 = 9.8 MB / batch 的 depth 张量,
  复制后用 **stride-0 expand**,所以 reVAE 消费之前实际是零额外内存
  (~98 MB 的 reVAE 输入张量 @ K=10)。可接受。

**文献支持:** 调研中发现 GRU 在常量输入下会指数收敛到一个稳定 fixed point
([Krishnamurthy et al. 2002.00025](https://arxiv.org/pdf/2002.00025)),
K=10 给 GRU 3-5 步"settling"远超需要。梯度跟"单帧 + 多走几步"等价。

**备选(暂时拒绝):** "静态 batch K=1,动态 batch K=10"。语义干净但要求
GRU 处理变长输入,要走 `pack_padded_sequence`,而且 TB 日志要分两路。
stage-3.4 不值得这么麻烦。

### Q2. `z_temporal` 在 head 图里落地的位置?

**推荐:** 跟 stage-1 的 `z_spatial` 同一个槽位 — 广播到 V×H,在 `YopoHead`
之前 concat。这是字面意义最小的改动,保持 `head_in = 201` 字节稳定。

**文献支持:** [ECCV 2020 Temporal Aggregate Representations](https://arxiv.org/pdf/2006.00830)
发现 concat 在**长程**视频理解上比 cross-attention coupling 差 7.5%,
但这是 10 分钟尺度;我们 K=10 @ 30Hz = 0.33 s 短程窗口,V×H = 15 个空间 anchor,
**broadcast-concat 是标准模式**(PEMTRS 和本代码 stage-1 都用)。
留下的性能空间被 V×H 的小空间范围约束着。

**备选(推迟):** 把 temporal 信号 concat 进 V×H 特征图后**之后**调制 backbone 特征。
需要 1×1 conv mixer 或者仔细的广播语义。如果需要可以做 stage-3.4.1 follow-up。

### Q3. 重构全部 K 帧还是只最后一帧?

**推荐:** 只最后一帧(跟 stage-1 监督一致)。

**文献支持:** PEMTRS Sec. III-A 只重构当前帧。Keras / DiffPhysDrone 都不重构
(分类 / RL)。没有反例。

### Q4. 推理时 GRU 需要 hidden 缓存吗?

**推荐:** stage-3.4 **不要**(只训练用)。部署时 sliding window 每帧重跑
完整 K 步 GRU;延迟由 K × reVAE encode 主导。如果这成为瓶颈,
stage-5 再加 stateful hidden cache。

**文献支持:** DiffPhysDrone(Nature MI 2025)用 **stateful** 单帧 GRU,
推理每帧只跑 1× CNN + 1× GRU。这是延迟最优,但需要管理 hidden 状态
(场景切换要 reset,启动时要对齐)。我们 v2 数据集是 K-frame baked,
天然适合 stateless;部署延迟优化留给 stage-5。

| 模式 | reVAE encode 次数 | GRU 步数 | 每帧 effective 计算 |
|---|---|---|---|
| Stateless K=10 | K(每次推理都重编全部 K 帧) | K | K × encode + K × GRU |
| Stateful K=1 | 1(只编最新帧) | 1 | 1 × encode + 1 × GRU |

stateful 每帧便宜 ~10×,**但**增加 state-management bug surface。
stage-5 部署时先 profile stateless K=10,如果 <10 ms 命中目标就不重构。

### Q5. 要在 temporal aggregator 里编码 dt 吗?

bake 存了 `dt_seq`(当前恒为 0.0333 s)。PEMTRS 在 Selector 里用 positional
encoding,但我们 bypass Selector。**推荐:stage-3.4 忽略 dt**(恒定帧率,
等时间间隔)。如果 stage-5 加变速推理再回来。

**文献支持:** DiffPhysDrone 固定控制率,不编码 dt;Keras CNN-RNN 假设
等间隔;ClimNet 用 positional encoding 但是给帧索引不是 dt。

---

## 6. stage-3.4 明确不实现的变体

| 变体 | 改什么 | 为什么推迟 |
|---|---|---|
| Option B — 多 waypoint GRU 输出 | 把 `YopoHead` 换成 `GRUDecoder` 输出多 waypoint。stage-3.1 每个损失都要按 waypoint 重写;kinodynamic 才真正激活 | 高风险,~2 周;demo paper 预算说不 |
| TemporalSelector ROI 预测 | `TemporalRegionSelector` 输出 `(B, future_horizon, 4)` ROIs;backbone 按 ROI 裁剪 | 多一步采样 + 多一项损失;当前单 camera FOV 让"下一步看哪里"信号差(verify 数据显示 ~70% off-FOV) |
| 重构全部 K 帧 | reVAE 解 K 次 | 10× decoder 代价,当前数据规模无明显收益 |
| Stateful GRU 推理 | 帧间 cache hidden | stage-5 部署的事 |
| Static→dynamic curriculum | 先训 K=1,再升 K=10 | 过早优化;只在 K=10 冷启失败时才有价值 |

---

## 7. 实施 phase(目标 3.5 天)

### Phase 1(第 1 天上午)— 模块 + unit test
1. 建 `policy/models/temporal_aggregator.py`,装 `TemporalAggregator(nn.Module)`。
2. 写 `scripts/smoke_stage3_4a.py`:
   - T1 reVAE 批量 encode:`(B*K, 1, H, W) → (B*K, latent)` reshape 往返。
   - T2 GRU 形状:`(B, K, latent) → (B, hidden)` 对 K=1 和 K=10。
   - T3 静态回退一致性:输入是 `(B, 1, latent).expand(-1, K, -1)` 时,
     GRU 输出有界(不发散)。
   - T4 梯度流:loss = output.sum(),backward;encoder + GRU 参数全部有非零 grad。

### Phase 2(第 1 天下午)— 网络接线
1. 扩 `YopoNetwork.__init__` 和 `forward` 接受 `depth_seq`,产出 §3 描述的
   temporal-z 路径。开关 `use_temporal`。
2. 写 `scripts/smoke_stage3_4b.py`:
   - T1 `use_temporal=False`:同 seed 下跟 stage-3.2 网络字节一致(有 stage-3.2
     ckpt 就 load,没有就比较两个 fresh net 同 seed init)。
   - T2 `use_temporal=True, K=1`:forward 消费 (B, 1, 1, H, W),输出 shape 匹配 stage-3.2。
     跟 stage-3.2 的数值漂移应在 GRU-init-相关的舍入量级,不会大。
   - T3 `use_temporal=True, K=10`:forward 消费 (B, 10, 1, H, W),输出 shape 匹配。
     无 NaN。
   - T4 参数计数:`use_temporal=True` 比 stage-3.2 恰好多 `nn.GRU(128,128,1).numel()`
     (约 99K 参数)。assert 保证。

### Phase 3(第 2 天)— trainer + dataset 接线
1. 改 `DynamicYOPOWrapper.__getitem__` 返回 `(K, 1, H, W)` 而不是 `(1, H, W)`。
2. 改 `YopoTrainer.train_one_epoch`,当 `use_temporal=True` 时把静态帧扩成 K-frame 复制。
3. 改 `forward_and_compute_loss` 透传 `depth_seq`。
4. 写 `scripts/smoke_stage3_4c.py`:
   - T1 cfg.frame_buffer.enable_temporal=True;trainer.policy.use_temporal=True
   - T2 一步静态 batch:dyn_loss/kino_loss == 0 EXACT(回归字节级 vs stage-3.1/3.2)。
   - T3 一步动态 batch:dyn_loss > 0。
   - T4 100 步混合训练:无 NaN;total loss 下降。
   - T5 batch shape 检查:两路 depth_seq 都是 (B, K, 1, H, W)。

### Phase 4(第 3 天)— v2 上 5k iter A/B + 扩展 ablation 表
1. 跑 `python scripts/run_stage4_ablation.py --steps 5000 --out
   results/ablation_stage_3_4.csv`,加两行:
   - **F temporal off**(当前 stage-3.2 完整配置 + v2 数据)— sanity
   - **G temporal on**(use_temporal=True,K=10,其余一致)
2. 比较 `dyn_dyn`、`stat_traj`、`total` 的 head→tail vs stage-4 E 行。
3. 更新 `docs/ARCHITECTURE.md`,加 "Stage-3.4" 一节。

### Phase 5(第 3.5 天)— commit、merge、tag
1. `stage-3.4-path-a` 分支 PR 到 main。
2. 打 tag `v0.3.4-temporal`。
3. README + paper outline §3.5/§5.4 填进去。

---

## 8. 风险 & 缓解

| 风险 | 缓解 |
|---|---|
| GRU 加 ~99K 参数但 50% 静态 batch 给不了有用梯度(退化的单帧输入) | §5 Q1 选了 expand-K-times;GRU 在静态上看常量输入学会忽略。如果看到静态 traj loss 回退,切到 "K=1 on static" |
| reVAE encode K=10 × batch=16 可能 OOM | phase 1 含一个 memory smoke;显存紧就降 batch 到 8(跟 stage-1 对齐)或者 reVAE.encoder 做 gradient checkpointing |
| temporal aggregator hidden=128 可能太小 | yaml 里好改;只要 hidden == revae_latent,head_in 兼容 |
| stage-3.2 DCA 接线在 `use_temporal=True` 下断掉 | phase 2 T2 smoke 覆盖 — DCA 仍然收 (B, V*H, head_in) token;latent 来源变了但契约没变 |
| K=10 下 dataloader I/O 成瓶颈(每个 item 多 10× PNG 解码) | v2 已经出现;FPS 比 stage-3.2 减半的话 `num_workers` 升到 4 |

---

## 9. 落盘清单

```
YOPO/policy/
  models/
    temporal_aggregator.py     # 新增(小,~50 LOC)
  yopo_network.py              # 改
  yopo_trainer.py              # 改
  yopo_dataset.py              # 改(只 DynamicYOPOWrapper.__getitem__)
YOPO/config/
  traj_opt.yaml                # frame_buffer 下 +3 键
scripts/
  smoke_stage3_4a.py           # 新:temporal aggregator 单元
  smoke_stage3_4b.py           # 新:网络 K-frame forward
  smoke_stage3_4c.py           # 新:trainer 端到端
docs/
  ARCHITECTURE.md              # +Stage-3.4 一节
  paper_outline.md             # 填 §3.5 + §5.4
results/
  ablation_stage_3_4.csv       # 新:temporal off vs on,5k iter
```

代码 LOC 估计:**+250 / -50**(一新模块,三改,三新 smoke)。
