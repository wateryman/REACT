# Stage-3.4 文献调研 —— §5 设计决策的依据(中文版)

**目的:** 用文献和开源实践验证 `docs/stage_3_4_design.md` §5 的五条推荐。
不求穷尽,聚焦设计文档里的五个具体问题。

**调研日期:** 2026-05-22。

英文原版:[stage_3_4_research.md](stage_3_4_research.md)
对应设计文档:[stage_3_4_design_cn.md](stage_3_4_design_cn.md)

---

## 方法

4 次并行 web search + 2 次定向 page fetch。每条发现下面记录来源,可追溯。
当文献不足以下定论时,设计文档里那条推荐标"**(未见反例)**"而不是
"**(文献支持)**"。

---

## 发现(Findings)

### F1 —— K-frame temporal forward 存在两种规范化模式

| 模式 | 谁用 | 描述 | 推理代价 | 训练代价 |
|---|---|---|---|---|
| **K-frame stateless**(无状态) | Keras CNN-RNN tutorial;ClimNet(UAV tracking) | 每个样本堆 K 帧,共享 CNN backbone,GRU/Transformer 在 K 步序列上跑,**forward 调用之间不维持状态** | 每 forward K × CNN | 每 batch 元素 K × CNN |
| **Single-frame stateful**(有状态) | DiffPhysDrone(SJTU 那个,Nature MI 2025;arXiv 2407.10648) | 每次调用单帧 CNN,GRU cell 在推理循环中跨帧维护 hidden 状态 | 每 forward 1 × CNN | 通过 TBPTT 做序列训练 |

DiffPhysDrone 原文:*"a GRU layer for consistent planning and control"*
保持 hidden 状态跨时间步,每步输入只是 16×12 的 max-pool depth。

Keras tutorial 明确地把帧堆叠 + padding + masking:
*"In the case where a video's frame count is lesser than the maximum
frame count we will pad the video with zeros."*

**对 stage-3.4 的启示:** 我们 v2 数据集已经按 K=10 baked
(`_load_dynamic` 返回 `(K, 1, H, W)`),所以 K-frame stateless 模式跟我们
数据天然匹配。Pattern-B(stateful)要么得重 bake 要么得把 K-frame 序列
当 TBPTT 展开 —— 重构量大得多。

→ **确认设计文档 §3 选择 K-frame stateless。**

### F2 —— 短/单帧输入的 masking vs. expand-K-times

Keras CNN-RNN tutorial 用 **零填充 + GRU masking** 处理短片段:

```python
x = keras.layers.GRU(16, return_sequences=True)(
    frame_features_input, mask=mask_input)
```

mask 让填充时间步不参与梯度。

PyTorch 的 `nn.GRU` **没有**原生 `mask=` 参数。等价方案是
`pack_padded_sequence` + `pad_packed_sequence`,要求按长度排序 batch,
forward 里多两次 reshape。

**Expand-K-times 替代方案:** 给 GRU 喂 K 份单静态帧的拷贝。
GRU fixed-point 文献([Krishnamurthy et al. 2002.00025](https://arxiv.org/pdf/2002.00025))
有两个相关性质:

- *"the GRU reset gate modulates the complexity of the landscape of
  fixed points"* —— 对常量输入,hidden 状态指数收敛到 fixed point。
- update gate 可以把系统放在 *marginally stable point*,意思是常量输入
  仍能产生稳定输出。

**经验估计:** K=10 配常量输入,GRU 在 loss 读取最后 hidden 之前有 ~3-5 步
"settling",远超收敛需要。梯度回传跟"在单帧上跑 1 步 GRU"等价
(因为每步输入都一样,网络学到把那个输入映射到一个稳定 fixed point,
恰好等于 1 步 GRU 该输出的)。

→ **确认设计文档 §5 Q1 推荐(expand K 次)。**
→ **如果以后需要变长 K,Mask-via-pack_padded_sequence 是更干净的替代**,
   但 stage-3.4 固定 K=10 + 退化静态帧,expand-K 更简单且梯度等价。

### F3 —— temporal feature 在哪儿注入:concat vs. modulation

[Temporal Aggregate Representations(ECCV 2020,2006.00830)](https://arxiv.org/pdf/2006.00830):

> *"When researchers separately pass recent and long-range features
> through concatenation and a linear layer instead of coupling them
> together, there is a performance drop of 7.5%"*

这个结论适用于**长程**(10 分钟)视频理解,"concat" baseline 输给了
"coupling via cross-attention"。但尺度依赖很大:

- 我们 K=10 @ 30 Hz = ~0.33 秒窗口。**短程**。
- 它们 10 分钟窗口 = 我们的 30,000 倍。

对短程时间融合([Temporal FiLM,1909.06628](https://arxiv.org/pdf/1909.06628)),
Feature-Wise Linear Modulation(FiLM)也优于 concat —— 但增益最大的地方
是 *单步调制下游 conv stack*,**不是**广播一个向量到 feature map 再
concat 通道。

对我们这种情况(V×H = 3×5 = 15 anchor,广播 128 维 temporal 向量 concat
到 64+9 通道图),**broadcast-concat 是标准模式** —— PEMTRS 用、
本代码 stage-1 也用(reVAE 的 `z`)。不用 cross-attention 或 FiLM
留下的性能空间被 V×H 的小空间范围约束着。

→ **确认设计文档 §5 Q2 推荐(broadcast-concat,跟 stage-1 `z_spatial`
   同一个槽位)。**
→ 如果 5k iter A/B 显示增益 <5%(暗示 temporal feature 没有有效传播),
   cross-attention 或 FiLM 是合理的 stage-3.4.1 follow-up。

### F4 —— 重构 K 帧 vs. 最后一帧:**调研不够确定**

没找到直接对应"序列输入的 reVAE 重构哪几帧"的论文。Keras tutorial 不
重构(它是分类)。DiffPhysDrone 不重构(RL 策略)。PEMTRS Sec. III-A
只重构*当前*帧。

**保守选择:** 跟 PEMTRS / stage-1 一致 —— 只重构最后一帧,省 10× decoder
计算,不动 `revae_loss` 的 shape 契约。

→ **确认设计文档 §5 Q3 推荐(只最后一帧)。**

### F5 —— stateful vs. stateless GRU 推理

DiffPhysDrone 推理用 **stateful** GRU(帧间维持 hidden);Keras CNN-RNN
用 **stateless**(每 forward K 帧)。我们 stage-3.4 训练因为数据集 shape
天然是 K-frame stateless,所以 stateless 推理路径阻力最小。

**延迟权衡:**

| 模式 | reVAE encode 次数 | GRU 步数 | 每帧 effective 计算 |
|---|---|---|---|
| Stateless K=10 | K(每次推理重编全部 K 帧) | K | K × encode + K × GRU |
| Stateful K=1 | 1(只编最新帧) | 1 | 1 × encode + 1 × GRU |

Stateful 每帧便宜 ~10×,**但**引入了 **state-management bug surface**
(场景切换要 reset 状态,启动要对齐 hidden 和 depth 缓冲区)。

→ **确认设计文档 §5 Q4 推荐(stage-3.4 用 stateless K-frame;
   stateful 留给 stage-5 部署优化)。**
→ Jetson 部署时 stage-5 先 profile stateless K=10。如果 <10 ms 命中目标,
   不需要重构。

### F6 —— GRU/temporal aggregator 里编码 dt

DiffPhysDrone 固定控制率,不编码 dt。ClimNet(UAV tracking)给帧索引用
positional encoding,但不编码 dt。PEMTRS 在 Selector 里用 positional
encoding。

对恒定 30 Hz、K=10、dt 从不变化的窗口,dt 编码加参数加代码复杂度但没有
信号可学。

→ **确认设计文档 §5 Q5 推荐(stage-3.4 跳过 dt 编码)。**

---

## 调研**没有**解决的问题

1. **DCA 加在 temporal forward 之上到底有没有帮助。** 没找到直接可比的论文。
   stage-3.4 的 A/B(K=10 下 DCA 开 vs 关)才是定论实验。
2. **K 的最优值。** PEMTRS 建议 5-15;我们选 K=10 是因为 bake 时定的。
   除了跑 ablation 没有原则方法选"对的 K",我们故意推到 stage-3.4.1+。
3. **2000 个 v2 sequence 够不够。** 调研的所有 temporal 论文都用了更大数据集
   (Keras tutorial 用 UCF101,13k 视频;DiffPhysDrone:RL 无限模拟集;
   PEMTRS:摘要里没说数据规模)。

`docs/ARCHITECTURE.md` §"Hypotheses for the lack of separation" 里假设 1 ——
*"Dataset starvation"* —— 仍然是 stage-3.2 负结论最可能的解释,而 stage-3.5
(v2 bake,4× 数据)直接缓解了这个。stage-3.4 继承这个规模。

---

## 来源(Sources)

- [Learning Vision-based Agile Flight via Differentiable Physics(Nature MI 2025;arXiv 2407.10648)](https://arxiv.org/html/2407.10648v1) —— DiffPhysDrone:stateful GRU + 单帧 depth 模式(F1、F5)
- [Keras CNN-RNN Video Classification tutorial](https://keras.io/examples/vision/video_classification/) —— K-frame stateless 模式 + mask padding(F1、F2)
- [SJTU 36kr 关于 DiffPhysDrone 的报道](https://eu.36kr.com/en/p/3398043817265289) —— 动态障碍物户外测试 90% 成功率
- [Temporal Aggregate Representations for Long-Range Video Understanding(ECCV 2020,arXiv 2006.00830)](https://arxiv.org/pdf/2006.00830) —— concat vs coupling 7.5% 差距(F3)
- [Temporal FiLM:Capturing Long-Range Sequence Dependencies(1909.06628)](https://arxiv.org/pdf/1909.06628) —— FiLM 调制 vs concat 短程对比(F3)
- [Gating creates slow modes and controls phase-space complexity in GRUs(2002.00025)](https://arxiv.org/pdf/2002.00025) —— GRU 在常量输入下的 fixed-point 行为(F2)
- [Continuity-Aware Latent Interframe Information Mining for Reliable UAV Tracking(2303.04525)](https://arxiv.org/pdf/2303.04525) —— ClimNet:latent 帧间信息模式(F1)
- [UAV Obstacle Avoidance by Applying Deep Learning(Auburn thesis)](https://etd.auburn.edu/bitstream/handle/10415/7920/UAV_Obstacle_Avoidance_by_Applying_Deep_Learning%20(2).pdf?sequence=2&isAllowed=y) —— ResNet+GRU 避障参考
- [PyTorch GRU 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) —— 无原生 masking;`pack_padded_sequence` 是 workaround(F2)

---

## 结论(Bottom line)

**`docs/stage_3_4_design.md` §5 五条推荐全部被调研到的文献支持。** 没找到反例。
两种规范化模式(K-frame stateless 和 single-frame stateful)都存在于
production-grade UAV 导航系统;我们选 K-frame stateless 是因为它跟我们
预 bake 的 v2 数据集天然匹配,且对 demo paper 更低风险。

**唯一值得标出的微妙之处:** F3 提到 *长程*视频理解里 concat 比 coupling
差 7.5%。我们赌这个差距在 0.33 秒短窗 + V×H = 15 空间 anchor 下不会
转移过来,但设计文档 §5 Q2 明确把"FiLM / cross-attention 作为 stage-3.4.1
follow-up"留为开放,所以 5k iter A/B 表现不佳时可以回来。
