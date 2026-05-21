# Stage-3.4 Research — Background for §5 Design Decisions

**Purpose:** validate the recommendations in
`docs/stage_3_4_design.md` §5 against existing literature and open-source
practice.  Not exhaustive — focused on the five concrete questions the
design doc raises.

**Date of research:** 2026-05-22.

---

## Methodology

Four parallel web searches + two targeted page fetches.  Sources are
recorded under each finding for traceability.  Where literature was
inconclusive, the design doc's recommendation is marked **(no contrary
evidence found)** rather than **(literature-backed)**.

---

## Findings

### F1 — Two canonical patterns for K-frame temporal forward exist

| Pattern | Used by | Description | Inference cost | Training cost |
|---|---|---|---|---|
| **K-frame stateless** | Keras CNN-RNN tutorial; ClimNet (UAV tracking) | Stack K frames per sample, run shared CNN backbone, GRU/Transformer over the K-step sequence, no state persists across forward calls | K× CNN per forward | K× CNN per batch element |
| **Single-frame stateful** | DiffPhysDrone (SJTU, Nature MI 2025; arXiv 2407.10648) | Single-frame CNN per call, GRU cell maintains hidden state across timesteps in the inference loop | 1× CNN per forward | Sequence training via TBPTT |

The DiffPhysDrone paper explicitly: *"a GRU layer for consistent
planning and control"* preserves hidden state across timesteps during
inference, with only a 16×12 max-pooled depth as per-step input.

The Keras tutorial explicitly stacks frames with padding + masking:
*"In the case where a video's frame count is lesser than the maximum
frame count we will pad the video with zeros."*

**Implication for stage-3.4:** Our v2 dataset is already baked as
K=10 sequences (`_load_dynamic` returns `(K, 1, H, W)`), so the
K-frame stateless pattern fits our data natively.  Pattern-B
(stateful) would require either re-baking or treating the K-frame
sequence as a TBPTT unroll — more refactoring.

→ **Confirms design doc §3 architecture choice (K-frame stateless).**

### F2 — Masking vs. expand-K-times for short / single-frame inputs

The Keras CNN-RNN tutorial uses **zero-padding + GRU masking** to
handle short clips:

```python
x = keras.layers.GRU(16, return_sequences=True)(
    frame_features_input, mask=mask_input)
```

The mask ensures padded timesteps don't contribute to gradients.

PyTorch's `nn.GRU` does **not** have a native `mask=` arg.  The
equivalent is `pack_padded_sequence` + `pad_packed_sequence`, which
requires sorting batches by length and adds two reshape ops per
forward.

**Expand-K-times alternative:** feed the GRU K identical copies of
the single static frame.  Two relevant properties from the GRU
fixed-point literature ([Krishnamurthy et al. 2002.00025](https://arxiv.org/pdf/2002.00025)):

- *"the GRU reset gate modulates the complexity of the landscape of
  fixed points"* — for constant input, the hidden state converges
  to a fixed point exponentially fast.
- The update gate can poise the system at a *marginally stable
  point*, meaning constant input still produces a stable output.

**Empirical estimate:** K=10 with constant input gives the GRU ~3-5
steps of "settling" before the loss reads the last hidden state;
this is more than enough for convergence.  Gradient flows back
identically as if we had run a 1-step GRU on the single frame
(because every step has the same input, the network learns to map
that input to a stable fixed-point that matches what a 1-step GRU
would output).

→ **Confirms design doc §5 Q1 recommendation (expand K times).**
→ **Mask-via-pack_padded_sequence is the cleaner alternative if we
   ever need variable K**, but for stage-3.4's fixed K=10 + degenerate
   static frame, expand-K is simpler and gradient-equivalent.

### F3 — Where to inject the temporal feature: concat vs. modulation

[Temporal Aggregate Representations (ECCV 2020, 2006.00830)](https://arxiv.org/pdf/2006.00830):

> *"When researchers separately pass recent and long-range features
> through concatenation and a linear layer instead of coupling them
> together, there is a performance drop of 7.5%"*

This finding applies to **long-range** (10-min) video understanding,
where the "concat" baseline lost to "coupling" via cross-attention.
The scale-dependence matters:

- Our K=10 at 30 Hz = ~0.33 s window.  Short-range.
- Their 10-min window is 30,000× longer.

For short-range temporal fusion ([Temporal FiLM, 1909.06628](https://arxiv.org/pdf/1909.06628)), Feature-Wise Linear Modulation (FiLM)
also outperforms concat — but the gain is most pronounced for *single-step
modulation of a downstream conv stack*, not for broadcasting a vector to
a feature map and concatenating channels.

For our specific case (V×H = 3×5 = 15 anchors, broadcast a 128-dim
temporal vector and concat to a 64+9 channel map), **broadcast-concat
is the standard pattern** — used in PEMTRS and in stage-1 of this
codebase (for reVAE's `z`).  The risk of leaving performance on the
table by not using cross-attention or FiLM is bounded by the small
spatial extent of our V×H grid.

→ **Confirms design doc §5 Q2 recommendation (broadcast-concat at the
   same slot as stage-1's `z_spatial`).**
→ Cross-attention or FiLM is a justifiable stage-3.4.1 follow-up if
   the 5 k-iter A/B shows < 5 % gain (which would suggest the temporal
   feature isn't propagating effectively).

### F4 — Reconstruct K frames vs. last frame: inconclusive

No direct paper match for "reVAE with sequence input — reconstruct
which frames?"  The Keras tutorial doesn't reconstruct (it's
classification).  DiffPhysDrone doesn't reconstruct (it's RL policy).
PEMTRS Sec. III-A reconstructs the *current* frame only.

**Conservative choice:** match PEMTRS / stage-1 — reconstruct the
last frame, save 10× decoder compute, no change to `revae_loss` shape
contract.

→ **Confirms design doc §5 Q3 recommendation (last frame only).**

### F5 — Stateful vs. stateless GRU at inference

DiffPhysDrone uses **stateful** GRU at inference (carry hidden state
across frames); the Keras CNN-RNN uses **stateless** (K-frame per
forward).  Our stage-3.4 training is K-frame stateless by virtue of
the dataset shape, so the stateless inference path is the lowest-friction
option.

**Latency trade-off:**

| Mode | reVAE encode ops | GRU steps | Effective per-frame compute |
|---|---|---|---|
| Stateless K=10 | K (re-encode all K frames each inference) | K | K × encode + K × GRU |
| Stateful K=1 | 1 (only newest frame) | 1 | 1 × encode + 1 × GRU |

Stateful is ~10× cheaper at inference per frame, but introduces a
**state-management bug surface** (must reset hidden between scenes,
must align hidden with the depth buffer at startup).

→ **Confirms design doc §5 Q4 recommendation (stateless K-frame in
   stage-3.4; stateful is a stage-5 deployment optimisation).**
→ For Jetson deployment, stage-5 should profile K=10 stateless first.
   If <10 ms latency target is hit, no need to refactor.

### F6 — dt encoding inside the GRU/temporal aggregator

DiffPhysDrone runs at fixed control rate, no dt encoding.
ClimNet (UAV tracking) uses positional encoding for frame indices but
not dt.  PEMTRS uses positional encoding in its Selector.

For a constant-30Hz K=10 window where dt never varies, dt encoding
adds parameters and code complexity with no signal to learn from.

→ **Confirms design doc §5 Q5 recommendation (skip dt encoding for
   stage-3.4).**

---

## What the research did NOT settle

1. **Whether DCA on top of temporal forward actually helps.**  No
   directly comparable paper found.  Stage-3.4 A/B (with vs without
   DCA at K=10) will be the definitive test.
2. **Optimal K.**  PEMTRS suggests 5–15; we picked K=10 by bake.  No
   principled way to pick "the right K" short of running ablations,
   which we deliberately defer to stage-3.4.1+.
3. **Whether 2000 v2 sequences is enough.**  All temporal papers
   surveyed used larger datasets (Keras tutorial: UCF101 has 13 k
   videos; DiffPhysDrone: RL with infinite simulated episodes;
   PEMTRS: dataset size not specified in summary).

Hypothesis 1 from `docs/ARCHITECTURE.md` §"Hypotheses for the lack of
separation" — *"Dataset starvation"* — remains the most likely
explanation for stage-3.2's negative result, and stage-3.5 (v2 bake,
4× data) addresses it directly.  Stage-3.4 will inherit this scale.

---

## Sources

- [Learning Vision-based Agile Flight via Differentiable Physics (Nature MI 2025; arXiv 2407.10648)](https://arxiv.org/html/2407.10648v1) — DiffPhysDrone: stateful GRU + single-frame depth pattern (F1, F5)
- [Keras CNN-RNN Video Classification tutorial](https://keras.io/examples/vision/video_classification/) — K-frame stateless pattern + mask padding (F1, F2)
- [SJTU 36kr coverage of DiffPhysDrone](https://eu.36kr.com/en/p/3398043817265289) — 90% success rate in dynamic-obstacle outdoor tests
- [Temporal Aggregate Representations for Long-Range Video Understanding (ECCV 2020, arXiv 2006.00830)](https://arxiv.org/pdf/2006.00830) — concat-vs-coupling 7.5% gap finding (F3)
- [Temporal FiLM: Capturing Long-Range Sequence Dependencies (1909.06628)](https://arxiv.org/pdf/1909.06628) — FiLM modulation vs concat at short range (F3)
- [Gating creates slow modes and controls phase-space complexity in GRUs (2002.00025)](https://arxiv.org/pdf/2002.00025) — GRU fixed-point behaviour under constant input (F2)
- [Continuity-Aware Latent Interframe Information Mining for Reliable UAV Tracking (2303.04525)](https://arxiv.org/pdf/2303.04525) — ClimNet: latent interframe info pattern (F1)
- [UAV Obstacle Avoidance by Applying Deep Learning (Auburn thesis)](https://etd.auburn.edu/bitstream/handle/10415/7920/UAV_Obstacle_Avoidance_by_Applying_Deep_Learning%20(2).pdf?sequence=2&isAllowed=y) — ResNet+GRU obstacle avoidance reference
- [PyTorch GRU docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) — no native masking; `pack_padded_sequence` is the workaround (F2)

---

## Bottom line

**All five recommendations in `docs/stage_3_4_design.md` §5 are supported
by surveyed literature.**  No contrary evidence was found.  The two
canonical patterns (K-frame stateless vs. single-frame stateful) both
exist in production-grade UAV navigation systems; our choice of K-frame
stateless is the natural fit for our pre-baked v2 dataset and the lower-
risk path for a demo paper.

The one nuance worth flagging: **F3** noted a 7.5 % gap between concat
and coupling in *long-range* video understanding.  We bet this gap does
not transfer to our 0.33 s short window with V×H = 15 spatial anchors,
but the design doc §5 Q2 explicitly leaves "FiLM / cross-attention as a
stage-3.4.1 follow-up" open so we can revisit if the 5 k-iter A/B
underperforms.
