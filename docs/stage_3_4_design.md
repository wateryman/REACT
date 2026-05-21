# Stage-3.4 Path-a Design — K-frame Anchor-Grid Forward

**Status:** Draft, awaiting review.  Sections § 1–5 define what we will
build; § 6 lists deferred variants we explicitly will *not* attempt in
stage-3.4.  Implementation phases are in § 7.

---

## 1. Goal & non-goals

### Goal

Replace the single-frame `depth` input of `YopoNetwork.forward` with a
K-frame depth sequence, **without touching the anchor-grid `YopoHead`
or any of the stage-3.1 losses**.  The network learns to encode
obstacle motion from the time dimension; the loss family that
supervised stage-3.1 (`motion_reshaped_collision_loss`,
`kinodynamic_loss`, plus trajectory + score + reVAE) is reused
byte-for-byte.

### Non-goals (explicitly deferred to future stages or never)

- Replacing `YopoHead` with `gru_decoder.GRUDecoder` (multi-waypoint
  output).  See § 6 "Option B" — different stage, different head,
  different loss routing.
- Activating `TemporalRegionSelector`'s future-horizon ROI prediction
  in the forward path.  We keep the module on disk but bypass it.
- End-to-end perception (depth → obstacle parameters).  Still consumes
  GT obstacle tokens via the existing `DynamicYOPOWrapper` path.
- Real-time inference re-shaping (TensorRT / ONNX export); that lands
  in stage-5.

### Success criteria

1. **Regression-clean fallback.**  With `cfg["frame_buffer"]["K"] = 1`
   the network is bit-identical to stage-3.1 (within fp32 numerical
   noise of `mean()` reductions across one dummy dim).
2. **At K=10, training runs end-to-end** on the v2 dataset with no
   NaN/Inf, batch_size=16 fits in available VRAM (we have ~12-16 GB
   to play with — confirm during phase 1).
3. **5 k-iter A/B vs stage-3.1 on v2**: temporal forward should drop
   `dyn_dyn` by ≥ 5 % relative to stage-3.1 baseline on the same data
   slice.  If gain is < 5 %, ship it anyway as the next ablation row
   ("temporal forward at this scale adds X %") — paper still benefits.

---

## 2. Existing pieces we can reuse

| Module | Where | Current status | What we use it for in 3.4 |
|---|---|---|---|
| `ReVAE` | [policy/models/revae.py](../YOPO/policy/models/revae.py) | Stage-1; encodes **one** frame at a time | Wrap to encode K frames in parallel via batch reshape |
| `YopoBackbone` (ResNet-18) | [policy/models/backbone.py](../YOPO/policy/models/backbone.py) | Stage-0; encodes one frame to a V×H feature map | Run on the **last** frame only (Option A keeps single-frame depth feature path) |
| `YopoHead` (1×1 conv) | [policy/models/head.py](../YOPO/policy/models/head.py) | Stage-0; takes `(B, head_in, V, H)` → `(B, 10, V, H)` | **Unchanged**.  This is the load-bearing decision of Option A |
| `TemporalRegionSelector` | [policy/models/temporal_selector.py](../YOPO/policy/models/temporal_selector.py) | Built stage-1 but **not on graph** | **Bypassed** in stage-3.4 (deferred to Option B / future) |
| `GRUDecoder` waypoint output | [policy/models/gru_decoder.py](../YOPO/policy/models/gru_decoder.py) | Built stage-1 but **not on graph** | **Bypassed**.  We instantiate `nn.GRU` directly (§ 3.3) and skip the multi-waypoint output heads |
| `DynamicCrossAttention` | [policy/models/dynamic_attention.py](../YOPO/policy/models/dynamic_attention.py) | Stage-3.2; side channel to anchor tokens | **Bypassed** in canonical 3.4 (dyn_obs tokens still flow, but DCA gate stays on `cfg["dynamic_attention"]["enable"]`; ablation can re-enable it on top) |
| `_load_dynamic` | [policy/yopo_dataset.py:153](../YOPO/policy/yopo_dataset.py) | Stage-2; **already** returns `(K, 1, H, W)` depth_seq + state_seq + dyn_obs + dt_seq | DynamicYOPOWrapper currently throws away K−1 frames at line 365.  Stage-3.4 stops throwing them away |
| `motion_reshaped_collision_loss` + `kinodynamic_loss` | [loss/](../YOPO/loss/) | Stage-3.1 | Unchanged.  Still operate on the predicted endstate from `YopoHead` |

---

## 3. New forward graph (canonical Option A)

```
                    ┌─────────────────────────────────────────────┐
        depth_seq   │                                             │
   (B, K, 1, H, W) ─┤  reVAE.encode (batched over B*K)            │   stage-3.4 NEW
                    │      → (B, K, latent_dim=128)               │
                    └─────────────┬───────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────────────────────┐
                    │  TemporalAggregator (nn.GRU, 1 layer)       │   stage-3.4 NEW
                    │      → take h_K (B, hidden=128)             │
                    └─────────────┬───────────────────────────────┘
                                  │
                                  ▼
                    z_temporal (B, 128) — replaces stage-1's "z[-1]"
                                  │
   depth_last (B,1,H,W)           │
        │                         │
        ▼                         │
   YopoBackbone(last frame)       │
   feature map (B,64,V,H) ◄───────┘
        │
        │ + obs (B,9,V,H)
        │ + broadcast(z_temporal) → (B,128,V,H)
        ▼
   concat → (B, head_in=201, V, H)
        │
        ▼
   YopoHead (1×1 conv) ───── unchanged
        │
        ▼
   endstate (B,9,V,H) + score (B,V,H)
```

### Shape contract per step (B=16, K=10, V=3, H=5, image 96×160)

| Tensor | Shape | Notes |
|---|---|---|
| `depth_seq` (input) | (16, 10, 1, 96, 160) | from DynamicYOPOWrapper.\_\_getitem\_\_ |
| `depth_flat` for reVAE | (160, 1, 96, 160) | `view(B*K, 1, H, W)` |
| reVAE latents `z_flat`, `mu_flat`, `logvar_flat` | (160, 128) each | `revae.encode` |
| `z_seq = z_flat.view(B, K, 128)` | (16, 10, 128) | reshape back |
| GRU → `h_K` | (16, 128) | last hidden of K-step rollout |
| `depth_last = depth_seq[:, -1]` | (16, 1, 96, 160) | for `YopoBackbone` |
| backbone output | (16, 64, 3, 5) | `hidden_state` channels at V×H |
| `obs` after prepare_input | (16, 9, 3, 5) | unchanged |
| broadcast `h_K` to V×H | (16, 128, 3, 5) | `h_K[:, :, None, None].expand(...)` |
| concat → head input | (16, 201, 3, 5) | head_in = 64+9+128 (same as stage-1) |
| `YopoHead` output | (16, 10, 3, 5) | `endstate(9) + score(1)` |

**Key:** `head_in = 201` is bit-identical to stage-1, so `YopoHead`'s
parameter count and init are unchanged.  Only the *source* of the
128-channel latent block is different (temporal GRU output vs.
single-frame reVAE z).

### reVAE reconstruction loss in K-frame mode

Stage-1 reconstructs the single input frame.  In 3.4 we keep the same
loss applied to **the last frame** of the sequence, using
`recon_last = revae.decode(z_seq[:, -1])`.  Earlier frames are
encoded but not reconstructed — saves the decoder pass and matches
the supervision signal of stage-1.

(Alternative: reconstruct all K frames.  Rejected for stage-3.4:
adds 10× decoder compute and is not needed to validate temporal
forward.  Add as ablation in a later phase if motivated.)

---

## 4. Touch points across the codebase

### 4.1 `policy/yopo_dataset.py` — `DynamicYOPOWrapper.__getitem__`

```diff
-        image = depth_seq[-1]              # (1, H, W)
+        # 🟦 stage-3.4: return full K-frame tensor; the per-anchor head
+        # consumes the last frame's V×H backbone features + GRU-aggregated
+        # latents from all K frames.
+        image_seq = depth_seq                 # (K, 1, H, W)
         ...
-        return image, pos, rot_wb, random_obs, map_idx, dyn_pad, dyn_mask
+        return image_seq, pos, rot_wb, random_obs, map_idx, dyn_pad, dyn_mask
```

**Static `YOPODataset.__getitem__` is NOT modified.**  Static path
stays single-frame `(1, H, W)`.  Trainer (4.3) handles the
shape-asymmetry by replicating static frames to `(K, 1, H, W)` at
batch level.

### 4.2 `policy/yopo_network.py` — `YopoNetwork`

- Constructor gains `use_temporal: bool` + `temporal_hidden: int = 128`.
  When True, instantiate `nn.GRU(input_size=revae_latent, hidden_size=temporal_hidden, num_layers=1, batch_first=True)`.
- `forward` signature gains `depth_seq` and changes behaviour:
  - When `use_temporal=False` (stage-3.1/3.2 fallback): old single-frame
    path runs unchanged — depth is `(B, 1, H, W)`.
  - When `use_temporal=True`: expects `(B, K, 1, H, W)`.  reVAE encodes
    all K frames; GRU aggregates; last frame goes to backbone; everything
    else flows as in § 3.
- DCA side channel (stage-3.2) is orthogonal: if both `use_dca=True` and
  `use_temporal=True`, the temporal-aggregated latent goes into the
  concat **before** DCA refinement, so DCA refines a temporally-aware
  feature map.  Smoke test 6 in § 7 phase 3 will verify.

### 4.3 `policy/yopo_trainer.py` — `train_one_epoch` + `forward_and_compute_loss`

- Read `use_temporal = bool(cfg["frame_buffer"]["enable_temporal"])`
  (new yaml key, defaults to False so old runs continue working).
- When `use_temporal=True` and dyn_obs payload is None (static batch),
  replicate the single-frame depth to (B, K, 1, H, W) via
  `depth.unsqueeze(1).expand(-1, K, -1, -1, -1)`.  Static training
  effectively re-uses the same frame K times so the temporal path stays
  on the graph but receives no motion signal.  This is the "graceful
  static fallback".
- When `use_temporal=True` and dyn_obs payload is present (dynamic
  batch), the wrapper returns `(B, K, 1, H, W)` directly; no shape
  manipulation needed.

### 4.4 `config/traj_opt.yaml`

```diff
 frame_buffer:
-  K: 10                  # sliding window size (5~15 per PEMTRS); inference-side only in stage 1
+  K: 10                  # sliding window size; consumed by stage-3.4 temporal forward
+  enable_temporal: false # 🟦 stage-3.4: turn on K-frame forward in YopoNetwork
+  temporal_hidden: 128   # 🟦 stage-3.4: GRU hidden size; defaults equal to revae_latent
```

### 4.5 `policy/state_transform.py` and `YopoNetwork.inference`

- `inference()` adds a `depth_seq` arg path.  At deployment time the
  controller feeds a circular buffer of the last K frames.  In stage-3.4
  the inference test only needs the offline-training shape; deployment
  is stage-5.

### 4.6 New module: `policy/models/temporal_aggregator.py`

Tiny wrapper: holds the GRU + a `forward(z_seq)` returning `(B, hidden)`.
Could be inlined in `YopoNetwork` but a separate module keeps the
network file readable and gives us a clean unit-test target.

---

## 5. Open design questions — recommended answers

Each Q has a recommendation but is open to user override.

### Q1. Static-batch handling under K-frame forward

**Recommendation:** Replicate the single static frame K times
(`expand`).  Rationale:
- Keeps `cfg["dynamic_ratio"]` as the sole mixed-sampling knob.
- Static path remains 90 % of the upstream supervision (the trainer
  step time budget is dominated by static).
- The temporal aggregator's GRU sees a constant input over K steps;
  output is bounded and deterministic — does not pollute training.
- Memory cost: 16 × 10 × 1 × 96 × 160 × 4 bytes = 9.8 MB per batch
  for the depth tensor; replicated tensor uses **stride-0 expand**
  so it's actually zero extra memory until reVAE consumes it (~98 MB
  of reVAE input tensors at K=10).  Acceptable.

Alternative (rejected for now): swap to "K=1 on static batches,
K=10 on dynamic batches".  Cleaner semantically but requires the
GRU to handle variable-length input, complicating reach-around for
TB logging.  Not worth it for stage-3.4.

### Q2. Where in the head graph does `z_temporal` land?

**Recommendation:** Same slot as stage-1's `z_spatial`, broadcast to
V×H, concatenated before `YopoHead`.  This is the literally-minimum
change that keeps `head_in = 201` bit-stable.

Alternative (deferred): concat into the V×H feature map **after** the
backbone instead of broadcasting — would let the temporal signal modulate
the backbone features.  Requires either a 1×1 conv mixer or careful
broadcast semantics; can ship as a stage-3.4.1 follow-up if needed.

### Q3. Reconstruct all K frames or just the last?

**Recommendation:** Last frame only (matches stage-1 supervision).

### Q4. Does the GRU need a hidden-state cache for inference?

**Recommendation:** **No** for stage-3.4 (training-only).  In
deployment we'll feed a sliding window every frame and re-run the
full GRU on K timesteps; latency is dominated by reVAE encode × K
anyway.  If that becomes the bottleneck we'll add a stateful hidden
cache in stage-5.

### Q5. Should we encode the time dt at all?

The bake stores `dt_seq` (currently constant 0.0333 s).  PEMTRS uses
positional encoding inside the Selector, but we're skipping the
Selector.  Recommendation: **ignore dt for stage-3.4** (constant
frame rate, equal time spacing).  Revisit if we add variable-rate
inference in stage-5.

---

## 6. Variants we explicitly do NOT implement in 3.4

| Variant | What it would change | Why deferred |
|---|---|---|
| Option B — multi-waypoint GRU output | Replace `YopoHead` with `GRUDecoder` waypoint heads.  Forces every stage-3.1 loss to be reformulated per-waypoint; kinodynamic loss finally activates | High risk, ~2 weeks; demo-paper-budget says no |
| Temporal selector ROI prediction | `TemporalRegionSelector` outputs `(B, future_horizon, 4)` ROIs; backbone is then cropped to those ROIs | Adds a sampling step + extra loss term; current single-camera FOV makes "where to look next" a poor signal (verify_dynamic_render shows ~70% off-FOV anyway) |
| Reconstruct all K frames | reVAE decodes K times | 10× decoder cost, no clear payoff at current dataset scale |
| Stateful GRU at inference | Cache hidden between frames | Stage-5 deployment concern |
| Static→dynamic curriculum | Train K=1 first, then bump K to 10 | Premature optimisation; only adds value if K=10 cold-start fails |

---

## 7. Implementation phases (3.5 days target)

### Phase 1 (day 1, morning) — module + unit test
1. Create `policy/models/temporal_aggregator.py` with `TemporalAggregator(nn.Module)`.
2. Write `scripts/smoke_stage3_4a.py` covering:
   - T1 reVAE batched encode: `(B*K, 1, H, W) → (B*K, latent)` reshape round-trip.
   - T2 GRU shape: `(B, K, latent) → (B, hidden)` for K=1, K=10.
   - T3 Static-fallback parity: when input is `(B, 1, latent).expand(-1, K, -1)`, GRU output is bounded (no divergence).
   - T4 Gradient flow: loss = output.sum(), backward; all encoder + GRU params have non-zero grad.

### Phase 2 (day 1, afternoon) — network wiring
1. Extend `YopoNetwork.__init__` and `forward` to accept `depth_seq` and
   produce the temporal-z path described in § 3.  Toggle on
   `use_temporal`.
2. Write `scripts/smoke_stage3_4b.py` covering:
   - T1 `use_temporal=False`: byte-identical to stage-3.2 network on
     a fixed seed (load the stage-3.2 ckpt if available, else compare
     two fresh nets with same seed at init).
   - T2 `use_temporal=True, K=1`: forward consumes (B, 1, 1, H, W),
     output shape matches stage-3.2.  Numerical drift vs stage-3.2
     should be on the order of GRU-init-related rounding, not large.
   - T3 `use_temporal=True, K=10`: forward consumes (B, 10, 1, H, W),
     output shapes match.  No NaN.
   - T4 Param count: `use_temporal=True` adds exactly
     `nn.GRU(128,128,1).numel()` over stage-3.2 (≈99 K params).
     Asserted.

### Phase 3 (day 2) — trainer + dataset wiring
1. Modify `DynamicYOPOWrapper.__getitem__` to return `(K, 1, H, W)`
   instead of `(1, H, W)`.
2. Modify `YopoTrainer.train_one_epoch` to expand static frames to
   K-frame replicas when `use_temporal=True`.
3. Modify `forward_and_compute_loss` to pass `depth_seq` through.
4. Write `scripts/smoke_stage3_4c.py` covering:
   - T1 cfg.frame_buffer.enable_temporal=True; trainer.policy.use_temporal=True
   - T2 one static-batch step: dyn_loss/kino_loss == 0 EXACT (regression
     byte-clean against stage-3.1/3.2).
   - T3 one dynamic-batch step: dyn_loss > 0.
   - T4 100-step mixed training: no NaN; total loss decreasing.
   - T5 batch shape inspection: depth_seq is (B, K, 1, H, W) on both
     paths.

### Phase 4 (day 3) — 5 k iter A/B on v2 + ablation table extension
1. Run `python scripts/run_stage4_ablation.py --steps 5000 --out
   results/ablation_stage_3_4.csv` with two new rows:
   - **F temporal off** (current stage-3.2 full config on v2) — sanity
   - **G temporal on** (use_temporal=True, K=10, everything else same)
2. Compare `dyn_dyn`, `stat_traj`, `total` head→tail vs stage-4's E row.
3. Update `docs/ARCHITECTURE.md` with a "Stage-3.4" section.

### Phase 5 (day 3.5) — commit, merge, tag
1. PR to main from `stage-3.4-path-a` branch.
2. Tag `v0.3.4-temporal`.
3. README + paper outline § 3.5/§ 5.4 filled in.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| GRU adds ~99 K params but the 50 % static batches give it no useful gradient (degenerate static-frame input) | Section 5 Q1 chose expand-static-K-times; the GRU sees constant input on static and learns to ignore it.  If we see static traj loss regress, switch to "K=1 on static" |
| reVAE encode at K=10 × batch=16 may OOM the GPU | Phase 1 includes a memory smoke; if VRAM is tight, drop batch to 8 (matches stage-1) or use gradient checkpointing on reVAE.encoder |
| Temporal aggregator hidden=128 might be too small | Easy to bump in yaml; head_in stays compatible as long as we keep hidden == revae_latent |
| Stage-3.2 DCA wiring breaks when `use_temporal=True` | Phase 2 T2 smoke covers this — DCA still receives (B, V*H, head_in) tokens; the source of the latent block changed but the contract didn't |
| Dataloader I/O becomes the bottleneck at K=10 (10x as much PNG decode per item) | Already happening at v2 — bump `num_workers` to 4 if FPS halves vs stage-3.2 |

---

## 9. What lands on disk

```
YOPO/policy/
  models/
    temporal_aggregator.py     # NEW (small, ~50 LOC)
  yopo_network.py              # modified
  yopo_trainer.py              # modified
  yopo_dataset.py              # modified (DynamicYOPOWrapper.__getitem__ only)
YOPO/config/
  traj_opt.yaml                # +3 keys under frame_buffer
scripts/
  smoke_stage3_4a.py           # NEW: temporal aggregator unit
  smoke_stage3_4b.py           # NEW: network K-frame forward
  smoke_stage3_4c.py           # NEW: trainer end-to-end
docs/
  ARCHITECTURE.md              # +Stage-3.4 section
  paper_outline.md             # fill §3.5 + §5.4
results/
  ablation_stage_3_4.csv       # NEW: temporal off vs on, 5k iter
```

Total LOC delta estimate: **+250 / -50** (one new module, three
modified, three new smoke scripts).
