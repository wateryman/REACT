# REACT Paper Outline (TBD — first draft, 2026-05-21)

**Status:** Draft.  Sections, claims, and experiment list are all still open
for revision.  Treat the numbers in §5 as placeholders until stage-3.4 and
stage-5 produce final results.

---

## Working Title

REACT: Recurrent Environment-Aware Collision-avoidance Trajectory
Planning for Agile UAV Flight in Dynamic Environments

(Alternatives: drop "REACT" if the YOPO trademark is confusing; the
network code already lives under `YOPO/` for upstream-diff readability —
see `docs/ARCHITECTURE.md` §1.)

---

## 1. Introduction

- Agile UAV planning in dynamic environments has hard latency constraints
  (<10 ms on Jetson) and must handle obstacles that move during flight.
- Existing learning planners split into two camps:
  - Slow optimization-based (MPC, ESDF-time) — accurate but >> 10 ms.
  - Single-frame one-stage (e.g. YOPO) — fast but blind to motion.
- **Contribution.** Three additive pieces over the YOPO single-frame
  baseline:
  1. Dynamic-aware loss family (`motion_reshaped_collision_loss` +
     `kinodynamic_loss`).
  2. PEMTRS-inspired temporal forward (reVAE → Selector → GRU decoder).
  3. D-3 CUDA-baked dynamic dataset + 端-to-end success rate ≥85 %
     in dynamic scenarios.

---

## 2. Related Work

- **YOPO** [TJU-Aerial-Robotics, 2024] — single-stage anchor-grid baseline
  this work builds on.
- **PEMTRS** [RA-L 2026, Nankai] — temporal architecture (reVAE +
  Transformer selector + GRU decoder); reimplemented from the published
  method section as source code is not public.
- Other learning planners: EgoPlanner, Agile-Autonomy, NeuPAN, …
- Dynamic obstacle handling in classical planning: MPC over predicted
  trajectories, ESDF-time field methods.

---

## 3. Method

### 3.1 Background — YOPO single-stage anchor planner

Recap of V×H = 3×5 anchor grid, per-anchor endstate regression
(`Smooth-L1` on pos/vel/acc), score head with softplus, choice of
trajectory at inference time.

### 3.2 reVAE auxiliary encoder (stage-1)

ResidualVAE encoder over the current depth frame; latent z (128) is
broadcast onto the V×H feature map before the YopoHead 1×1 conv.
Loss: `lam_recon · MSE + lam_kl · KL`, with `lam_kl = 0.001` to avoid
posterior collapse.

### 3.3 Dynamic-aware losses (stage-3.1, "path c")

- **`motion_reshaped_collision_loss`** —
  `softplus(d_safe − dist − α · closing_speed)`, where
  `closing_speed = relu(rel_v · dir_from_obs_to_traj)`.  Penalises
  trajectories that approach a moving obstacle along its velocity
  vector more than ones that pass tangentially.
- **`kinodynamic_loss`** — squared hinge on
  `|v| − v_max`, `|a| − a_max`, `|j| − j_max`.

### 3.4 DCA side channel (stage-3.2, "path b", ablation only)

`DynObsEncoder` MLP (7 → 201) projects per-obstacle rows
`[rel_pos, abs_vel, radius]` to feature-channel size; cross-attention
against the V×H anchor tokens.  **Negative result** at current dataset
scale (~1 % oscillation around path c); kept for ablation completeness.

### 3.5 Temporal forward (stage-3.4, "path a")

**Option A landed** (anchor-grid head preserved).  K-frame depth
sequence (B, K=10, 1, H, W) → reVAE encodes each frame in parallel
via batch reshape → `TemporalAggregator` (single-layer nn.GRU, 99 K
params, hidden=128 = reVAE latent) → last hidden state takes the
broadcast slot stage-1's z used to occupy.  YopoHead unchanged.

Option B (multi-waypoint GRU decoder output) explicitly deferred —
would re-shape every stage-3.1 loss per waypoint; ~2-week scope.

---

## 4. Dataset

### 4.1 Static — `dataset/`
Upstream YOPO baseline, unchanged: **30 envs × 10 000 depth frames**.

### 4.2 Dynamic — `dataset_dynamic/v1`
D-3 bake via extended CUDA raycaster (`DynSphere` + `ray_sphere_depth`):
**10 envs × 50 seq × 10 frames** = 500 sequences.  Per-sequence stored:
`depth_t{0..9}.png`, `dyn_obs.json` (ball trajectories), `state.json`
(drone trajectory), `meta.json`.

Geometric verification gate: `tools/verify_dynamic_render.py` re-projects
ball centers and checks per-pixel depth within tolerance.

### 4.3 Dataset scaling experiment (planned, stage-3.5)
Bake `dataset_dynamic/v2` at 2000+ sequences to test whether
`Dataset starvation` (the §3.4 hypothesis #1) is what kept DCA from
helping in stage-4.

---

## 5. Experiments

### 5.1 Training setup
batch=16, lr=1.5e-4, mixed sampling 50/50, seed=42, AdamW, grad-clip 0.1.

### 5.2 Stage-4 ablation (✅ done)
5 additive rows (A baseline-yopo → E full) × {2 k, 5 k} iter.  Headline:
dyn/kino loss adds the only meaningful dynamic-collision signal;
DCA side channel adds <1 % (noise).  Full table in
`docs/ARCHITECTURE.md` §Stage-4.

### 5.3 Stage-3.2 DCA A/B (✅ done)
5 k-iter A/B at fixed config: ~1 % `dyn_dyn` improvement, below noise.

### 5.4 Stage-3.4 temporal vs single-frame (✅ done; negative result)

A/B on v2 (5 k iter, batch=16, seed=42), `results/ablation_stage_3_4.csv`:

| Row | total tail | stat_traj | dyn_traj | **dyn_dyn** | wall |
|---|---:|---:|---:|---:|---:|
| F temporal off | 4.261 | 3.544 | 3.762 | **0.2366** | 145 s |
| G temporal on  | 4.187 | 3.559 | 3.770 | **0.2343** | 219 s |

`dyn_dyn` improves **1.0 %** with K-frame forward — same order as the
stage-3.2 DCA (~1 %).  Three independent architectural upgrades (loss
weighting, side channel, temporal) all converge at the same ~1 %
ceiling, strongly suggesting dataset scale (2 000 seq) is the
binding constraint.

Per-step wall +51 % (10× reVAE encodes).  Stage-5 deployment should
revisit the DiffPhysDrone stateful-GRU pattern (1 encode + carried
hidden) before profiling Jetson latency.

### 5.5 Deployment (⏸ TBD, stage-5)
- Success rate over N dynamic scenarios (sim or real); target ≥85 %.
- Inference latency on Jetson; target <10 ms.

---

## 6. Discussion

- Why DCA didn't help at this scale: dataset starvation + token
  redundancy with the already-encoded depth image.
- Why kinodynamic loss is dormant at single-waypoint: the configured
  envelope (v=8, a=10, j=30) is loose; output is one waypoint so V/A
  rarely approach limits.  Becomes binding for multi-waypoint
  trajectories (path a Option B).
- Limitations: assumes ground-truth obstacle radius and velocity (no
  perception end-to-end); single-drone; 3-DoF sphere obstacles only.

---

## 7. Conclusion + Future Work

- End-to-end perception (depth → obstacle parameters → planning).
- Multi-agent / multi-drone coordination.
- 6-DoF obstacles (swinging gates, non-rigid people).
- Online learning during deployment.

---

## Open Decisions (block parts of §3 / §5)

1. **Path-a output shape** — anchor-grid (Option A) vs N-waypoint
   sequence (Option B)?  Determines §3.5 and §5.4 design.
2. **Dynamic dataset size** — keep v1 (500 seq) or scale to v2 (≥2000)?
3. **Deployment target** — sim only, or real-flight as well?  Affects
   §5.5 scope and submission venue.

(Add notes inline as decisions land.)
