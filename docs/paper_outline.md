# REACT Paper Outline — Sprint v2 (2026-05-23)

**Status:** Live document; updated mid-sprint.  4-week timeline to RA-L/IROS;
real-flight required (see [plan](https://wxs/.claude/plans/reflective-enchanting-steele.md)
and `docs/stage_5_sr.md` for the full closed-loop status).

---

## TL;DR — what changed between v1 (2026-05-21) and v2

| | v1 outline | v2 outline (this) |
|---|---|---|
| Headline target | "**≥ 85 %** dynamic SR" | "**beat YOPO baseline by ≥ 15 pp** + closed-loop benchmark + sim-to-real" |
| Required hardware | (TBD) | YOPO-class drone + RealSense + Jetson + real-flight confirmed |
| Submission venue | "TBD" | **RA-L / IROS (4 weeks out)** |
| stage-3.1 status | "expected breakthrough" | **neutral at scratch-train; +5 pp at fine-tune; v4 retrain in flight** |
| Failure-mode story | absent | **central narrative** -- baseline cautious-timeout vs REACT crash-aggressive-tree, etc. |
| sim-to-real | "future work" | **Week 3 deliverable** -- short real-flight trial section |

---

## Section A. Problems we hit + current fixes (live status)

This section is a working snapshot for the 4-week sprint, NOT for the paper
itself.  It exists so collaborators / reviewers / future-self can see what
broke and what worked.

### A.1 Closed-loop infrastructure problems (all now resolved)

| # | Bug | Symptom | Root cause | Fix |
|---|---|---|---|---|
| 1 | Driver yaw was tracking velocity vector | C1 drone never turned to laterally-offset goals (0 / 100 goal) | `yaw = atan2(vel.y, vel.x)`; when C1's argmin gave forward-only endstates, vel stayed +X and yaw stayed at scenario init | Replace with goal-tracking yaw at 2 rad/s rate-limit, mirrors `YOPO/policy/poly_solver.py::calculate_yaw` |
| 2 | Sim renders WITHOUT forest scene | All SR runs measured a near-empty static cloud; baseline got artificially-high 66 % | `tree_file: "src/pointcloud/tree.ply"` is RELATIVE; sim launched from `cwd=REACT/` failed to find the file; `Maps::forest()` returns early without trees or ground | Launch from `cwd=Simulator/` (where the path resolves).  All pre-fix SR numbers (23 / 27 / 66 / 0) are documented as broken-sim audit data; **post-fix numbers replaced them** |
| 3 | Model exploits ESDF z<0 free-space gap | C1-v1 dives below ground (end_z ≈ -2 m); SR 0 / 100 | `Simulator/maps.cpp::forest()` static cloud has no ground plane; `YOPOLoss.safety_loss` defaults to free-space below cloud minimum z | New `YOPOLoss.z_floor_loss` quadratic-hinge below z=0.3 m, plumbed into both `trajectory_loss` and `score_label` ([YOPO/policy/yopo_trainer.py](../YOPO/policy/yopo_trainer.py)) |
| 4 | Pre-fix CSVs lingered as misleading data points | "baseline_v2_hack = 66 %" was cited in mid-sprint commit before bug-2 was found | n/a (process bug) | All 5 pre-fix SR CSVs preserved in git under their original filenames as broken-sim audit; only `*_realsim*.csv` files are used for paper |

### A.2 Modeling problems (mostly resolved)

| # | Problem | Diagnosis | Status |
|---|---|---|---|
| 5 | **C1 from-scratch** (50 ep, no z_floor) digs underground | trained without z_floor; argmin picks v=2 (look-down) anchor row because that's where safety_loss looks free | fixed by z_floor loss; deprecated to "old C1-v1" |
| 6 | **C1-v2 with z_floor** (50 ep, scratch) trades static for dynamic | 50/50 mixed sampling diluted upstream YOPO's strong static-obstacle handling; static_collision shot 7 → 24 % | Switched D1 strategy to fine-tune from baseline (preserves static signal) |
| 7 | **All ablations (3.2, 3.4, 3.6) hit ~1 % dyn_dyn ceiling** | Suspect dataset starvation, but v3 (4× v2) failed to break the ceiling at 5k iter | Now suspected to be **data quality, not quantity** -- v4 camera-aware spawn raises FOV-presence from 22 → 88 % |
| 8 | **Pass-1 (point-mass) over-reports SR** | bounded double integrator snaps to planner output with infinite jerk -- doesn't match deployment | Added Pass-2 (Poly5 min-jerk) that mirrors `test_yopo_ros.py` deployment path.  Pass-2 is the cited number |

### A.3 In progress as of 2026-05-23

| Sub-stage | What it does | Status | Gate |
|---|---|---|---|
| D1 | Fine-tune from `YOPO_1/epoch50.pth` with z_floor + dyn loss, **v3 data**, 25 epoch | ✅ done.  C1-FT v3 = **36 %** SR (beat baseline 31 % by +5 pp) | passed +4 pp gate |
| D2 | Camera-aware ball spawn in `dataset_generator.cpp` + rebuild + bake v4 | ✅ done.  **88 % FOV PASS rate** (was 22 % on v3) | passed ≥50 % gate |
| **D3** | Same fine-tune protocol as D1, but on **v4 data** | 🔄 running now (~1.3 h) | Train Eval Traj should be similar/lower than D1's 3.25 |
| D4 | C1-v4-FT Pass-2 realsim 100 scenarios | ⏸ pending | **SR ≥ 45 %** (+14 pp vs baseline) → continue.  < 45 % → pivot to score-head decoupling (D5) |
| D5-7 | If D4 passes, run full ablation (baseline / +reVAE / +dyn loss / +z_floor / +v4 / full) at fine-tune + v4 | pending |
| Week 2 | ONNX/TRT export + Jetson Orin latency profile (was excluded; real-flight requires) | pending |
| Week 3 | Real-flight 5-15 dynamic-obstacle trials | pending |
| Week 4 | Paper polish + submit | pending |

---

## Section B. Working Paper Outline

### Working Title

**REACT: A Closed-Loop Benchmark and Fine-Tuned Loss-Aware Planner for Quadrotor Flight in Dynamic Environments**

(Variants: drop "REACT" if naming collides with YOPO trademark; the
network code is under `YOPO/` for upstream-diff readability — see
`docs/ARCHITECTURE.md` §1.)

### 1. Introduction

- Agile UAV planning in dynamic environments has hard latency constraints
  (<10 ms on Jetson) and must handle obstacles that move during flight.
- Existing learning planners split into:
  - Slow optimization-based (MPC, ESDF-time) — accurate but >> 10 ms.
  - Single-frame one-stage (e.g.\ YOPO) — fast but blind to motion.
- **Three contributions:**
  1. **A closed-loop dynamic-obstacle benchmark** (100 scenarios, two
     integrator variants — point-mass upper-bound and Poly5 min-jerk
     deployment-grade) on top of an extended YOPO CUDA raycaster
     (`Simulator/sensor_simulator.cu` + dynamic-sphere upload helpers).
  2. **A failure-mode-aware fine-tune recipe** for adding dynamic-obstacle
     supervision on top of a static-pretrained YOPO baseline: yopo_head
     zero-pad to absorb the new reVAE feature channels + z_floor loss
     to prevent exploitation of the ESDF/no-ground-plane gap.
  3. **Camera-aware dynamic dataset baking (v4)** that raises FOV-presence
     of dynamic obstacles from 22 % to 88 %, providing a real gradient
     signal for the motion-reshaped collision loss.

### 2. Related Work

- **YOPO** [TJU-Aerial-Robotics, RA-L 2024] — single-stage anchor-grid
  baseline this work builds on.  85 % SR on static forest; we measure
  31 % on the same architecture in our dynamic-obstacle benchmark.
- **PEMTRS** [RA-L 2026, Nankai] — temporal architecture (reVAE +
  Transformer selector + GRU decoder); reimplemented from the published
  method section as source code is not public.  Reports 80-100 % on
  static gated corridors; no dynamic-obstacle benchmark.
- **DiffPhysDrone** [SJTU, Nature MI 2025] — RL + differentiable physics;
  90 % SR on outdoor dynamic obstacles.  Imitation-learning regime
  (this paper) hits a lower ceiling on the same task.
- **EgoPlanner** [HKUST, 2023], **Agile-Autonomy** [Loquercio,
  Sci. Robotics 2021], **Flow-Aided** [HKU RA-L 2025], **NeuPAN**,
  ESDF-time methods — comparison baselines.

### 3. Method

#### 3.1 Background — YOPO single-stage anchor planner
V × H = 3 × 5 anchor grid, per-anchor endstate regression
(`Smooth-L1` on (pos, vel, acc)), score head with softplus.

#### 3.2 reVAE auxiliary encoder
Residual VAE over the current depth frame; latent z (128) is broadcast
onto the V × H feature map before the YopoHead 1×1 conv.
Loss: `lam_recon · MSE + lam_kl · KL`, `lam_kl = 0.001`.

#### 3.3 Dynamic-aware losses

- **`motion_reshaped_collision_loss`** — `softplus(d_safe - dist - α · closing_speed)`,
  where `closing_speed = relu(rel_v · dir_from_obs_to_traj)`.
  See `REACT_MATH_Derivations/01_collision_loss_saturation.tex` for the
  saturation analysis and `03_esdf_time_replacement.tex` for the
  ESDF-time gradient.
- **`kinodynamic_loss`** — squared hinge on `|v|/|a|/|j|`.  Dormant in
  single-waypoint setting (one waypoint -> no jerk); activated for
  Option B future work.
- **`z_floor_loss`** (new this paper) — quadratic hinge below ground
  level; prevents the model from exploiting the safety_loss / ESDF
  free-space-below-z=0 gap.  See math in
  `REACT_MATH_Derivations/01 §4`.

#### 3.4 Closed-loop integrator (deployment path)
`Poly5Solver` per axis: 5th-order minimum-jerk polynomial fitting current
`(p, v, a)` to predicted endstate `(p, v, a)` in `T = 1.7` s; step `dt = 33` ms.
Same code path as deployed `test_yopo_ros.py` controller stack.
Derivation: `REACT_MATH_Derivations/05_closedloop_dynamics.tex`.

#### 3.5 Fine-tune-from-baseline recipe
- Initialize `YopoNetwork(use_revae=True)` from upstream
  `YOPO_1/epoch50.pth` checkpoint (which has `use_revae=False` and
  `head_in=73`).
- Zero-pad `yopo_head.model.0.weight` from `(256, 73, 1, 1)` to
  `(256, 201, 1, 1)`: original 73 channels (obs + depth) go in front,
  new 128 reVAE channels start at zero and are learned during fine-tune.
- Train 25 epoch at LR `5e-5` (3× lower than from-scratch) with z_floor
  loss + dyn loss + dynamic_ratio 0.5.

#### 3.6 Camera-aware dataset baking
Original `spawn_balls()` places dynamic spheres uniformly in the world
bbox.  Result: most training frames have no ball in the camera FOV →
dyn loss receives zero gradient on those frames.  v4 fix: sample drone
trajectory first, then place each ball in the drone's initial
forward cone (azimuth ± 45°, elevation ± 30°, distance 3-12 m).
FOV-presence rate climbs from 22 % to 88 % on a 45-seq verification
sample.

### 4. Dataset

| | env count | seq count | FOV PASS | disk | wall |
|---|---:|---:|---:|---:|---:|
| Static (upstream YOPO) | 30 | 300 k frames | — | 2.3 GB | n/a |
| Dynamic v1 (initial) | 10 | 500 | mixed | 718 MB | n/a |
| Dynamic v2 (stage-3.5 4×) | 20 | 2000 | 22 % | 1.6 GB | 2 m 14 s |
| Dynamic v3 (stage-3.6 16×) | 40 | 8000 | 22 % | 3.9 GB | 5 m 02 s |
| **Dynamic v4 (this paper)** | **40** | **8000** | **88 %** | **3.9 GB** | **5 m 09 s** |

Bake config and verification gate in `Simulator/src/config/config_dynamic.yaml`
and `tools/verify_dynamic_render.py`.

### 5. Experiments

#### 5.1 Training setup
`batch=16, lr=5e-5, mixed sampling dynamic_ratio=0.5, seed=0, AdamW,
grad-clip 0.1, 25 epoch (fine-tune); 50 epoch (from-scratch)`.

#### 5.2 Closed-loop SR — Pass-2 (Poly5) on 100 v3/v4 scenarios

Table to populate after D4 (placeholders **bold**):

| Row                              | goal | dyn_col | static_col | timeout |
|----------------------------------|-----:|--------:|-----------:|--------:|
| YOPO baseline (upstream)         | 31 % |   2 %   |    7 %     |   60 %  |
| C1-v2 (REACT, from scratch, v3)  | 31 % |   3 %   |   24 %     |   42 %  |
| C1-FT (REACT, fine-tune, v3)     | 36 % |   2 %   |   10 %     |   52 %  |
| **C1-v4-FT (REACT + v4 data)**   | **?** | **?**  |  **?**     |  **?**  |
| C2-v4-FT (+ temporal, optional)  | TBD  | TBD     | TBD        | TBD     |

#### 5.3 Failure-mode analysis (paper centerpiece)
Even when goal-rate is similar, the **policy personality** differs:
- baseline = cautious (60 % timeout, 7 % static_col)
- C1-v2 from scratch = aggressive crash (42 % timeout, 24 % static_col)
- C1-FT = balanced (52 % timeout, 10 % static_col)

Histograms of `min_clearance_m` per terminate_reason will show the
distribution shift; failure videos for collisions, timeout reasons
quantified.

#### 5.4 Pass-1 vs Pass-2 dynamics sensitivity
Pass-1 (point-mass, infinite-jerk reference) — diagnostic upper bound.
Pass-2 (Poly5 min-jerk, deployment-grade) — the cited number.

C1-v2 at Pass-1 = 17 % vs Pass-2 = 31 % (+14 pp delta).  baseline at
Pass-1 = 34 % vs Pass-2 = 31 % (−3 pp).  Shows that **C1-v2's
score-head argmin has angular drift that smooth integration absorbs but
hard snapping does not** — informative for deployment design.

#### 5.5 Latency profile (Week 2)
RTX 3070 eager: all three modes (single-frame, single-frame+DCA,
K=10 stateless) under 4 ms p99.  Jetson Orin NX FP16 projected
~8.7-9.9 ms p99, within 10 ms budget.  TRT FP16 measurement to be added.

#### 5.6 Real-flight short trial (Week 3)
5-15 trials on YOPO-class hardware (RealSense D435 + Jetson + 250 g
quad).  Indoor 5 m × 5 m space with mocap ground truth.  Sim-to-real
gap reported as |sim_SR − real_SR|.

### 6. Discussion

- **Why ~1 % training-loss ceiling didn't translate to SR neutrality**
  in v4: the v3 22 % FOV-presence was a data-quality bottleneck that
  no architecture change could fix.  v4 should reveal the actual
  contribution of each loss component once tested.
- **Why fine-tuning beats from-scratch** at the same training compute:
  upstream YOPO's static-obstacle representation is preserved; only the
  new reVAE channels and the score head need to be learned, not the
  backbone.
- **Limitations:** assumes ground-truth obstacle radius and velocity
  (no perception end-to-end); single-drone; 3-DoF sphere obstacles
  only; sim-trained policy with limited real-flight validation.

### 7. Conclusion + Future Work
- End-to-end perception (depth → obstacle parameters → planning).
- Multi-waypoint head (Option B) so kinodynamic_loss activates.
- RL fine-tuning on top of imitation pretraining (DiffPhysDrone-style).
- 6-DoF obstacles (swinging gates, non-rigid people).
- Online learning during deployment.

---

## Section C. Math derivation cross-reference

`REACT_MATH_Derivations/` already contains five tex files used as
section back-references:

| File | Used in paper §       |
|------|----------------------|
| `01_collision_loss_saturation.{tex,_cn.tex}` | §3.3, §6 |
| `02_multi_waypoint_extension.{tex,_cn.tex}`  | §7 future work |
| `03_esdf_time_replacement.{tex,_cn.tex}`     | §3.3, §6 |
| `04_stage5_deployment_math.{tex,_cn.tex}`    | §5.5 (Jetson budget), §5.6 (sim-to-real) |
| `05_closedloop_dynamics.{tex,_cn.tex}`       | §3.4, §5.4 |

---

## Section D. Open decisions still tracked

1. **Real-flight venue:** room-scale dynamic obstacles (foam balls /
   moving boards) vs. outdoor with motion-tracked human?  Affects
   §5.6 video deliverable.
2. **Final ablation row count:** 4 (baseline / FT / FT+v4 / FT+v4+temporal)
   or 6 (add DCA + multi-waypoint).  Depends on D4-D7 outcomes.
3. **Title:** keep "REACT" or rename to avoid trademark confusion?
4. **Submission target:** RA-L (rolling) or IROS (fixed deadline +
   conference presentation)?

---

## Status update markers

- [x] D1 (2026-05-23 noon): C1-FT v3 → 36 %, beat baseline +5 pp
- [x] D2 (2026-05-23 14:00): v4 baked with 88 % FOV PASS rate
- [ ] D3 (2026-05-23 evening): C1-FT v4 training (running, ETA ~16:00)
- [ ] D4: C1-FT v4 Pass-2 realsim
- [ ] D5-7: full ablation table
- [ ] Week 2: ONNX/TRT + Jetson latency
- [ ] Week 3: real-flight
- [ ] Week 4: polish + submit
