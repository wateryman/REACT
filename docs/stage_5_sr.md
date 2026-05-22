# Stage-5.B Closed-Loop SR Final Report

**Date:** 2026-05-22
**Sim:** REACT `sensor_simulator` ROS node (Simulator/, stage-5.B.1 dynamic-sphere extension), launched with `cwd = Simulator/` so the static forest cloud loads correctly.
**Scenarios:** 100 dynamic scenarios from `tools/eval_scenarios/v3_dyn_100/` (3-8 balls per scenario, 1-5 m/s, drone goal 20 m forward with ±30° yaw spread).
**Driver:** `scripts/stage5_closedloop_eval.py` Pass-1 (bounded double integrator) and Pass-2 (Poly5 min-jerk, matches deployment).

---

## Final 2×2 result table

| Metric                  | baseline (upstream YOPO_1) | C1-v2 (REACT stage-3.1 + z_floor) | Δ        |
|-------------------------|---------------------------:|-----------------------------------:|---------:|
| Pass-1 goal             |    34 %                     |       17 %                          | **-17 pp** |
| Pass-1 dyn_collision    |     2 %                     |        1 %                          |    -1    |
| Pass-1 static_collision |    25 %                     |       55 %                          | **+30 pp** |
| Pass-1 timeout          |    39 %                     |       27 %                          |   -12    |
| **Pass-2 goal**         |   **31 %**                  |   **31 %**                          | **0 pp** |
| Pass-2 dyn_collision    |     2 %                     |        3 %                          |    +1    |
| Pass-2 static_collision |     7 %                     |       24 %                          | +17 pp   |
| Pass-2 timeout          |    60 %                     |       42 %                          | -18 pp   |

CSV files:
- [results/stage5_sr_baseline_v3_realsim.csv](../results/stage5_sr_baseline_v3_realsim.csv)
- [results/stage5_sr_baseline_v3_realsim_pass2.csv](../results/stage5_sr_baseline_v3_realsim_pass2.csv)
- [results/stage5_sr_C1_v2_realsim_pass1.csv](../results/stage5_sr_C1_v2_realsim_pass1.csv)
- [results/stage5_sr_C1_v2_realsim_pass2.csv](../results/stage5_sr_C1_v2_realsim_pass2.csv)

The Pass-2 numbers are the ones to cite alongside the deployment target -- they use the same Poly5Solver that `YOPO/test_yopo_ros.py` feeds the SO(3) controller (see [REACT_MATH_Derivations/05_closedloop_dynamics.tex](../REACT_MATH_Derivations/05_closedloop_dynamics.tex)).

---

## Headline verdict

**REACT stage-3.1 (path-c, loss-only) is *neutral* on closed-loop SR in our benchmark:** 31 % vs 31 % at Pass-2.  Both methods fall well short of the ≥85 % paper target.

The failure modes are clearly *different*:
- baseline is **cautious** -- 60 % timeouts but only 7 % tree-strikes.  Upstream YOPO learned to slow down or detour around tree-rich regions.
- C1-v2 is **aggressive** -- 42 % timeouts (faster goal pursuit) but 24 % tree-strikes.  The 50/50 static/dynamic mixed-sampling diluted the static-obstacle training signal; C1-v2 navigates trees substantially worse than baseline.

The stage-3.1 motion-reshaped loss did not measurably improve dynamic-obstacle avoidance (dyn_collision 2 % vs 3 % is within noise) but it visibly changed the drone's *style*.

---

## Why Pass-1 vs Pass-2 differ so much

| Mode                  | Pass-1 (point_mass) | Pass-2 (Poly5) |
|-----------------------|--------------------:|---------------:|
| baseline goal         |    34 %             |     31 %       |
| C1-v2 goal            |    17 %             |     31 %       |
| baseline static_col   |    25 %             |      7 %       |
| C1-v2 static_col      |    55 %             |     24 %       |

Pass-1 "snaps" the drone to the predicted-velocity vector every 33 ms.  Any small angular drift in the score-head's argmin (e.g., C1-v2 occasionally picking a slightly-off-axis anchor) gets executed instantly, leading to oscillation near obstacles and frequent tree-strikes.

Pass-2 fits a 5th-order min-jerk polynomial between current state and predicted endstate, then steps it forward by dt.  The same angular drift becomes a smooth curve that the drone can complete without crashing.  baseline+Poly5 shrinks static_collision from 25 % to 7 % (-18 pp); C1-v2+Poly5 shrinks it from 55 % to 24 % (-31 pp).

**Conclusion: at deployment, the Poly5 path is mandatory.**  The Pass-1 numbers are useful for diagnosing planner vs controller failure-mode split but should not be the headline.

---

## Two non-obvious findings unblocked along the way

### F1 — Sim launched from wrong cwd silently disables the forest scene

`Simulator/src/src/maps.cpp::Maps::forest()` returns early if `tree.ply` fails to load (lines 988-992).  The `tree_file` config entry is a relative path, so launching `sensor_simulator` from `cwd=REACT/` rather than `cwd=Simulator/` causes the file lookup to fail, the early return kills the tree/ground generation, and the published `/depth_image` shows only random walls.

All SR numbers measured before this fix (the 23 % / 27 % / 66 % / 0 % entries on the old branch) were measured on a **scene with no trees**.  They are preserved in git for audit but should not be cited.

### F2 — Driver yaw bug that masked itself by the broken sim

The Plan A driver originally aligned drone yaw with the velocity vector.  When the trained model predicted strongly-forward endstates with near-zero lateral component (which C1 happened to do), the drone never yawed laterally toward a goal that wasn't dead-ahead -- it flew on a constant yaw line and missed.  Fix: align yaw with `goal - drone_pos` rate-limited at 2 rad/s, mirroring `YOPO/policy/poly_solver.py::calculate_yaw` used by `test_yopo_ros.py` at deployment.

---

## Three side-experiments that turned out not to matter

### S1 — C1-v1 without z_floor

The first stage-3.1 50-epoch training (`YOPO_11/epoch50.pth`) consistently predicted endstates ~2 m below ground.  Root cause: `YOPOLoss.safety_loss` queries an ESDF that has no ground plane and defaults to "free space" below z=0, so the score head exploits this gap to assign low cost to look-down anchors.  In broken-sim this produced SR=0/100.  We never measured it in real-sim because by then we already had Plan B's z_floor fix.

The math derivation [REACT_MATH_Derivations/01_collision_loss_saturation.tex §4](../REACT_MATH_Derivations/01_collision_loss_saturation.tex) called this out as a structural feature of the safety_loss formulation; the empirical signature confirmed it.

### S2 — Driver anchor filter (`end_z > -1 m`)

Plan A patched the driver to ignore look-down anchors at inference time.  At broken-sim this lifted C1's SR from 0 % to 27 %.  With Plan B (z_floor training) the filter becomes effectively no-op because C1-v2 rarely predicts z < z_floor at all.  We left the filter in place because removing it doesn't change C1-v2's numbers materially.

### S3 — baseline-v2-hack 66 % was a sim artifact

The 66 % baseline goal-rate reported at commit `f4b7427` was measured on the **broken (empty) sim** -- the drone had nothing to crash into.  Once real forest content was restored, baseline dropped to 34 % at Pass-1 and 31 % at Pass-2.

---

## Implications for the paper

1. **The closed-loop SR table is the definitive REACT result for stage-3.1.**  The on-disk loss numbers (Train/DynLoss 0.11, etc.) and ablation tables from stages 3.x and 4 should be presented but interpreted as "internal training metrics that do not predict closed-loop SR."

2. **Stage-3.1 result framing.**  Two options for the paper:
   - **Frame as null result:** "We observe that REACT-stage-3.1 yields equal closed-loop SR to upstream YOPO (31 % vs 31 %), with shifted failure modes (more aggression, more tree-strikes, fewer timeouts).  The motion-reshaped dyn-loss training did not unlock dynamic obstacle avoidance under our benchmark."
   - **Frame as failure-mode analysis:** "Stage-3.1's behavior change is detectable and consistent (more aggression, more crashes, fewer timeouts), so the loss IS shaping the policy -- but the shape is not net-beneficial for SR.  Future work: weighted mixed-sampling, or fine-tune from baseline weights rather than from scratch."

3. **The 85 % paper target is not met by either method.** This is honest reporting; the paper either needs further architectural / data work (Option B multi-waypoint head, camera-aware bake, data scale-up beyond 8000) or it should be reframed around the analysis itself (closed-loop bench, sim-vs-deployment gap quantification, failure-mode comparison).

---

## What we did NOT measure (and why it's OK to ship without)

- C1-v1 in real-sim (broken-sim showed 0 %; z_floor was added in v2 specifically to fix that).
- Stage-3.2 (DCA path-b) in real-sim closed loop (we already have the dyn_dyn 1 % ablation that says it's noise; closed-loop SR would be expected to track baseline).
- Stage-3.4 (path-a temporal) in real-sim closed loop (dyn_dyn 1 % at 5 k iter; full 50-epoch retraining + real-sim SR is the natural follow-up if we want a multi-row ablation table).
- C2 stage-3.4 v2 (z_floor + temporal) was committed as a 5-7 h additional training run that we have not started.

Each of these is a 2-4 h investment from current state.  Recommend running them only if the paper outline requires the corresponding row in the final ablation table.

---

## Reproduction commands

```bash
# Start sim from Simulator/ (cwd matters -- tree.ply path is relative!)
cd Simulator && ./devel/lib/sensor_simulator/sensor_simulator &
cd ..

# Baseline (upstream YOPO checkpoint)
python scripts/stage5_closedloop_eval.py \
  --ckpt YOPO/saved/YOPO_1/epoch50.pth \
  --label baseline_v3_realsim \
  --dynamics poly5 \
  --out results/stage5_sr_baseline_v3_realsim_pass2.csv

# C1-v2 (REACT stage-3.1 + z_floor, 50 epoch)
python scripts/stage5_closedloop_eval.py \
  --ckpt YOPO/saved/YOPO_12/epoch50.pth \
  --use-revae \
  --label C1_v2_zfloor_realsim_pass2 \
  --dynamics poly5 \
  --out results/stage5_sr_C1_v2_realsim_pass2.csv

# Each run: ~15 minutes on RTX 3070.
```
