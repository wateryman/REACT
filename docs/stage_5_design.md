# Stage-5 Deployment Design

**Status:** Draft, post-stage-3.6.
**Goal:** measure the two paper-target numbers (<10 ms latency, ≥85 % closed-loop SR) without committing to any further training change, so that any supervision upgrade we *do* later make is data-driven.

The math behind the staging order is in
[REACT_MATH_Derivations/04_stage5_deployment_math.tex](../REACT_MATH_Derivations/04_stage5_deployment_math.tex) §4 (CN: [04_stage5_deployment_math_cn.tex](../REACT_MATH_Derivations/04_stage5_deployment_math_cn.tex)).

---

## 1. Scope decomposition

Stage-5 splits into two near-independent sub-stages.  Either can land first, but 5.A returns the latency answer faster (no closed-loop infrastructure needed) so we start there.

### 5.A — Inference latency profile (~3 days)

1. **5.A.0 — Dev-box baseline.** Run `policy.inference()` on a synthetic batch with the current stage-3.1-ish weights (use the v0.3.4-temporal HEAD model when `use_temporal=False`, equivalent to stage-3.1+3.2 default). Measure p50 / p95 / p99 latency for:
   - single-frame (`depth.shape = (1, 1, 96, 160)`)
   - K-frame stateless (`depth.shape = (1, 10, 1, 96, 160)`)
   - "stateful K=1" — same as single-frame but with the temporal aggregator on graph (mimics the inference-time stateful pattern)

   Done on RTX 3070 8 GB (dev box), CUDA 11.8.  This is a *correlate*, not a Jetson number — but gives an upper bound on per-step compute.

2. **5.A.1 — TorchScript / `torch.compile`** trace of the network. Compare timings.  Document any node that fails to trace (most commonly: dynamic-shape branches inside `if self.use_temporal:` — we may need to compile two separate graphs).

3. **5.A.2 — ONNX export** of the static-shape network (one for each mode). Verify outputs match within 1e-3 of the eager forward.  Save under `deploy/onnx/`.

4. **5.A.3 — TensorRT engine build** (FP16 first, INT8 with calibration if FP16 misses the budget). Build on the same RTX 3070 first — production Jetson engines must be built *on* the Jetson because they're device-specific, but the FP16 RTX number is the closest signal we have.

5. **5.A.4 — Latency report.** Tabulate eager vs TorchScript vs ONNX vs TRT-FP16 (vs TRT-INT8) for each of the three input modes.  Decide whether single-frame stage-3.1 has enough margin to fit Jetson at <10 ms (likely yes), and whether K=10 stateless fits (marginal, per stage_5_deployment_math §2.2).

### 5.B — Closed-loop success-rate measurement (~4 days)

1. **5.B.0 — Scenario set.** Generate 100 dynamic scenarios that match the v3 bake distribution (same `dyn_obs` params: 3-8 balls, 1-5 m/s, 0.3-0.6 m radius, 30×30 m field). Reuse the static-forest pointclouds from `dataset_dynamic/v3/env_*/`.  Save as `tools/eval_scenarios/v3_dyn_100/*.yaml`.

2. **5.B.1 — Closed-loop driver.** Spawn the drone, run controller + planner at 30 Hz against the simulator, log:
   - per-frame state (`pos`, `vel`, planner endstate, score)
   - per-frame closest-obstacle distance
   - terminal condition: `goal_reached` / `collision` / `timeout` (default 10 s)

   The existing `Simulator/` + `Controller/` ROS packages already provide most of this; we add a thin Python driver that loops over scenarios and aggregates results.

3. **5.B.2 — Metric computation.**
   - Success rate (SR) = `goal_reached` / total scenarios
   - Mean time to goal among successes
   - Minimum clearance to obstacle distribution
   - Collision-detail report for failed scenarios (which ball, which frame, was prediction known-bad?)

4. **5.B.3 — Cross-config A/B.** Run two configs on the same 100 scenarios:
   - **C1**: stage-3.1 single-frame model (`use_temporal=False`, `use_dca=False`).
   - **C2**: stage-3.4 temporal model (`use_temporal=True`).

   The score-head selection factor `k` from
   [04_stage5_deployment_math.tex](../REACT_MATH_Derivations/04_stage5_deployment_math.tex) §3 is exactly the SR/dyn_dyn ratio we observe here.  Reports go to `results/stage5_closedloop_*.csv`.

---

## 2. Decision rule at the end of stage-5

| Outcome | Action |
|---|---|
| C1 SR ≥ 85 % | **Ship stage-3.1.**  Write paper around it.  Stage-3.2/3.4 stay as ablation rows.  Skip supervision upgrade. |
| 70 % ≤ C1 SR < 85 % | Compare C2 to C1.  If C2 is clearly better, ship C2 (stage-3.4) instead.  Otherwise pursue Option B (multi-waypoint head — see `02_multi_waypoint_extension.tex`). |
| C1 SR < 70 % | Triage: is it the planner (high dyn_dyn at failure frames?) or the controller (high tracking error at safe predictions?) or the data (most failures in <22 % FOV-presence frames?).  Stage-3.7 plan follows from this triage. |

---

## 3. Risks & mitigations

| Risk | Mitigation |
|---|---|
| RTX 3070 latency wildly different from Jetson (10× faster) | Document both numbers; quote relative scaling from public Jetson benchmarks of similar networks. |
| Closed-loop sim has bugs that mask real model performance | 5.B.0's scenarios include reference paths (straight-line goals where stage-3.1 should trivially succeed); failure on these means the driver is buggy, not the planner. |
| GPU memory pressure on dev box (8 GB) under TRT | Run inference benchmarks at batch=1 (always); training-style batches not needed in 5.A. |
| `torch2trt` API differences vs current torch 2.4 | Existing test_yopo_ros.py uses it; expect mild patching.  Fallback: direct ONNX → TRT via `trtexec`. |

---

## 4. What we will NOT do in stage-5

- Real-flight tests on hardware.  Out of scope without an actual drone.
- Re-training.  The whole point of stage-5 is to measure what we already have.
- Camera-aware bake (data quality lever from
  `01_collision_loss_saturation.tex` §4 last bullet).  That's stage-3.7+ if 5.B says we need it.
- Multi-waypoint head (Option B).  Same — stage-3.7 if needed.

---

## 5. Phases & deliverables

| Phase | Output |
|---|---|
| 5.A.0 — eager baseline | `scripts/stage5_latency_baseline.py` + `results/stage5_latency_baseline.csv` |
| 5.A.1 — TorchScript trace | log delta vs eager |
| 5.A.2 — ONNX export | `deploy/onnx/yopo_stage_3_1.onnx`, `deploy/onnx/yopo_stage_3_4.onnx` |
| 5.A.3 — TRT engine | `deploy/trt/*.engine` + build log |
| 5.A.4 — latency report | `docs/stage_5_latency.md` (or section in ARCHITECTURE.md) |
| 5.B.0 — scenario set | `tools/eval_scenarios/v3_dyn_100/*.yaml` (100 files) |
| 5.B.1 — closed-loop driver | `scripts/stage5_closedloop_eval.py` |
| 5.B.2 — metric impl | inside `stage5_closedloop_eval.py` |
| 5.B.3 — A/B run | `results/stage5_closedloop_C1.csv`, `results/stage5_closedloop_C2.csv`, `docs/stage_5_SR.md` |

Total estimated work: **~7 working days** (3.A + 4.B).  Phase 5.A can land independently; 5.B has no dependency on 5.A's results.
