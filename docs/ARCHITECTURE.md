# REACT Architecture

This page explains the repository's three-package layout, the data flow at
both bake-time and training-time, and where REACT's changes sit relative to
the upstream YOPO baseline.

## Why the `YOPO/` subdirectory keeps its name

The outer directory layout (`Controller/ Simulator/ YOPO/` siblings) mirrors
[TJU-Aerial-Robotics/YOPO](https://github.com/TJU-Aerial-Robotics/YOPO) byte
for byte at the structural level. Anyone familiar with YOPO can `git diff` the
two trees to see exactly which files REACT touched.

Renaming to e.g. `react_planner/` would lose this property and add a 200+ line
churn (paths, imports, CMake, docs, every commit message that references the
file by name). The decision: **defer rename to v1.0 release if it's still
desired then**. README's directory tree carries a one-line note pointing
readers here.

## The three packages

| Package | Role | Build chain | REACT changes (so far) |
|---|---|---|---|
| **Controller/** | ROS quadrotor dynamics + so3 position/attitude controllers | `catkin_make` | none (unchanged from YOPO) |
| **Simulator/** | CUDA raycaster against a random-forest point cloud; ROS sensor publisher + offline dataset generator | `catkin_make --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5` | stage-2: `DynSphere` struct + `ray_sphere_depth` device function in `sensor_simulator.cu`; `dataset_generator.cpp` gains a `mode: "dynamic"` branch that bakes K-frame sequences |
| **YOPO/** | Python training + inference (depth -> trajectory) | `pip install -r YOPO/requirements.txt` (system Python; preflight checks) | stage-1: `policy/models/{revae,temporal_selector,gru_decoder}.py` + `policy/utils/frame_buffer.py` + reVAE loss integrated in `yopo_network.py` + `yopo_trainer.py`; stage-2: `policy/yopo_dataset.py` gains `--dynamic` switch |

Plus two small directories at the top level:

- `scripts/` — preflight, smoke tests, training runners, tfevents extractor.
  Standalone Python scripts; not a package.
- `tools/` — verification scripts for the data baked by `Simulator/`.

## Data flow

### Bake-time (stage-2 D-3, mode=dynamic)

```
config_dynamic.yaml ──┐
                      ▼
 ┌─────────────── dataset_generator (C++, Simulator/) ────────────────┐
 │                                                                     │
 │   for each env:                                                     │
 │     mocka::Maps -> static point cloud (forest)                      │
 │     GridMap     -> CUDA voxel index                                 │
 │     pcl::KdTreeFLANN -> collision query for drone-spawn safe_dist   │
 │                                                                     │
 │     for each sequence (K=10 frames):                                │
 │       spawn 3-8 DynamicBall (random pos/vel/radius)                 │
 │       sample drone trajectory: K frames, body-+X velocity, retry    │
 │                                  if any frame collides with cloud   │
 │                                                                     │
 │       for each k in [0, K):                                         │
 │         step balls (axis-aligned reflect at bbox)                   │
 │         uploadDynamicSpheres -> GPU                                 │
 │         renderDepthImage(GridMap, camera, T_wc, DynSphere*) ─┐      │
 │                                                              │      │
 │       depth_t{k}.png (uint16 normalized by max_depth_m=20m)  │      │
 │       state.json (pos, quat_wc, vel_world per frame)          │      │
 │       dyn_obs.json (ball pos/vel/radius per frame)            │      │
 │       meta.json   (K, dt, intrinsics, depth_encoding)         │      │
 └──────────────────────────────────────────────────────────────┴──────┘
                                                          dataset_dynamic/v1/
```

The same `renderDepthImage()` function with `n_dyn=0` produces the original
static dataset under `dataset/` — there is exactly one renderer; static and
dynamic data are pixel-identical in distribution at every voxel the dynamic
balls don't touch.

### Training-time (stage-1 today, stage-3 TBD)

Stage 1 (current, single-frame forward path):

```
dataset/                                  YOPODataset (dynamic=False)
  env_X/depth_*.png    ────────────────►   __getitem__ returns 5-tuple
  pose-X.csv                              (image, pos, rot, obs, map_id)
                                                       │
                                                       ▼
                                            YopoNetwork.forward(depth, obs)
                                              │
                                              ├─ ResNet18 (depth feature, V×H)
                                              ├─ ReVAE (depth -> latent 128)         🟩 stage-1 added
                                              ├─ broadcast latent to V×H
                                              └─ cat -> YopoHead -> (9, V, H) + V×H score
                                                       │
                                                       ▼
                                          YOPOLoss = smooth + safety + goal + acc
                                          + score Huber                              🟦 YOPO
                                          + 0.1 * (mse_recon + 0.001 * KL)           🟩 stage-1 added
```

Stage 3 (planned, architecture choice deferred — see below):

```
dataset_dynamic/v1/                       YOPODataset (dynamic=True)
  env_X/seq_Y/depth_t*.png  ────────────►  __getitem__ returns dict
  env_X/seq_Y/state.json                  (depth_seq, state_seq, dyn_obs,
  env_X/seq_Y/dyn_obs.json                 dt_seq, meta)
  env_X/seq_Y/meta.json                                │
                                                       ▼
                                          ??? (one of three choices)
                                                       │
                                                       ▼
                                          existing loss + motion_reshaped + kinodynamic
```

## Module map

Files marked 🟧 are REACT additions; the rest are inherited from upstream YOPO
or are tooling REACT added without touching upstream behaviour.

```
YOPO/
├── train_yopo.py                   stage-1 entry; instantiates YopoTrainer
├── policy/
│   ├── yopo_network.py             🟩 stage-1: reVAE wired into YopoNetwork
│   ├── yopo_trainer.py             🟩 stage-1: revae_loss added to total
│   ├── yopo_dataset.py             🟧 stage-2 2.e: --dynamic switch
│   ├── poly_solver.py / primitive.py / state_transform.py     (YOPO upstream)
│   ├── models/
│   │   ├── backbone.py / head.py / resnet.py                  (YOPO upstream)
│   │   ├── revae.py                🟩 stage-1: residual VAE encoder
│   │   ├── temporal_selector.py    🟩 stage-1: Transformer ROI selector (unwired)
│   │   └── gru_decoder.py          🟩 stage-1: GRU + cross-attn + 3 heads (unwired)
│   └── utils/
│       └── frame_buffer.py         🟩 stage-1: K-frame sliding window (unwired)
├── loss/
│   ├── loss_function.py            🟩 stage-1: revae_loss() static method added
│   ├── smoothness_loss.py / safety_loss.py / guidance_loss.py (YOPO upstream)
└── config/
    └── traj_opt.yaml               🟩 stage-1: + revae/frame_buffer/selector/gru_decoder/loss_weights
                                    🟧 stage-2 2.e: + dataset_dynamic_path

Simulator/src/
├── include/
│   ├── sensor_simulator.h          (YOPO upstream)
│   ├── sensor_simulator.cuh        🟧 stage-2 2.a: + DynSphere struct + extended fn signatures
│   └── maps.hpp / perlinnoise.hpp                              (YOPO upstream)
├── src/
│   ├── sensor_simulator.cu         🟧 stage-2 2.a: + ray_sphere_depth + kernel dyn loop + upload/free helpers
│   ├── dataset_generator.cpp       🟧 stage-2 2.b: + dynamic-mode orchestrator + DynamicBall + inline JSON
│   ├── test_dyn_sphere.cpp         🟧 stage-2 2.a: smoke (3/3 PASS at <0.05m)
│   ├── sensor_simulator.cpp / test_simulator*.cpp / perlinnoise.cpp / maps.cpp   (YOPO upstream)
└── config/
    ├── config.yaml                 (YOPO upstream; mode: static)
    └── config_dynamic.yaml         🟧 stage-2 2.b: mode: dynamic + sequence params

scripts/                            (REACT tooling, all 🟧/🟦)
tools/                              (REACT tooling, all 🟧/🟦)
docs/                               docs + viz samples
```

## Path-c (stage-3.1) and path-b (stage-3.2): empirical findings

The original "unwired modules" plan offered three paths to wiring the
dynamic-obstacle channel into the forward graph (c: loss-only;
b: side-channel tokens; a: full K-frame).  Paths c and b are now both
implemented and on `main`; here is the verdict.

### Stage-3.1 (path c, `v0.3.1-loss-only`)

`motion_reshaped_collision_loss` and `kinodynamic_loss` were added to
the YOPOLoss class as standalone weighted terms; `forward` was not
changed.  Mixed-sampling (50% static / 50% dynamic) plus the new losses
were validated by `scripts/run_stage3_1k.py` — all five 過関 gates clear.

### Stage-3.2 (path b, `v0.3.2-side-channel`)

`YopoNetwork.forward` gained an optional `dyn_obs_tokens` / `dyn_obs_mask`
pair.  When `cfg.dynamic_attention.enable=true` the network instantiates
`DynObsEncoder` (7 -> 201 dim MLP) and `DynamicCrossAttention` (n_heads=1
because head_in=201 is not divisible by 4).  The trainer builds tokens
from the baked dyn_obs payload as `[rel_pos, abs_vel, radius]` where
`rel_pos = obs_pos - drone_pos` (world-frame, no yaw rotation).

### 5k-iter A/B comparison (DCA off vs DCA on, same dataset, same lr, batch=16)

| Metric              | path c (DCA off) | path b (DCA on) | Delta |
|---------------------|------------------|-----------------|-------|
| total head -> tail  | 5.52 -> 4.20     | 5.52 -> 4.17    | -0.03 (0.8%)  |
| static traj head -> tail | 4.13 -> 3.55 | 4.27 -> 3.49   | -0.05 (1.5%)  |
| **dyn dyn head -> tail** | **0.236 -> 0.225** | **0.239 -> 0.223** | **-0.002 (0.9%)** |

DCA gives a ~1% marginal improvement on dynamic loss at this training scale —
below noise.  The architectural plumbing works (sub-A through sub-D smoke
tests all green; DCA fires exactly when `dyn_obs_tokens` is supplied,
gradient flows back to tokens), but the **side channel is not yet pulling
its weight** at our current dataset size (450 train sequences, 50%
sampling -> ~225 dynamic batches per epoch effectively).

### Hypotheses for the lack of separation, ranked by likelihood

1. **Dataset starvation.** DCA parameters (205 K) only receive gradient
   on dynamic batches; that's ~50 % of training steps.  In 5 k iter the
   DCA layers see ~2.5 k effective updates from random init -- not enough
   to learn obstacle-attention patterns.
2. **Token redundancy with depth.** The depth image already encodes
   obstacle locations.  Adding the same info via tokens may be redundant
   unless the network can decode "this is the same ball" cross-frame, which
   stage-3.2 single-frame forward cannot do.
3. **World-frame tokens without yaw rotation.** Network has to learn the
   yaw transform implicitly; trainable but burns capacity.
4. **Static safety_loss pollution.** Dynamic batches use a random static
   `map_idx` so YOPOLoss.safety_loss queries an ESDF that doesn't match
   the actual dynamic scene; this is a deliberate stage-3.1 shortcut but
   may be noising up the gradient signal.

### Decision: ship stage-3.2 as-is, document negative result, do not invest more here

The architecture is correct and reusable.  Future work that wants to test
"path-b really helps" should first address (1) by 4-8× more dynamic data,
or pivot to **path a (full K-frame forward)** which has a much higher
theoretical ceiling because the network can learn obstacle motion from
the depth sequence directly.

In the paper, stage-3.1 is the headline architecture; stage-3.2 sits as
an ablation table row showing "explicit GT token side channel adds <1%
at this scale".

## Stage-4 ablation (`v0.4-ablation`)

`scripts/run_stage4_ablation.py` mutates `cfg._data` in-place between
runs and instantiates a fresh `YopoTrainer` for each row, so a single
invocation produces a head-to-head table.  The five rows are additive:

| Row | reVAE | DCA | dyn_ratio | lam_dyn | lam_kino |
|---|---|---|---|---|---|
| A baseline-yopo | off | off | 0.0 | 0.0 | 0.0 |
| B +reVAE         | on  | off | 0.0 | 0.0 | 0.0 |
| C +dyn/kino loss | on  | off | 0.5 | 3.0 | 0.5 |
| D +DCA only      | on  | on  | 0.5 | 0.0 | 0.0 |
| E full           | on  | on  | 0.5 | 3.0 | 0.5 |

### Results (batch=16, lr=1.5e-4, seed=42)

Tail-window means (last 10% of steps).  `nan` means the row never
produced dynamic batches.  Full CSVs with `dyn_kino_tail`, `n_dyn`/
`n_stat`, wall-clock under `results/`.

**5 k-iter run** (`results/ablation_5k.csv`, ~13 min total):

| Row | total | stat_traj | dyn_traj | dyn_dyn | reVAE |
|---|---:|---:|---:|---:|---:|
| A baseline-yopo | 3.84 | 3.50 | nan  | nan  | 0.00  |
| B +reVAE         | 3.85 | 3.48 | nan  | nan  | 0.036 |
| C +dyn/kino loss | 4.18 | 3.54 | 3.71 | **0.224** | 0.036 |
| D +DCA only      | 4.09 | 3.55 | 3.72 | 0.00 | 0.035 |
| E full           | 4.20 | 3.55 | 3.72 | **0.226** | 0.036 |

**2 k-iter run** (`results/ablation.csv`, ~5 min total) -- same trend,
included as a fast-iteration reproduction target:

| Row | total | stat_traj | dyn_traj | dyn_dyn | reVAE |
|---|---:|---:|---:|---:|---:|
| A baseline-yopo | 4.16 | 3.61 | nan  | nan  | 0.00 |
| B +reVAE         | 4.15 | 3.62 | nan  | nan  | 0.044 |
| C +dyn/kino loss | 4.49 | 3.74 | 3.82 | **0.23** | 0.047 |
| D +DCA only      | 4.33 | 3.64 | 3.83 | 0.00 | 0.044 |
| E full           | 4.43 | 3.64 | 3.83 | **0.22** | 0.045 |

### Findings (consistent at 2 k and 5 k iter)

1. **reVAE alone (B vs A) is a wash** at single-frame: 5 k Δtotal =
   +0.003, Δstat_traj = -0.014 — within noise.  reVAE only pays off
   when the temporal forward (path a) consumes its posterior across
   frames, which we don't yet do.
2. **Dyn/kino loss (C vs B)** raises total loss (5 k: 3.85 → 4.18)
   because of the added penalty term, but introduces the only
   meaningful dynamic collision signal (`dyn_dyn` 0.224 vs `nan`).
   This is the real contribution of stage-3.1.
3. **DCA without dyn-loss signal (D vs B)** is marginally worse on
   `total` (5 k: 4.09 vs 3.85) — extra parameters, no gradient
   direction beyond shared static traj/score supervision.  Confirms
   DCA needs a loss to learn anything.
4. **DCA on top of dyn-loss (E vs C)** moves `dyn_dyn` from 0.224 to
   0.226 at 5 k (and 0.23 → 0.22 at 2 k) — a ~1 % oscillation around
   noise.  Consistent with the stage-3.2 5 k-iter A/B (≈1 % gain).
   The gap does **not** widen at 5 k vs 2 k, so the side channel is
   not just under-trained at 2 k — it really isn't pulling its weight
   at current dataset scale.
5. `dyn_kino_tail` is essentially 0 across every row that runs the
   kinodynamic loss (0.0001 at 5 k, exact 0 at 2 k) — the configured
   envelope (`v_max=8`, `a_max=10`, `j_max=30`) is loose enough that
   one-waypoint predictions almost never trigger it.  The loss is
   wired correctly but inactive at these defaults; the envelope only
   becomes binding for multi-waypoint trajectories (path a).

### Implication for the paper

- Headline: stage-3.1 dyn/kino loss adds measurable dynamic-collision
  signal with a small static-traj cost.
- Stage-3.2 DCA is an architectural ablation row that documents "naive
  side channel of GT obstacle tokens does not help at this data scale".
- Reproduction:
  - 2 k (~5 min):  `python scripts/run_stage4_ablation.py --steps 2000`
  - 5 k (~13 min): `python scripts/run_stage4_ablation.py --steps 5000 --out results/ablation_5k.csv`

## Stage-3.4 K-frame temporal forward (path a, `v0.3.4-temporal`)

Per `docs/stage_3_4_design_cn.md` Option A: depth input becomes
(B, K, 1, H, W); reVAE encodes all K frames; `TemporalAggregator`
(nn.GRU, 99 K params) reduces the K-step latent sequence to a single
embedding that takes the broadcast slot stage-1's z used to occupy.
**The YopoHead anchor grid and every stage-3.1 loss are unchanged.**

### A/B on v2 dataset (5 k iter, batch=16, seed=42)

| Metric | F temporal off | G temporal on | Δ |
|---|---:|---:|---:|
| total tail | 4.261 | 4.187 | -0.074 (-1.7 %) |
| stat_traj tail | 3.544 | 3.559 | +0.015 (+0.4 %) |
| dyn_traj tail | 3.762 | 3.770 | +0.008 (+0.2 %) |
| **dyn_dyn tail** | **0.2366** | **0.2343** | **-0.0023 (-1.0 %)** |
| dyn_kino tail | 0.0001 | 0.0001 | 0 |
| reVAE tail | 0.0355 | 0.0427 | +0.0072 (+20 %) |
| wall_s | 145.4 | 219.1 | **+51 %** |

Full CSV at `results/ablation_stage_3_4.csv`.

### Verdict: another <1 % negative result, but for a *different* reason

The path-a K-frame forward improves `dyn_dyn` by **1.0 %** — same order
as the stage-3.2 DCA side channel's ~1 % (on v1) and its v2 retry
(also ~1 %, row F vs E_full on v2).  **Three independent architectural
upgrades all converge at the same ~1 % ceiling.**

The §1 success criterion (≥5 % dyn_dyn drop) was not met.  But path-a
fails *differently* from path-b:

- Path-b (DCA) hand-feeds the network GT obstacle tokens; if the
  network couldn't use them at 500-2000 seq, more architecture won't
  help.
- Path-a (temporal) lets the network *find* the motion signal itself
  from the K-frame depth sequence.  This is a strictly more flexible
  formulation that subsumes path-b's information content.

Both failing at ~1 % on independent mechanisms strongly suggests the
bottleneck is **dataset scale, not architecture**.  All three temporal
papers surveyed in `docs/stage_3_4_research_cn.md` used at least 10×
more data (UCF101: 13 k clips; DiffPhysDrone: unbounded RL).

### Compute cost

K-frame temporal adds **+51 % wall** per training step (10× more reVAE
encodes, partially amortised by the GPU keeping the backbone+head warm
across the same step).  At inference, the same K=10 stateless path
would multiply the per-frame latency by ≈10× over stage-3.1.  Stage-5
deployment should re-investigate the DiffPhysDrone stateful-GRU
pattern (single-frame encode + persistent hidden) before profiling on
Jetson.

### Implication for the paper

- Path-a (temporal) goes into the ablation table as a third row showing
  "architectural upgrade does not separate from baseline at this dataset
  scale".  Combined with paths b and c this paints a clear "next
  experiment" arrow: scale dynamic data to 10 k+ sequences.
- Headline contribution stays stage-3.1 (dyn/kino loss).  Stage-3.2 and
  Stage-3.4 are the two ablation rows.
- Reproduction:
  - `python scripts/run_stage4_ablation.py --steps 5000 --only F_v2_temporal_off,G_v2_temporal_on --out results/ablation_stage_3_4.csv`

## Build/run commands at a glance

```bash
# every session
bash scripts/preflight.sh

# build the ROS packages (one-time per environment)
cd Controller && catkin_make --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cd ..
cd Simulator  && catkin_make --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cd ..

# bake the static dataset (rosrun pipeline, see README Quick Start)
# bake the dynamic dataset
cd Simulator
./devel/lib/sensor_simulator/dataset_generator --config src/config/config_dynamic.yaml

# train (stage-1 default; static dataset only)
cd YOPO && python train_yopo.py

# smoke checks
python scripts/smoke_stage1.py                  # module shapes
python scripts/smoke_stage1_integration.py      # forward + backward
python scripts/smoke_stage1_train.py            # 100-iter on real data
python scripts/smoke_stage2_2e.py               # dataset --dynamic + static regression
python tools/verify_dynamic_render.py dataset_dynamic/v1/env_0008/seq_0007
```
