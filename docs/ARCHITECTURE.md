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

## "Unwired" modules

Stage 1's `FrameBuffer`, `TemporalRegionSelector`, and `GRUDecoder` are on
disk and unit-tested, but `YopoNetwork.forward(depth, obs)` is still a
single-frame anchor-grid pipeline. Stage 3 will choose how to wire them:

- **(c) loss-only**: just add `motion_reshaped_collision_loss` +
  `kinodynamic_loss` to the total; forward stays single-frame. Cheapest;
  network can only react to current-frame ball positions (reactive
  avoidance only). Expected dynamic success rate: 50-70%.
- **(b) dyn-obs side-channel**: `forward(depth, obs, dyn_obs_tokens=None)`
  with `dyn_obs_tokens` encoded from `dyn_obs.json` (training) or a runtime
  detector (deployment). DynamicCrossAttention injects the tokens before the
  head. Anchor grid head stays. Expected: 75-85%.
- **(a) full K-frame forward**: `forward(depth_seq, ...)` running ReVAE per
  frame, feeding Selector + GRUDecoder + DynamicCrossAttention. Head
  rebuilt around `n_anchors` flat. Maximum capacity; head no longer V×H
  grid. Expected: 80-90% but ~1 week of refactor.

Pick decided when stage-3 opens, with stage-2's 1800 in-view-ball frames as
the data backing. See the project guide §6.1 stage-3 for the smoke tests
that any choice has to pass.

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
