# REACT

**R**ecurrent **E**nvironment-**A**ware **C**ollision-avoidance **T**rajectory planner.

REACT is a learning-based one-stage planner for agile drone flight in **dynamic** environments, built on top of the [YOPO](https://github.com/TJU-Aerial-Robotics/YOPO) baseline with a recurrent temporal architecture inspired by PEMTRS (RA-L 2026).

## Research Goal

Extend YOPO's single-frame, one-stage planning paradigm with:
- a recurrent temporal backbone (residual VAE + Transformer region selector + GRU decoder) that aggregates a sliding window of past depth frames;
- a dynamic-obstacle-aware loss family (relative-motion-reshaped distance field + kinodynamic consistency);
- training scenarios augmented with moving spheres, swinging gates and pedestrian-like obstacles, baked offline through the YOPO CUDA raycaster (D-3 path -- see `CiteYopo/REACT-实施指南.md` §5 version history for why we ditched the earlier Flightmare-online and Flightmare-offline plans).

The target metric is the dynamic-scenario success rate (paper target: >= 85%), while keeping single-frame inference latency below 10 ms on Jetson-class hardware.

## Repository Layout

This repository keeps the original YOPO outer structure (with sibling ROS packages for the controller and simulator) and adds REACT-specific modules under `YOPO/`:

```
REACT/
  Controller/             # ROS controller package (from YOPO, unchanged)
  Simulator/              # ROS simulator extended with dynamic-sphere ray-sphere
                          # intersection in the CUDA raycaster (stage-2 D-3)
  YOPO/                   # name kept to mirror upstream YOPO's layout so
                          # `git diff upstream` stays auditable; the *contents*
                          # are REACT's network/loss/dataset code (see docs/ARCHITECTURE.md)
    train_yopo.py         # training entry (extended in stage-1 with reVAE loss)
    policy/               # network backbones, dataset, trainer
    loss/                 # smoothness / safety / goal (YOPO original) + REACT losses
    config/               # YAML configs (traj_opt.yaml static + config used for the simulator)
  scripts/
    preflight.sh          # environment self-check (run this first every session)
    smoke_stage1_*.py     # stage-1 module + integration + 100-iter training smoke
    smoke_stage2_2e.py    # stage-2 YOPODataset dynamic-mode smoke + static regression
    run_one_epoch.py      # epoch runner used to validate stage-1 baseline
    extract_tb.py         # tensorboard scalar -> ASCII sparkline summary
  tools/
    verify_dynamic_render.py  # stage-2 geometric verification gate (D-3)
  docs/
    ARCHITECTURE.md       # three-package layout + data flow + module map
    dyn_sample_*.png      # sample depth-image visualizations
```

## Quick Start

```bash
# 1) environment self-check
bash scripts/preflight.sh

# 2) follow CiteYopo/REACT-实施指南.md for stage-by-stage development
```

## Current Status

| Stage | State | What landed |
|---|---|---|
| 1 — PEMTRS port | ✅ merged to `main`, tag `v0.1-pemtrs-port` | ReVAE wired into YopoNetwork as auxiliary encoder; reVAE loss in trainer; 100-iter gate PASS; 1-epoch eval err parity with train |
| 2 — Dynamic dataset (D-3) | ✅ merged to `main`, tag `v0.2-cuda-dynamic` | CUDA raycaster gains `DynSphere` + `ray_sphere_depth`; `dataset_generator` dynamic mode bakes 500-seq production set; `verify_dynamic_render.py` go/no-go gate; `YOPODataset --dynamic` switch |
| 3.1 — Dynamic loss (path c) | ✅ merged to `main`, tag `v0.3.1-loss-only` | `motion_reshaped_collision_loss` + `kinodynamic_loss`; YOPOLoss wrappers; trainer mixed sampling 50/50; 1k-iter 5/5 gates PASS (static traj 4.71->3.87, dyn 0.27->0.23) |
| 3.2 — Side channel (path b) | ✅ merged to `main`, tag `v0.3.2-side-channel` | DynObsEncoder + DynamicCrossAttention wired into YopoNetwork.forward; trainer builds drone-relative tokens. **Architecture works but 5k-iter A/B shows only ~1% improvement over path c at current dataset size — see docs/ARCHITECTURE.md for the full analysis.** |
| 3.4 — K-frame forward (path a) | ⏸ pending | Open question: invest in full temporal forward (highest ceiling, ~1 week) or pivot to deployment? |
| 4 — Ablation | ✅ merged to `main`, tag `v0.4-ablation` | `scripts/run_stage4_ablation.py` runs 5 additive configs (A baseline → E full). 2k-iter table in `results/ablation.csv`. Headline: dyn/kino loss adds dyn-collision signal; DCA side-channel ~1% noise. See `docs/ARCHITECTURE.md` Stage-4 section |
| 5 — Deployment | ⏸ pending | |

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | baseline + tagged releases only |
| `stage-1-pemtrs-port` | PEMTRS temporal architecture port (merged, kept for history) |
| `stage-1-tooling` | epoch runner + tfevents extractor utilities |
| `stage-2-cuda-dynamic` | YOPO CUDA raycaster dynamic-sphere extension + dynamic dataset baker (D-3) |
| `stage-3-dynamic-upgrade` (planned) | dynamic-aware losses + temporal-aware forward (architecture TBD) |
| `stage-4-ablation` | component ablations (5-row table) — merged |
| `stage-5-deploy` (planned) | on-board deployment |

Each stage is developed on its own branch; merges to `main` are gated by review.

## Acknowledgments

REACT is built upon the open-source **YOPO** codebase by TJU-Aerial-Robotics ([paper](https://ieeexplore.ieee.org/document/10528860), [repo](https://github.com/TJU-Aerial-Robotics/YOPO)) -- all original copyrights and licenses are preserved (see `LICENSE`).

The temporal architecture follows the **PEMTRS** method (IEEE RA-L 2026, Nankai University), reimplemented from the published method section as the original source code is not public.

## License

This repository inherits the YOPO upstream license (see `LICENSE`).
