# REACT

**R**ecurrent **E**nvironment-**A**ware **C**ollision-avoidance **T**rajectory planner.

REACT is a learning-based one-stage planner for agile drone flight in **dynamic** environments, built on top of the [YOPO](https://github.com/TJU-Aerial-Robotics/YOPO) baseline with a recurrent temporal architecture inspired by PEMTRS (RA-L 2026).

## Research Goal

Extend YOPO's single-frame, one-stage planning paradigm with:
- a recurrent temporal backbone (residual VAE + Transformer region selector + GRU decoder) that aggregates a sliding window of past depth frames;
- a dynamic-obstacle-aware loss family (relative-motion-reshaped distance field + kinodynamic consistency);
- training scenarios augmented with moving spheres, swinging gates and pedestrian-like obstacles inside Flightmare.

The target metric is the dynamic-scenario success rate (paper target: >= 85%), while keeping single-frame inference latency below 10 ms on Jetson-class hardware.

## Repository Layout

This repository keeps the original YOPO outer structure (with sibling ROS packages for the controller and simulator) and adds REACT-specific modules under `YOPO/`:

```
REACT/
  Controller/             # ROS controller package (from YOPO, unchanged in stage 1)
  Simulator/              # ROS simulator package (extended for dynamic obstacles)
  YOPO/
    train_yopo.py         # training entry (will be extended with REACT losses)
    policy/               # network backbones, dataset, trainer
    loss/                 # smoothness / safety / goal (YOPO original) + REACT losses
    config/               # YAML configs
  scripts/
    preflight.sh          # environment self-check (run this first every session)
  docs/
```

## Quick Start

```bash
# 1) environment self-check
bash scripts/preflight.sh

# 2) follow CiteYopo/REACT-implementation-guide for stage-by-stage development
```

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | baseline + tagged releases only |
| `stage-1-pemtrs-port` | PEMTRS temporal architecture port |
| `stage-2-dynamic-sim` | dynamic-obstacle Flightmare scenes |
| `stage-3-dynamic-upgrade` | dynamic-aware losses + upgrade modules |
| `stage-4-ablation` | component ablations |
| `stage-5-deploy` | on-board deployment |

Each stage is developed on its own branch; merges to `main` are gated by review.

## Acknowledgments

REACT is built upon the open-source **YOPO** codebase by TJU-Aerial-Robotics ([paper](https://ieeexplore.ieee.org/document/10528860), [repo](https://github.com/TJU-Aerial-Robotics/YOPO)) -- all original copyrights and licenses are preserved (see `LICENSE`).

The temporal architecture follows the **PEMTRS** method (IEEE RA-L 2026, Nankai University), reimplemented from the published method section as the original source code is not public.

## License

This repository inherits the YOPO upstream license (see `LICENSE`).
