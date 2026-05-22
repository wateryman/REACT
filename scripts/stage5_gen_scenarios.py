"""Stage-5.B.0 — generate 100 dynamic-obstacle scenarios for closed-loop SR eval.

Each scenario YAML pins:
  - drone start (pos, yaw)
  - goal (world-frame xyz)
  - timeout (s)
  - N balls, each with (pos_init, vel, radius)

Distribution matches the v3 bake (Simulator/src/config/config_dynamic.yaml):
  count    3..8
  speed    1..5 m/s
  radius   0.3..0.6 m
  bbox xy  [-15, 15]
  bbox z   [0.5, 6]

Drone-side:
  start xy in [-10, 10]^2, z in [1.0, 3.0]
  yaw direction in [-30deg, 30deg] from +X
  goal at start + 20m along yaw, z in [1.0, 3.0]
  timeout 10 s (allows 2 m/s straight-line plus margin)

Seed: scenario index ensures deterministic regeneration / cross-config A/B.

Output: tools/eval_scenarios/v3_dyn_100/scenario_<NNN>.yaml
"""
import math
import os
import sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REACT_ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(REACT_ROOT, "tools", "eval_scenarios", "v3_dyn_100")

# Bake-aligned distribution params.
COUNT_MIN, COUNT_MAX = 3, 8
SPEED_MIN, SPEED_MAX = 1.0, 5.0
RADIUS_MIN, RADIUS_MAX = 0.3, 0.6
BBOX_XY = 15.0
BBOX_Z_LO, BBOX_Z_HI = 0.5, 6.0

# Drone-side scenario params.
DRONE_XY = 10.0          # start in [-10, 10]^2
DRONE_Z_LO, DRONE_Z_HI = 1.0, 3.0
GOAL_DIST = 20.0
YAW_DEG_RANGE = 30.0
TIMEOUT_S = 10.0

N_SCENARIOS = 100


def sample_one(rng):
    """Sample one scenario as a plain dict (YAML-friendly)."""
    # Drone start.  z slightly above ground.
    sx = float(rng.uniform(-DRONE_XY, DRONE_XY))
    sy = float(rng.uniform(-DRONE_XY, DRONE_XY))
    sz = float(rng.uniform(DRONE_Z_LO, DRONE_Z_HI))
    yaw_deg = float(rng.uniform(-YAW_DEG_RANGE, YAW_DEG_RANGE))
    yaw_rad = math.radians(yaw_deg)

    # Goal: GOAL_DIST forward along yaw, separate z roll.
    gx = sx + GOAL_DIST * math.cos(yaw_rad)
    gy = sy + GOAL_DIST * math.sin(yaw_rad)
    gz = float(rng.uniform(DRONE_Z_LO, DRONE_Z_HI))

    # Balls.  Drawn anywhere in the bake's bbox; some will be off-screen at
    # start and that's fine -- this matches the bake's distribution.
    n_balls = int(rng.integers(COUNT_MIN, COUNT_MAX + 1))   # inclusive
    balls = []
    for _ in range(n_balls):
        px = float(rng.uniform(-BBOX_XY, BBOX_XY))
        py = float(rng.uniform(-BBOX_XY, BBOX_XY))
        pz = float(rng.uniform(BBOX_Z_LO, BBOX_Z_HI))
        # Isotropic velocity direction; magnitude in [SPEED_MIN, SPEED_MAX].
        theta = float(rng.uniform(0.0, 2 * math.pi))
        phi = float(rng.uniform(-math.pi / 2, math.pi / 2))
        speed = float(rng.uniform(SPEED_MIN, SPEED_MAX))
        vx = speed * math.cos(phi) * math.cos(theta)
        vy = speed * math.cos(phi) * math.sin(theta)
        vz = speed * math.sin(phi)
        r = float(rng.uniform(RADIUS_MIN, RADIUS_MAX))
        balls.append({
            "pos":    [round(px, 3), round(py, 3), round(pz, 3)],
            "vel":    [round(vx, 3), round(vy, 3), round(vz, 3)],
            "radius": round(r, 3),
        })

    return {
        "drone": {
            "start_pos": [round(sx, 3), round(sy, 3), round(sz, 3)],
            "start_yaw_deg": round(yaw_deg, 2),
            "goal_pos":  [round(gx, 3), round(gy, 3), round(gz, 3)],
        },
        "timeout_s": TIMEOUT_S,
        "balls": balls,
        "bbox": {
            "xy_half":   BBOX_XY,
            "z_lo":      BBOX_Z_LO,
            "z_hi":      BBOX_Z_HI,
        },
        "dt": 0.0333,        # 30 Hz, matches bake
    }


def main():
    import numpy as np

    os.makedirs(OUT_DIR, exist_ok=True)
    n_balls_hist = [0] * (COUNT_MAX + 1)
    speed_acc = 0.0
    speed_n = 0

    for idx in range(N_SCENARIOS):
        rng = np.random.default_rng(seed=10_000 + idx)   # avoid v1/v2/v3 seeds (7,11,13)
        sc = sample_one(rng)
        path = os.path.join(OUT_DIR, f"scenario_{idx:03d}.yaml")
        with open(path, "w") as fh:
            yaml.safe_dump(sc, fh, sort_keys=False)

        n_balls_hist[len(sc["balls"])] += 1
        for b in sc["balls"]:
            speed_acc += math.sqrt(sum(v * v for v in b["vel"]))
            speed_n += 1

    print(f"== stage-5.B.0: wrote {N_SCENARIOS} scenarios to {OUT_DIR} ==")
    print(f"  ball-count histogram (3..8): "
          f"{n_balls_hist[3]}, {n_balls_hist[4]}, {n_balls_hist[5]}, "
          f"{n_balls_hist[6]}, {n_balls_hist[7]}, {n_balls_hist[8]}")
    print(f"  mean ball speed: {speed_acc/speed_n:.3f} m/s (target ~3 m/s)")
    print(f"  total balls across scenarios: {speed_n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
