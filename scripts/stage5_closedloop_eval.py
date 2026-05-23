"""Stage-5.B.2 — closed-loop SR evaluation driver.

Connects to a running sensor_simulator ROS node (stage-5.B.1 dynamic-sphere
support required), drives 100 scenarios from tools/eval_scenarios/v3_dyn_100/,
and measures closed-loop success rate.

Two integrator options via --dynamics:
  point_mass : Pass-1, bounded double integrator (planner SR upper bound).
  poly5      : Pass-2, 5th-order polynomial per axis -- matches the deployed
               Poly5Solver in YOPO/test_yopo_ros.py.  Use this number for
               anything you cite alongside the real-flight target.

Pipeline per scenario, 30 Hz:
  1. Step balls in Python (linear motion + bbox reflection).
  2. Publish drone /sim/odom and /sim/dyn_obs.
  3. Wait for fresh /depth_image (rolling-latest with timeout).
  4. Run YopoNetwork.inference -> per-anchor (endstate, score); argmin score.
  5. Convert best endstate (body-frame pos/vel/acc) to world frame.
  6. Step drone via simple "follow endstate velocity over dt" model.
  7. Termination: goal_reached | static_collision | dyn_collision | timeout.

Pre-flight:
  - Running roscore.
  - rosrun sensor_simulator sensor_simulator (stage-5.B.1 build).

Run:
  python scripts/stage5_closedloop_eval.py \\
      --ckpt YOPO/saved/YOPO_1/epoch50.pth \\
      --label baseline \\
      --scenarios tools/eval_scenarios/v3_dyn_100 \\
      --out results/stage5_closedloop_baseline.csv

The driver writes a CSV with one row per scenario:
  idx, terminate_reason, time_to_goal_s, min_clearance_m, n_steps, mean_speed
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REACT_ROOT = os.path.dirname(HERE)
YOPO_DIR = os.path.join(REACT_ROOT, "YOPO")
sys.path.insert(0, YOPO_DIR)

from config.config import cfg
from policy.yopo_network import YopoNetwork
from policy.state_transform import StateTransform
from policy.poly_solver import Poly5Solver

import rospy
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R


# ----------- helpers ----------------------------------------------------------

def yaw_to_quat(yaw_rad):
    """Body +X facing world +X when yaw=0.  Returns (x, y, z, w)."""
    return R.from_euler("ZYX", [yaw_rad, 0, 0]).as_quat()


def step_balls(balls, dt, bbox):
    """In-place ball step with axis-aligned reflection at the bbox.
    balls: list[ {pos: np.array(3), vel: np.array(3), radius: float} ]
    bbox:  {xy_half, z_lo, z_hi}
    """
    for b in balls:
        b["pos"] = b["pos"] + b["vel"] * dt
        for axis, lo, hi in ((0, -bbox["xy_half"], bbox["xy_half"]),
                              (1, -bbox["xy_half"], bbox["xy_half"]),
                              (2, bbox["z_lo"], bbox["z_hi"])):
            if b["pos"][axis] > hi:
                b["pos"][axis] = 2 * hi - b["pos"][axis]
                b["vel"][axis] = -b["vel"][axis]
            elif b["pos"][axis] < lo:
                b["pos"][axis] = 2 * lo - b["pos"][axis]
                b["vel"][axis] = -b["vel"][axis]


def min_clearance_to_balls(drone_pos, balls):
    """Surface-to-surface clearance, ignoring negative values.  drone radius
    is folded into the safety check by the caller with drone_radius."""
    if not balls:
        return float("inf")
    d_min = float("inf")
    for b in balls:
        d = float(np.linalg.norm(drone_pos - b["pos"]) - b["radius"])
        if d < d_min:
            d_min = d
    return d_min


# ----------- ROS wrapper ------------------------------------------------------

class SimBridge:
    """Holds ROS pub/sub state.  init_node() must be called once per process."""
    def __init__(self):
        self.bridge = CvBridge()
        self.latest_depth = None
        self.latest_depth_t = 0.0
        self.pub_odom = rospy.Publisher("/sim/odom", Odometry, queue_size=1)
        self.pub_dyn  = rospy.Publisher("/sim/dyn_obs", PoseArray, queue_size=1)
        rospy.Subscriber("/depth_image", Image, self._depth_cb,
                          queue_size=1, tcp_nodelay=True)

    def _depth_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.latest_depth = img
        self.latest_depth_t = time.time()

    def publish_state(self, pos, quat_xyzw, balls):
        odom = Odometry()
        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])
        odom.pose.pose.orientation.x = float(quat_xyzw[0])
        odom.pose.pose.orientation.y = float(quat_xyzw[1])
        odom.pose.pose.orientation.z = float(quat_xyzw[2])
        odom.pose.pose.orientation.w = float(quat_xyzw[3])
        self.pub_odom.publish(odom)

        arr = PoseArray()
        for b in balls:
            p = Pose()
            p.position.x = float(b["pos"][0])
            p.position.y = float(b["pos"][1])
            p.position.z = float(b["pos"][2])
            p.orientation.w = float(b["radius"])    # radius hack
            arr.poses.append(p)
        self.pub_dyn.publish(arr)

    def wait_for_fresh_depth(self, t0, timeout=0.2):
        """Block until latest_depth_t > t0 or timeout."""
        t_end = time.time() + timeout
        while time.time() < t_end:
            if self.latest_depth_t > t0:
                return self.latest_depth
            time.sleep(0.005)
        return self.latest_depth   # stale fallback


# ----------- planner wrapper --------------------------------------------------

class Planner:
    """Wraps YopoNetwork with depth pre-processing and best-anchor selection."""
    def __init__(self, ckpt_path, use_revae, use_dca, use_temporal, K=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_temporal = use_temporal
        self.K = K
        # Match the checkpoint's architecture
        self.net = YopoNetwork(use_revae=use_revae,
                                revae_latent=128,
                                use_dca=use_dca, dca_n_heads=1,
                                use_temporal=use_temporal,
                                temporal_hidden=128 if use_temporal else None)
        state = torch.load(ckpt_path, weights_only=True, map_location=self.device)
        # Allow loading a baseline ckpt missing reVAE/DCA/temporal keys.
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[warn] non-strict load: missing={len(missing)}, "
                  f"unexpected={len(unexpected)}")
        self.net.to(self.device).eval()
        self.state_transform = StateTransform()
        self.height = int(cfg["image_height"])
        self.width  = int(cfg["image_width"])
        self.max_dis = 20.0
        self.depth_buffer = []   # for temporal mode

    def _preprocess_depth(self, depth_raw):
        """ROS 32FC1 (meters) or 16UC1 (mm) -> (1, 1, H, W) normalized."""
        if depth_raw.dtype == np.uint16:
            d = depth_raw.astype(np.float32) / 1000.0
        else:
            d = depth_raw.astype(np.float32)
        if d.shape[0] != self.height or d.shape[1] != self.width:
            d = cv2.resize(d, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        d = np.minimum(d, self.max_dis) / self.max_dis
        # nan / too-close inpaint (match test_yopo_ros.py)
        nan_mask = np.isnan(d) | (d < 0.04 / self.max_dis)
        if nan_mask.any():
            interp = cv2.inpaint(np.uint8(d * 255), np.uint8(nan_mask), 1,
                                  cv2.INPAINT_NS)
            d = interp.astype(np.float32) / 255.0
        return d.reshape(1, 1, self.height, self.width)

    @torch.inference_mode()
    def plan(self, depth_raw, pos, rot_wc, vel_w, acc_w, goal_w,
              anchor_filter: bool = True):
        depth = self._preprocess_depth(depth_raw)
        depth_t = torch.from_numpy(depth).to(self.device, non_blocking=True)

        if self.use_temporal:
            self.depth_buffer.append(depth_t)
            if len(self.depth_buffer) > self.K:
                self.depth_buffer.pop(0)
            while len(self.depth_buffer) < self.K:
                # warm-up: pad with the first available frame
                self.depth_buffer.insert(0, self.depth_buffer[0])
            # stack to (1, K, 1, H, W)
            depth_t = torch.stack(self.depth_buffer, dim=1)[0:1].squeeze(2)
            depth_t = depth_t.unsqueeze(2) if depth_t.dim() == 4 else depth_t
            # Actually we want (1, K, 1, H, W).  Each frame is (1, 1, H, W).
            # Re-stack cleanly:
            depth_t = torch.stack([f[0] for f in self.depth_buffer], dim=0)  # (K,1,H,W)
            depth_t = depth_t.unsqueeze(0)                                    # (1,K,1,H,W)

        # Body-frame obs vector: vel(3) + acc(3) + goal(3), all in body
        rot_cw = rot_wc.T
        goal_b = rot_cw @ (goal_w - pos)
        vel_b  = rot_cw @ vel_w
        acc_b  = rot_cw @ acc_w
        obs = np.concatenate([vel_b, acc_b, goal_b]).astype(np.float32)
        obs_t = torch.from_numpy(obs[None, :]).to(self.device)

        endstate, score, _recon, _mu, _logvar = self.net.inference(depth_t, obs_t)
        # endstate: (1, 9, V, H), score: (1, V, H)
        e = endstate[0].cpu().numpy()       # (9, V, H)
        Vh, Wh = endstate.shape[-2], endstate.shape[-1]
        s = score[0].cpu().numpy().reshape(-1)
        # 🟦 stage-5.B hack: the trained model exploits a gap in safety_loss
        # (ESDF is undefined below z=0 because the static point cloud has no
        # ground plane), and consistently scores "look down" anchors lowest.
        # Filter argmin to anchors whose body-frame end_z > -1.0 m so the
        # drone doesn't dive into the ground.  This is a deploy-side hack
        # to validate the architecture; the long-term fix is a z-floor
        # loss term in the trainer (stage-5.B plan B).
        if anchor_filter:
            end_z_body = e[2].reshape(-1)       # (V*H,)
            z_mask = end_z_body > -1.0
            if z_mask.any():
                s_masked = np.where(z_mask, s, np.inf)
                best = int(np.argmin(s_masked))
            else:
                best = int(np.argmin(s))         # fallback if all below threshold
        else:
            best = int(np.argmin(s))
        v_idx, h_idx = best // Wh, best % Wh
        # Body-frame endstate fields: [px, py, pz, vx, vy, vz, ax, ay, az]
        end_pos_b = e[0:3, v_idx, h_idx]
        end_vel_b = e[3:6, v_idx, h_idx]
        end_acc_b = e[6:9, v_idx, h_idx]
        # World frame
        end_pos_w = pos + rot_wc @ end_pos_b
        end_vel_w = rot_wc @ end_vel_b
        end_acc_w = rot_wc @ end_acc_b
        return {
            "end_pos_w": end_pos_w,
            "end_vel_w": end_vel_w,
            "end_acc_w": end_acc_w,
            "score": float(s[best]),
        }


# ----------- closed-loop runner ----------------------------------------------

def run_scenario(scenario, bridge, planner, args):
    drone_radius = 0.2
    sphere_safety = 0.0          # surface-to-surface < 0 == collision
    static_collision_depth_m = 0.25

    pos = np.array(scenario["drone"]["start_pos"], dtype=np.float32)
    yaw_rad = math.radians(float(scenario["drone"]["start_yaw_deg"]))
    quat = yaw_to_quat(yaw_rad)
    rot_wc = R.from_quat(quat).as_matrix().astype(np.float32)
    goal = np.array(scenario["drone"]["goal_pos"], dtype=np.float32)
    timeout = float(scenario["timeout_s"])
    dt = float(scenario.get("dt", 0.0333))
    bbox = scenario["bbox"]
    balls = [{
        "pos": np.array(b["pos"], dtype=np.float32),
        "vel": np.array(b["vel"], dtype=np.float32),
        "radius": float(b["radius"]),
    } for b in scenario["balls"]]

    vel = np.zeros(3, dtype=np.float32)
    acc = np.zeros(3, dtype=np.float32)
    min_clearance = float("inf")
    speeds = []

    n_steps = int(timeout / dt)
    planner.depth_buffer = []   # reset temporal buffer per scenario

    for step in range(n_steps):
        # 1. step balls
        step_balls(balls, dt, bbox)
        # 2. publish state
        t_pub = time.time()
        bridge.publish_state(pos, quat, balls)
        # 3. wait for fresh depth
        depth = bridge.wait_for_fresh_depth(t_pub, timeout=0.15)
        if depth is None:
            return {"terminate_reason": "no_depth", "step": step,
                    "time_to_goal_s": float("nan"),
                    "min_clearance_m": min_clearance, "n_steps": step,
                    "mean_speed": 0.0}
        # 3b. static-collision check.  Two indicators:
        # (i) center depth patch min < threshold (front-of-drone obstacle)
        # (ii) 🟦 drone world z below ground floor (z < 0.1 m) -- catches
        #      the "model dives below ground" failure mode when the
        #      anchor filter doesn't fully save us.
        h0, w0 = depth.shape[0], depth.shape[1]
        cy0, cx0 = h0 // 2, w0 // 2
        center_patch = depth[cy0 - 2:cy0 + 3, cx0 - 2:cx0 + 3]
        center_min = float(np.nanmin(center_patch))
        ground_floor = 0.1
        if center_min < static_collision_depth_m or pos[2] < ground_floor:
            return {"terminate_reason": "static_collision", "step": step,
                    "time_to_goal_s": float("nan"),
                    "min_clearance_m": min_clearance, "n_steps": step,
                    "mean_speed": float(np.mean(speeds)) if speeds else 0.0}
        # 4. plan
        plan = planner.plan(depth, pos, rot_wc, vel, acc, goal,
                             anchor_filter=not args.no_anchor_filter)
        # 5. step drone state.  Two modes:
        target_pos = plan["end_pos_w"]
        target_vel = plan["end_vel_w"]
        target_acc = plan["end_acc_w"]
        traj_time = float(cfg["sgm_time"])   # ~1.7s endstate horizon

        if args.dynamics == "point_mass":
            # Pass-1: bounded double-integrator.  Cheap upper bound on
            # planner SR; ignores trajectory shape between waypoints.
            v_max = float(cfg["vel_max_train"])
            v_desired = (target_pos - pos) / max(traj_time, 1e-3)
            v_norm = float(np.linalg.norm(v_desired))
            if v_norm > v_max:
                v_desired = v_desired * (v_max / v_norm)
            a_max = float(cfg["acc_max_train"])
            dv = v_desired - vel
            dv_norm = float(np.linalg.norm(dv))
            if dv_norm > a_max * dt:
                dv = dv * (a_max * dt / dv_norm)
            vel = vel + dv
            acc = dv / dt
            pos = pos + vel * dt
        else:
            # Pass-2: Poly5Solver -- match the deployed test_yopo_ros.py
            # path.  3 independent 5th-order polynomials (x, y, z),
            # boundary conditions = (current state) -> (predicted endstate
            # at +traj_time).  Step dt forward and read (p, v, a) at t=dt.
            poly_x = Poly5Solver(pos[0], vel[0], acc[0],
                                  target_pos[0], target_vel[0], target_acc[0],
                                  traj_time)
            poly_y = Poly5Solver(pos[1], vel[1], acc[1],
                                  target_pos[1], target_vel[1], target_acc[1],
                                  traj_time)
            poly_z = Poly5Solver(pos[2], vel[2], acc[2],
                                  target_pos[2], target_vel[2], target_acc[2],
                                  traj_time)
            pos = np.array([poly_x.get_position(dt),
                             poly_y.get_position(dt),
                             poly_z.get_position(dt)], dtype=np.float32)
            vel = np.array([poly_x.get_velocity(dt),
                             poly_y.get_velocity(dt),
                             poly_z.get_velocity(dt)], dtype=np.float32)
            acc = np.array([poly_x.get_acceleration(dt),
                             poly_y.get_acceleration(dt),
                             poly_z.get_acceleration(dt)], dtype=np.float32)
        # 7. update heading to face GOAL (not velocity).  This mirrors
        # YOPO/policy/poly_solver.py::calculate_yaw used by test_yopo_ros.py
        # at deployment: drone yaws toward the goal vector so the body-frame
        # obs.goal stays roughly aligned with body +X.  Without this,
        # planner outputs that lack a lateral component (e.g., the stage-3.1
        # well-trained C1) cause the drone to drift on a constant-yaw line
        # and miss laterally-offset goals.  Yaw-rate is rate-limited.
        speed = float(np.linalg.norm(vel))
        speeds.append(speed)
        goal_dir_xy = goal[0:2] - pos[0:2]
        goal_dist_xy = float(np.linalg.norm(goal_dir_xy))
        if goal_dist_xy > 0.3:
            yaw_target = math.atan2(goal_dir_xy[1], goal_dir_xy[0])
            yaw_err = math.atan2(math.sin(yaw_target - yaw_rad),
                                   math.cos(yaw_target - yaw_rad))
            max_dyaw = 2.0 * dt   # 2 rad/s max yaw rate
            yaw_rad = yaw_rad + max(-max_dyaw, min(max_dyaw, yaw_err))
            quat = yaw_to_quat(yaw_rad)
            rot_wc = R.from_quat(quat).as_matrix().astype(np.float32)
        # 8. ball collision check (after integration so we don't double-count)
        clearance = min_clearance_to_balls(pos, balls) - drone_radius
        if clearance < min_clearance:
            min_clearance = clearance
        if clearance < sphere_safety:
            return {"terminate_reason": "dyn_collision", "step": step,
                    "time_to_goal_s": float("nan"),
                    "min_clearance_m": min_clearance, "n_steps": step,
                    "mean_speed": float(np.mean(speeds))}
        # 9. goal check
        if float(np.linalg.norm(pos - goal)) < 0.8:
            return {"terminate_reason": "goal", "step": step,
                    "time_to_goal_s": (step + 1) * dt,
                    "min_clearance_m": min_clearance, "n_steps": step + 1,
                    "mean_speed": float(np.mean(speeds))}

    return {"terminate_reason": "timeout", "step": n_steps,
            "time_to_goal_s": float("nan"),
            "min_clearance_m": min_clearance, "n_steps": n_steps,
            "mean_speed": float(np.mean(speeds)) if speeds else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to .pth checkpoint")
    ap.add_argument("--label", required=True, help="config label for CSV")
    ap.add_argument("--scenarios", default="tools/eval_scenarios/v3_dyn_100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--use-revae", action="store_true")
    ap.add_argument("--use-dca", action="store_true")
    ap.add_argument("--use-temporal", action="store_true")
    ap.add_argument("--max-scenarios", type=int, default=100)
    ap.add_argument("--no-anchor-filter", action="store_true",
                    help="disable the end_z > -1m anchor filter (Plan A hack).  "
                         "With z_floor-trained models the filter is a no-op; with "
                         "the baseline (untrained on z_floor) the filter clips "
                         "look-down anchors and inflates baseline SR.  Use this "
                         "flag for apples-to-apples comparison without the hack.")
    ap.add_argument("--dynamics", choices=["point_mass", "poly5"],
                    default="point_mass",
                    help="closed-loop integrator: point_mass (Pass-1, fast SR "
                         "upper bound) or poly5 (Pass-2, matches the deployed "
                         "Poly5Solver in test_yopo_ros.py)")
    args = ap.parse_args()

    if not os.path.isabs(args.scenarios):
        args.scenarios = os.path.join(REACT_ROOT, args.scenarios)
    if not os.path.isabs(args.out):
        args.out = os.path.join(REACT_ROOT, args.out)

    sc_paths = sorted([os.path.join(args.scenarios, f)
                        for f in os.listdir(args.scenarios)
                        if f.startswith("scenario_") and f.endswith(".yaml")])
    sc_paths = sc_paths[:args.max_scenarios]
    print(f"== stage-5.B.2 closed-loop eval | label={args.label} ==")
    print(f"   checkpoint:   {args.ckpt}")
    print(f"   scenarios:    {len(sc_paths)}")
    print(f"   architecture: revae={args.use_revae} dca={args.use_dca} "
          f"temporal={args.use_temporal}")
    print(f"   dynamics:     {args.dynamics}")
    print(f"   anchor_filter:{'OFF' if args.no_anchor_filter else 'ON (default)'}")

    rospy.init_node("stage5_closedloop", anonymous=True)
    bridge = SimBridge()
    planner = Planner(args.ckpt,
                      use_revae=args.use_revae,
                      use_dca=args.use_dca,
                      use_temporal=args.use_temporal)
    time.sleep(1.0)   # let the sim node settle

    results = []
    for idx, sp in enumerate(sc_paths):
        with open(sp) as fh:
            sc = yaml.safe_load(fh)
        t0 = time.time()
        r = run_scenario(sc, bridge, planner, args)
        r["idx"] = idx
        r["wall_s"] = round(time.time() - t0, 2)
        results.append(r)
        print(f"  [{idx:3d}] {r['terminate_reason']:18s}  "
              f"min_clear={r['min_clearance_m']:6.3f} m  "
              f"steps={r['n_steps']:3d}  wall={r['wall_s']:5.2f}s  "
              f"v_mean={r['mean_speed']:.2f} m/s")

    # Summary
    n = len(results)
    n_goal = sum(1 for r in results if r["terminate_reason"] == "goal")
    n_dyn  = sum(1 for r in results if r["terminate_reason"] == "dyn_collision")
    n_stat = sum(1 for r in results if r["terminate_reason"] == "static_collision")
    n_to   = sum(1 for r in results if r["terminate_reason"] == "timeout")
    print()
    print(f"=== {args.label} SR summary ({n} scenarios) ===")
    print(f"  goal_reached     : {n_goal:3d}  ({100*n_goal/n:.1f}%)")
    print(f"  dyn_collision    : {n_dyn:3d}  ({100*n_dyn/n:.1f}%)")
    print(f"  static_collision : {n_stat:3d}  ({100*n_stat/n:.1f}%)")
    print(f"  timeout          : {n_to:3d}  ({100*n_to/n:.1f}%)")

    # CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = ["idx", "terminate_reason", "time_to_goal_s",
                   "min_clearance_m", "n_steps", "mean_speed", "wall_s"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
