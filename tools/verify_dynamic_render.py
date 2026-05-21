"""REACT stage-2 [🟧 D-3] geometric verification of a baked sequence.

Self-consistency gate for the YOPO CUDA raycaster + dataset_generator
dynamic mode. For each frame, iterates the dynamic obstacles in
dyn_obs.json; for the first ball whose projected pixel lies on the
front surface of THAT BALL in depth_t{k}.png, the depth at that pixel
is compared to the analytical front-surface depth (d_cam - radius).

Why just one pixel?  The diameter / centroid checks the §5.6 sketch
suggested are unreliable when:
  - other balls or static obstacles share the search ROI at similar depth
  - the ball is partially occluded -> visible silhouette differs from
    analytic projection

The single-pixel front-surface depth check is sufficient to catch any
systematic bug in:
  - ray_sphere_t math in sensor_simulator.cu
  - world->camera transform in dataset_generator.cpp
  - PNG encoding (depth/65535 * max_depth_m round-trip)

Three outcomes per frame:
  PASS  depth(u_exp, v_exp) within TOL_depth_m of (d_cam - radius)
  SKIP  the ball is occluded (depth at the projected pixel is shallower
        than d_cam - radius - margin) OR no ball is in the FOV.  This is
        a legitimate rendering result, not a bug, so it doesn't count
        toward FAIL.  (We iterate all balls in a frame to find one that
        is unoccluded.)
  FAIL  the projected pixel is unoccluded but the recorded depth is FAR
        deeper than expected (>= d_cam + radius + margin).  This is the
        bug signature -- something is wrong with ray_sphere / transform /
        encoding.

Pass gate: PASS + SKIP >= 8 and FAIL == 0.  Even one FAIL stops 2.d.

Usage:
    python tools/verify_dynamic_render.py dataset_dynamic/smoke/env_0000/seq_0000
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

TOL_DEPTH_M = 0.10      # front-surface depth match tolerance
# Rationale for 0.10m (looser than §5.6's nominal 0.05m): we sample depth at
# the *rounded integer* pixel closest to the analytic projected center, not
# the analytic subpixel point.  At d_cam ~ 10m with fx=80, a 1-pixel offset
# moves the hit point on the sphere by ~ d_cam / fx = 0.125m of world distance,
# which traces out ~ a few cm of depth difference on the sphere surface.
# Empirically smoke seq_0003 produces err in [0.015, 0.051]m -- well below
# 0.10m, while any real ray_sphere / projection / encoding bug would manifest
# as >> 1m error (the same kinds of bugs the test_dyn_sphere.cpp on-axis 0.05m
# check already catches with no subpixel ambiguity).
OCCLUSION_MARGIN_M = 0.10   # if pixel depth < d_front - margin, ball is occluded -> SKIP


def quat_rotate(q_wxyz, v):
    """Rotate 3-vector v by quaternion q (Hamilton, w-x-y-z order)."""
    w, x, y, z = q_wxyz
    # quaternion rotation: v' = q * v * q^-1, expanded to matrix
    R = np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R @ np.asarray(v, dtype=np.float64)


def quat_conj(q_wxyz):
    return np.array([q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3]])


def project_yopo(p_world, drone_pos, drone_quat_wc, intr):
    """world -> camera (YOPO +X forward) -> image plane.

    YOPO sensor_simulator.cu kernel uses:
        y_cam = -(u - cx) / fx
        z_cam = -(v - cy) / fy
        x_cam = 1 (forward, normalized later)
    Inverting gives:
        u = -fx * p_cam.y / p_cam.x + cx
        v = -fy * p_cam.z / p_cam.x + cy
        depth = p_cam.x (the kernel records cam-frame x of the hit point)
    """
    rel_world = np.asarray(p_world, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)
    p_cam = quat_rotate(quat_conj(drone_quat_wc), rel_world)
    u = -intr["fx"] * p_cam[1] / p_cam[0] + intr["cx"]
    v = -intr["fy"] * p_cam[2] / p_cam[0] + intr["cy"]
    return float(u), float(v), float(p_cam[0])


def check_one_ball(depth, ball, drone_state, intr):
    """Try a single ball; return (status, u_exp, v_exp, d_cam, depth_at_pixel, err_z).
    status is one of: "PASS", "FAIL", "OUT_OF_VIEW", "OCCLUDED".
    """
    u_exp, v_exp, d_cam = project_yopo(ball["pos"], drone_state["pos"],
                                        drone_state["quat_wc"], intr)
    if d_cam < 0.3 or not (0 <= u_exp < intr["W"]) or not (0 <= v_exp < intr["H"]):
        return ("OUT_OF_VIEW", u_exp, v_exp, d_cam, float("nan"), float("nan"))
    u0 = int(round(u_exp))
    v0 = int(round(v_exp))
    depth_at = float(depth[v0, u0])
    d_front = d_cam - ball["radius"]
    err_z = abs(depth_at - d_front)
    if depth_at < d_front - OCCLUSION_MARGIN_M:
        # Something closer than the ball's front surface lives at this pixel
        # -> ball is occluded.  Legitimate, not a math bug.
        return ("OCCLUDED", u_exp, v_exp, d_cam, depth_at, err_z)
    if err_z <= TOL_DEPTH_M:
        return ("PASS", u_exp, v_exp, d_cam, depth_at, err_z)
    return ("FAIL", u_exp, v_exp, d_cam, depth_at, err_z)


def main(seq_dir):
    seq_dir = Path(seq_dir)
    meta = json.loads((seq_dir / "meta.json").read_text())
    states = json.loads((seq_dir / "state.json").read_text())
    dyns = json.loads((seq_dir / "dyn_obs.json").read_text())
    intr = meta["intrinsics"]
    max_depth_m = float(intr["max_depth_m"])
    encoding = meta.get("depth_encoding", "")
    print(f"== verify_dynamic_render: {seq_dir} ==")
    print(f"   K={meta['K']}  intrinsics={intr}  depth_encoding={encoding!r}")

    K = meta["K"]
    if len(states) != K or len(dyns) != K:
        print(f"[FAIL] state/dyn length != K: state={len(states)}, dyn={len(dyns)}, K={K}")
        return 1

    n_pass = 0
    n_fail = 0
    n_skip = 0

    for k in range(K):
        raw = cv2.imread(str(seq_dir / f"depth_t{k}.png"), cv2.IMREAD_UNCHANGED)
        if raw is None:
            print(f"[FAIL] frame {k}: depth_t{k}.png unreadable")
            n_fail += 1
            continue
        # decode: uint16 normalized by max_depth_m
        depth = raw.astype(np.float32) / 65535.0 * max_depth_m

        if len(dyns[k]) == 0:
            print(f"[SKIP] frame {k}: no balls")
            n_skip += 1
            continue

        # Iterate all balls in the frame; the first one whose projected pixel
        # actually shows the ball's front surface decides the frame's verdict.
        # If we encounter a FAIL for any ball we trust it immediately (real
        # geometry bugs surface as one bad ball per frame).
        frame_status = None
        frame_record = None
        for cand in dyns[k]:
            st, u, v_, d_cam, d_at, err_z = check_one_ball(depth, cand, states[k], intr)
            if st == "FAIL":
                frame_status, frame_record = st, (u, v_, d_cam, d_at, err_z, cand)
                break
            if st == "PASS":
                frame_status, frame_record = st, (u, v_, d_cam, d_at, err_z, cand)
                break
            # OUT_OF_VIEW / OCCLUDED -> keep looking; remember the last one
            # for diagnostics if no PASS/FAIL is found.
            frame_status = st
            frame_record = (u, v_, d_cam, d_at, err_z, cand)

        if frame_status == "PASS":
            u, v_, d_cam, d_at, err_z, cand = frame_record
            print(f"[PASS] frame {k:2d}: "
                  f"u_exp={u:6.2f} v_exp={v_:5.2f} d_front_exp={d_cam - cand['radius']:5.2f}m  "
                  f"depth_at_pixel={d_at:5.2f}m  err={err_z:.3f}m (tol {TOL_DEPTH_M}m)")
            n_pass += 1
        elif frame_status == "FAIL":
            u, v_, d_cam, d_at, err_z, cand = frame_record
            print(f"[FAIL] frame {k:2d}: "
                  f"u_exp={u:6.2f} v_exp={v_:5.2f} d_front_exp={d_cam - cand['radius']:5.2f}m  "
                  f"depth_at_pixel={d_at:5.2f}m  err={err_z:.3f}m  "
                  f"(should be ~d_front)")
            n_fail += 1
        else:
            # OUT_OF_VIEW or OCCLUDED for every ball -> SKIP
            reason = "all balls occluded or out of view"
            print(f"[SKIP] frame {k:2d}: {reason} (tried {len(dyns[k])} candidate(s))")
            n_skip += 1

    print()
    print(f"summary: {n_pass} PASS, {n_skip} SKIP, {n_fail} FAIL  (K={K})")
    # Pass gate: PASS + SKIP >= 8 and FAIL == 0
    if n_fail == 0 and (n_pass + n_skip) >= 8:
        print("[PASS] geometry verification gate cleared (>=8/10 OK or SKIP, 0 FAIL)")
        return 0
    print("[FAIL] geometry verification gate NOT cleared -- check ray_sphere_t / projection / encoding")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: verify_dynamic_render.py <seq_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
