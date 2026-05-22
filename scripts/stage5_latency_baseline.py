"""Stage-5.A.0 — eager-PyTorch inference latency baseline.

Measures forward-pass latency on the dev GPU for three input modes that
matter at deployment:

  M1 single-frame stage-3.1 (use_temporal=False, use_dca=False).
     depth shape (1, 1, 96, 160).
  M2 stage-3.2 side-channel (use_temporal=False, use_dca=True).  dyn_obs
     tokens are supplied so the DCA path actually runs.
  M3 stage-3.4 K-frame stateless (use_temporal=True, use_dca=False).
     depth shape (1, 10, 1, 96, 160).

For each mode:
  - Warm-up: 20 forward passes (kernels compiled, allocator settled).
  - Timing: 200 forward passes with CUDA events around each call.
  - Report p50, p90, p99, mean, ratio-to-M1.

This is a *correlate* to the Jetson target -- absolute numbers are RTX 3070,
but the relative scaling between modes is informative and the analytic
budget in REACT_MATH_Derivations/04_stage5_deployment_math.tex §2 gives
the Jetson conversion.

Outputs a CSV at results/stage5_latency_baseline.csv.

Run from REACT/ root:
  python scripts/stage5_latency_baseline.py
"""
import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REACT_ROOT = os.path.dirname(HERE)
YOPO_DIR = os.path.join(REACT_ROOT, "YOPO")
os.chdir(YOPO_DIR)
sys.path.insert(0, YOPO_DIR)

import numpy as np
import torch

from policy.yopo_network import YopoNetwork


def cuda_time_ms(func, n: int):
    """Run func() n times, return per-call times in ms via CUDA events."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    times = []
    for i in range(n):
        starts[i].record()
        func()
        ends[i].record()
    torch.cuda.synchronize()
    for s, e in zip(starts, ends):
        times.append(s.elapsed_time(e))
    return np.asarray(times, dtype=np.float64)


def percentiles(x):
    return {
        "p50":  float(np.percentile(x, 50)),
        "p90":  float(np.percentile(x, 90)),
        "p99":  float(np.percentile(x, 99)),
        "mean": float(np.mean(x)),
        "std":  float(np.std(x)),
    }


def mode_M1():
    return {
        "name": "M1_single_frame",
        "use_temporal": False, "use_dca": False,
        "depth_shape": (1, 1, 96, 160), "obs_shape": (1, 9, 3, 5),
        "dyn_obs": False,
    }


def mode_M2():
    return {
        "name": "M2_dca",
        "use_temporal": False, "use_dca": True,
        "depth_shape": (1, 1, 96, 160), "obs_shape": (1, 9, 3, 5),
        "dyn_obs": True,
    }


def mode_M3():
    return {
        "name": "M3_kframe_stateless",
        "use_temporal": True, "use_dca": False,
        "depth_shape": (1, 10, 1, 96, 160), "obs_shape": (1, 9, 3, 5),
        "dyn_obs": False,
    }


def build_net(mode, device):
    net = YopoNetwork(use_revae=True, revae_latent=128,
                      use_dca=mode["use_dca"], dca_n_heads=1,
                      use_temporal=mode["use_temporal"],
                      temporal_hidden=128 if mode["use_temporal"] else None)
    net = net.to(device).eval()
    return net


def run_one_mode(mode, args, device):
    print(f"\n========== {mode['name']} ==========")
    print(f"  cfg: use_temporal={mode['use_temporal']}, use_dca={mode['use_dca']}")

    net = build_net(mode, device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  params: {n_params:,}")

    depth = torch.randn(*mode["depth_shape"], device=device)
    obs   = torch.randn(*mode["obs_shape"], device=device)
    dyn_tokens = None
    dyn_mask   = None
    if mode["dyn_obs"]:
        # 8 obstacle slots, 7 dims [rel_pos, abs_vel, radius]
        dyn_tokens = torch.randn(1, 8, 7, device=device)
        dyn_mask   = torch.tensor([[True]*5 + [False]*3], device=device)

    def call():
        with torch.no_grad():
            net(depth, obs, dyn_obs_tokens=dyn_tokens, dyn_obs_mask=dyn_mask)

    # Warm up
    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()

    # Time
    t = cuda_time_ms(call, args.repeat)
    p = percentiles(t)
    print(f"  latency ms: p50={p['p50']:.3f}  p90={p['p90']:.3f}  "
          f"p99={p['p99']:.3f}  mean={p['mean']:.3f} +- {p['std']:.3f}")

    return {
        "name":      mode["name"],
        "params":    n_params,
        **p,
        "depth_shape": str(mode["depth_shape"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=200,
                    help="number of timed iterations per mode")
    ap.add_argument("--warmup", type=int, default=20,
                    help="number of warm-up iterations per mode")
    ap.add_argument("--out", type=str,
                    default=os.path.join(REACT_ROOT, "results",
                                          "stage5_latency_baseline.csv"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available -- this benchmark requires a GPU.")
        return 1

    device = torch.device("cuda")
    print(f"== stage-5.A.0 latency baseline on {torch.cuda.get_device_name(0)} ==")
    print(f"   torch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"   per-mode: {args.warmup} warm-up + {args.repeat} timed iters")

    rows = [run_one_mode(m(), args, device) for m in (mode_M1, mode_M2, mode_M3)]

    # Ratio-to-M1
    m1_p50 = rows[0]["p50"]
    print(f"\n=== ratio to M1 (single-frame, p50={m1_p50:.3f} ms) ===")
    for r in rows:
        print(f"  {r['name']:25s}: p50={r['p50']:6.3f} ms  ({r['p50']/m1_p50:.2f}x)")

    if not os.path.isabs(args.out):
        args.out = os.path.join(REACT_ROOT, args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {args.out}")

    # 10 ms budget check (RTX number; Jetson is harder)
    print(f"\n=== <10 ms budget check (RTX 3070 numbers) ===")
    for r in rows:
        ok = "WITHIN" if r["p99"] < 10.0 else "OVER"
        margin = 10.0 - r["p99"]
        print(f"  {r['name']:25s}: p99={r['p99']:6.3f} ms  [{ok}]  margin={margin:+.2f} ms")
    print("\nNote: Jetson Orin NX FP16 is ~2-3x slower than RTX 3070 for "
          "convolutional workloads; INT8 closes most of the gap.  Use this "
          "table to RANK modes, not to claim the Jetson number directly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
