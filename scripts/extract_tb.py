"""Extract scalar curves from a tensorboard log dir and print a compact
ASCII-sparkline summary. Useful when running headless (no TB UI handy).

Usage:
    python scripts/extract_tb.py [tb_log_dir]
    # default: YOPO/saved/YOPO_<latest>
"""
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BLOCKS = " ▁▂▃▄▅▆▇█"
HERE = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(HERE, "..", "YOPO", "saved")


def latest_log_dir():
    candidates = [d for d in os.listdir(SAVED_DIR)
                  if d.startswith("YOPO_") and d.split("_")[1].isdigit()]
    if not candidates:
        raise FileNotFoundError(f"no YOPO_* under {SAVED_DIR}")
    latest = max(candidates, key=lambda d: int(d.split("_")[1]))
    return os.path.join(SAVED_DIR, latest)


def spark(vals, width=40):
    if not vals:
        return ""
    if len(vals) > width:
        step = len(vals) / width
        sampled = [vals[int(i * step)] for i in range(width)]
    else:
        sampled = vals
    lo, hi = min(sampled), max(sampled)
    rng = hi - lo if hi > lo else 1.0
    return "".join(BLOCKS[min(8, max(0, int((v - lo) / rng * 8)))] for v in sampled)


def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else latest_log_dir()
    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = sorted(ea.Tags()["scalars"])

    print(f"== TensorBoard summary: {log_dir} ==")
    print(f"  available tags: {tags}\n")

    for tag in tags:
        events = ea.Scalars(tag)
        vals = [e.value for e in events]
        if not vals:
            continue
        first, last = vals[0], vals[-1]
        pct = (last - first) / abs(first) * 100 if first != 0 else 0.0
        print(f"  {tag:<22} n={len(vals):4d}  first={first:8.4f}  last={last:8.4f}  "
              f"min={min(vals):8.4f}  max={max(vals):8.4f}  Δ={pct:+6.1f}%")
        print(f"  {' ' * 22} curve [{spark(vals)}]")
        print()


if __name__ == "__main__":
    main()
