"""Stage-4 ablation: train 5 configurations head-to-head and tabulate the
tail-window loss medians.

Configs (rows are additive — each builds on the previous):
  A baseline-yopo : reVAE off, DCA off, dynamic_ratio 0
                    -> closest to the original YOPO single-frame baseline.
  B +reVAE        : reVAE on,  DCA off, dynamic_ratio 0
                    -> stage-1 architecture, no dynamic data at all.
  C +dyn/kino     : reVAE on,  DCA off, dynamic_ratio 0.5, lam_dyn/kino default
                    -> stage-3.1 (path c), losses only.
  D +DCA          : reVAE on,  DCA on,  dynamic_ratio 0.5, lam_dyn=lam_kino=0
                    -> stage-3.2 architecture, no dyn-loss supervision.
  E full          : reVAE on,  DCA on,  dynamic_ratio 0.5, lam_dyn/kino default
                    -> stage-3.2 final config.

For each row we mutate cfg in-place, then construct a fresh YopoTrainer and
run --steps mixed-sampling iterations.  Per-step losses are accumulated and
the head-N / tail-N means written to results/ablation.csv.

The script is deterministic up to dataloader workers: torch.manual_seed +
numpy.random.seed are reset before each row.

Run from REACT/ root:
  python scripts/run_stage4_ablation.py --steps 2000
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

from config.config import cfg


# ---------- ablation matrix ---------------------------------------------------

# Each entry: dotted-path overrides applied to cfg._data before trainer init.
# Default cfg snapshot (from traj_opt.yaml @ HEAD):
#   revae.enable=true, dynamic_attention.enable=true, dynamic_ratio=0.5,
#   loss_weights.lam_dyn=3.0, loss_weights.lam_kino=0.5
ABLATIONS = [
    ("A_baseline_yopo", {
        "revae.enable": False,
        "dynamic_attention.enable": False,
        "dynamic_ratio": 0.0,
        "loss_weights.lam_dyn": 0.0,
        "loss_weights.lam_kino": 0.0,
    }),
    ("B_plus_reVAE", {
        "revae.enable": True,
        "dynamic_attention.enable": False,
        "dynamic_ratio": 0.0,
        "loss_weights.lam_dyn": 0.0,
        "loss_weights.lam_kino": 0.0,
    }),
    ("C_plus_dyn_kino", {
        "revae.enable": True,
        "dynamic_attention.enable": False,
        "dynamic_ratio": 0.5,
        "loss_weights.lam_dyn": 3.0,
        "loss_weights.lam_kino": 0.5,
    }),
    ("D_plus_DCA_no_loss", {
        "revae.enable": True,
        "dynamic_attention.enable": True,
        "dynamic_ratio": 0.5,
        "loss_weights.lam_dyn": 0.0,
        "loss_weights.lam_kino": 0.0,
    }),
    ("E_full", {
        "revae.enable": True,
        "dynamic_attention.enable": True,
        "dynamic_ratio": 0.5,
        "loss_weights.lam_dyn": 3.0,
        "loss_weights.lam_kino": 0.5,
    }),
]


def apply_overrides(overrides):
    """Mutate cfg._data in-place per dotted-key overrides."""
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node = cfg._data
        for k in parts[:-1]:
            node = node[k]
        node[parts[-1]] = value


def snapshot_cfg():
    """Capture the keys we modify so we can restore between rows."""
    return {
        "revae.enable":             cfg["revae"]["enable"],
        "dynamic_attention.enable": cfg["dynamic_attention"]["enable"],
        "dynamic_ratio":            cfg["dynamic_ratio"],
        "loss_weights.lam_dyn":     cfg["loss_weights"]["lam_dyn"],
        "loss_weights.lam_kino":    cfg["loss_weights"]["lam_kino"],
    }


# ---------- one-row training driver -------------------------------------------

def run_one(name, overrides, args):
    print(f"\n========== {name} ==========")
    apply_overrides(overrides)
    print("  cfg overrides applied:")
    for k, v in overrides.items():
        print(f"    {k:32s} = {v}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Late import so cfg mutations are visible to the trainer.
    from policy.yopo_trainer import YopoTrainer
    trainer = YopoTrainer(
        learning_rate=args.lr,
        batch_size=args.batch,
        loss_weight=[1.0, 1.0],
        tensorboard_path="saved/_scratch",
        checkpoint_path="",
        save_on_exit=False,
    )
    trainer.policy.train()
    lam_vae = float(cfg["loss_weights"]["lam_vae"])

    stat_traj, stat_dyn = [], []
    dyn_traj, dyn_dyn, dyn_kino = [], [], []
    revae_all, total_all = [], []
    is_dyn_flags = []

    t0 = time.time()
    step = 0
    while step < args.steps:
        for static_batch in trainer.train_dataloader:
            if static_batch[0].shape[0] != trainer.batch_size:
                continue

            dyn_payload = None
            if (trainer.dyn_train_iter is not None
                    and torch.rand(1).item() < trainer.dynamic_ratio):
                try:
                    db = next(trainer.dyn_train_iter)
                except StopIteration:
                    trainer.dyn_train_iter = iter(trainer.dyn_train_dataloader)
                    db = next(trainer.dyn_train_iter)
                if db[0].shape[0] == trainer.batch_size:
                    depth, pos, rot, obs_b, map_id, dyn_pad, dyn_mask = db
                    dyn_payload = (dyn_pad, dyn_mask)
                else:
                    depth, pos, rot, obs_b, map_id = static_batch
            else:
                depth, pos, rot, obs_b, map_id = static_batch

            trainer.optimizer.zero_grad()
            (traj, score, revae, dyn, kino, _, _, _, _) = \
                trainer.forward_and_compute_loss(depth, pos, rot, obs_b, map_id,
                                                  dyn_obs=dyn_payload)
            loss = (trainer.loss_weight[0] * traj
                    + trainer.loss_weight[1] * score
                    + lam_vae * revae + dyn + kino)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.policy.parameters(),
                                           trainer.max_grad_norm)
            trainer.optimizer.step()

            total_all.append(loss.item())
            revae_all.append(revae.item())
            if dyn_payload is not None:
                dyn_traj.append(traj.item())
                dyn_dyn.append(dyn.item())
                dyn_kino.append(kino.item())
                is_dyn_flags.append(True)
            else:
                stat_traj.append(traj.item())
                stat_dyn.append(dyn.item())
                is_dyn_flags.append(False)

            if step % max(1, args.steps // 20) == 0:
                kind = "DYN " if dyn_payload is not None else "stat"
                print(f"  step {step:5d}/{args.steps} [{kind}]  "
                      f"total={loss.item():6.3f}  traj={traj.item():.3f}  "
                      f"dyn={dyn.item():.4f}  kino={kino.item():.4f}")

            step += 1
            if step >= args.steps:
                break

    wall = time.time() - t0
    n_dyn = sum(is_dyn_flags)
    n_stat = step - n_dyn
    win = max(50, args.steps // 10)

    def tail(xs):
        return float(np.mean(xs[-win:])) if xs else float("nan")

    def head(xs):
        return float(np.mean(xs[:win])) if xs else float("nan")

    row = {
        "name":            name,
        "steps":           step,
        "wall_s":          round(wall, 1),
        "n_dyn":           n_dyn,
        "n_stat":          n_stat,
        "total_head":      round(head(total_all), 4),
        "total_tail":      round(tail(total_all), 4),
        "stat_traj_head":  round(head(stat_traj), 4) if stat_traj else float("nan"),
        "stat_traj_tail":  round(tail(stat_traj), 4) if stat_traj else float("nan"),
        # 🟦 stage-4: on rows C/D/E the dataloader produces real dynamic
        # scenes so dyn_traj is the imitation-learning trajectory loss under
        # moving obstacles -- comparable across all three regardless of the
        # lam_dyn weighting.
        "dyn_traj_tail":   round(tail(dyn_traj), 4) if dyn_traj else float("nan"),
        "dyn_dyn_head":    round(head(dyn_dyn), 4) if dyn_dyn else float("nan"),
        "dyn_dyn_tail":    round(tail(dyn_dyn), 4) if dyn_dyn else float("nan"),
        "dyn_kino_tail":   round(tail(dyn_kino), 4) if dyn_kino else float("nan"),
        "revae_tail":      round(tail(revae_all), 4),
    }
    print(f"  [{name}] wall={wall:.1f}s  total head/tail = "
          f"{row['total_head']}/{row['total_tail']}  "
          f"stat_traj tail = {row['stat_traj_tail']}  "
          f"dyn_dyn tail = {row['dyn_dyn_tail']}")

    # Free GPU mem before next row so a 4-row pile-up doesn't OOM.
    del trainer
    torch.cuda.empty_cache()
    return row


# ---------- main --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000,
                    help="iterations per ablation row")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated subset of row names to run "
                         "(e.g. 'C_plus_dyn_kino,E_full')")
    ap.add_argument("--out", type=str,
                    default=os.path.join(REACT_ROOT, "results", "ablation.csv"))
    args = ap.parse_args()

    # The script does os.chdir(YOPO_DIR) at import time so any relative --out
    # would land under YOPO/results/.  Resolve against REACT_ROOT explicitly.
    if not os.path.isabs(args.out):
        args.out = os.path.join(REACT_ROOT, args.out)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    selected = set(args.only.split(",")) if args.only else None
    rows_to_run = [(n, o) for n, o in ABLATIONS if selected is None or n in selected]
    print(f"== stage-4 ablation: {len(rows_to_run)} row(s) x {args.steps} steps ==")

    baseline_snapshot = snapshot_cfg()
    results = []
    for name, overrides in rows_to_run:
        # Reset cfg to the yaml defaults before each row so a previous row's
        # mutation never leaks into the next.
        apply_overrides(baseline_snapshot)
        results.append(run_one(name, overrides, args))

    # Restore yaml defaults at the very end.
    apply_overrides(baseline_snapshot)

    # Write CSV
    fieldnames = list(results[0].keys())
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nWrote {args.out}")

    # ASCII summary
    print("\n=== ablation summary (tail-window means) ===")
    header = (f"{'row':<22}  {'total':>10}  {'stat_traj':>10}  "
              f"{'dyn_traj':>10}  {'dyn_dyn':>10}  {'dyn_kino':>10}  "
              f"{'reVAE':>10}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['name']:<22}  {r['total_tail']:>10}  "
              f"{r['stat_traj_tail']:>10}  {r['dyn_traj_tail']:>10}  "
              f"{r['dyn_dyn_tail']:>10}  {r['dyn_kino_tail']:>10}  "
              f"{r['revae_tail']:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
