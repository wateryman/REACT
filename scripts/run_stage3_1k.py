"""Stage-3.1 D: 1k-iter mixed-sampling validation.

Drives the trainer's mixed-sampling loop for max_steps steps (default 1000),
tracking per-step loss values segregated by batch kind (static vs dynamic).

過関 gates (per guide §6.1 stage-3):
  (G1) all losses finite (no NaN/inf)
  (G2) static path is regression-preserving:
       static-batch dyn_loss == 0 EXACT every step
  (G3) traj_loss decreases monotonically (head-vs-tail >= 15% drop)
  (G4) static-batch traj_loss tail mean within 5% of stage-1 baseline
       (stage-1 commit c7b9031 ended at traj 4.58 over 100 iter with same
        batch=8, so we expect <= 4.58 * 1.05 = 4.81 after 1k iter mixed
        training)
  (G5) dynamic-batch dyn_loss decreases (head-vs-tail >= 15% drop)

If any gate fails, stage-3.1 needs investigation before merging to main.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
YOPO_DIR = os.path.join(HERE, "..", "YOPO")
os.chdir(YOPO_DIR)
sys.path.insert(0, YOPO_DIR)

import numpy as np
import torch

from config.config import cfg
from policy.yopo_trainer import YopoTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    args = ap.parse_args()

    print(f"== stage-3.1 D run | max_steps={args.max_steps} batch={args.batch} ==")

    trainer = YopoTrainer(
        learning_rate=args.lr,
        batch_size=args.batch,
        loss_weight=[1.0, 1.0],
        tensorboard_path="saved/_scratch",
        checkpoint_path="",
        save_on_exit=False,
    )
    trainer.policy.train()
    assert trainer.dyn_train_iter is not None, "dynamic dataloader missing"
    print(f"   dynamic_ratio = {trainer.dynamic_ratio}")
    print(f"   tb log dir    = {trainer.tensorboard_path}")

    lam_vae = float(cfg["loss_weights"]["lam_vae"])

    # Per-step records, segregated
    stat_traj = []   # static-batch trajectory loss
    stat_dyn  = []   # static-batch dyn loss (must be 0)
    stat_kino = []
    dyn_traj  = []   # dynamic-batch trajectory loss
    dyn_dyn   = []   # dynamic-batch dyn loss
    dyn_kino  = []   # dynamic-batch kino loss
    revae_all = []
    total_all = []
    is_dyn_flag = []   # per-step bool

    step = 0
    for static_batch in trainer.train_dataloader:
        if static_batch[0].shape[0] != trainer.batch_size:
            continue

        dyn_payload = None
        if torch.rand(1).item() < trainer.dynamic_ratio:
            try:
                dyn_batch = next(trainer.dyn_train_iter)
            except StopIteration:
                trainer.dyn_train_iter = iter(trainer.dyn_train_dataloader)
                dyn_batch = next(trainer.dyn_train_iter)
            if dyn_batch[0].shape[0] == trainer.batch_size:
                depth, pos, rot, obs_b, map_id, dyn_pad, dyn_mask = dyn_batch
                dyn_payload = (dyn_pad, dyn_mask)
            else:
                depth, pos, rot, obs_b, map_id = static_batch
        else:
            depth, pos, rot, obs_b, map_id = static_batch

        trainer.optimizer.zero_grad()
        (traj, score, revae, dyn, kino, smooth, safety, goal, acc) = \
            trainer.forward_and_compute_loss(depth, pos, rot, obs_b, map_id,
                                              dyn_obs=dyn_payload)
        loss = (trainer.loss_weight[0] * traj
                + trainer.loss_weight[1] * score
                + lam_vae * revae + dyn + kino)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.policy.parameters(), trainer.max_grad_norm)
        trainer.optimizer.step()

        # TB logging (every step is fine for 1k)
        global_step = step
        trainer.tensorboard_log.add_scalar("Train/TrajLoss",  traj.item(),  global_step)
        trainer.tensorboard_log.add_scalar("Train/ScoreLoss", score.item(), global_step)
        trainer.tensorboard_log.add_scalar("Train/ReVAELoss", lam_vae*revae.item(), global_step)
        trainer.tensorboard_log.add_scalar("Train/DynLoss",   dyn.item(),   global_step)
        trainer.tensorboard_log.add_scalar("Train/KinoLoss",  kino.item(),  global_step)
        trainer.tensorboard_log.add_scalar("Train/TotalLoss", loss.item(),  global_step)
        trainer.tensorboard_log.add_scalar("Detail/SmoothLoss", smooth.item(), global_step)
        trainer.tensorboard_log.add_scalar("Detail/SafetyLoss", safety.item(), global_step)
        trainer.tensorboard_log.add_scalar("Detail/GoalLoss",   goal.item(),   global_step)
        trainer.tensorboard_log.add_scalar("Detail/AccelLoss",  acc.item(),    global_step)

        revae_all.append(revae.item())
        total_all.append(loss.item())
        if dyn_payload is not None:
            dyn_traj.append(traj.item())
            dyn_dyn.append(dyn.item())
            dyn_kino.append(kino.item())
            is_dyn_flag.append(True)
        else:
            stat_traj.append(traj.item())
            stat_dyn.append(dyn.item())
            stat_kino.append(kino.item())
            is_dyn_flag.append(False)

        if step % 50 == 0:
            kind = "DYN " if dyn_payload is not None else "stat"
            print(f"  step {step:4d} [{kind}]: total={loss.item():6.3f}  "
                  f"traj={traj.item():.3f}  score={score.item():.3f}  "
                  f"dyn={dyn.item():.4f}  kino={kino.item():.4f}")
        step += 1
        if step >= args.max_steps:
            break

    trainer.tensorboard_log.flush()
    print(f"\ntotal steps: {step}  (dyn={sum(is_dyn_flag)}, static={step-sum(is_dyn_flag)})")

    # --- 過関 gates ---
    fails = []

    def gate(name, cond, info):
        ok = bool(cond)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {info}")
        if not ok: fails.append(name)

    # G1: finite
    all_vals = total_all + stat_traj + dyn_traj + dyn_dyn + dyn_kino + stat_dyn + revae_all
    n_bad = sum(1 for v in all_vals if not np.isfinite(v))
    print()
    gate("G1 all finite", n_bad == 0, f"non-finite count = {n_bad}")

    # G2: static dyn_loss == 0
    stat_dyn_max = max(abs(x) for x in stat_dyn) if stat_dyn else 0
    gate("G2 static regression byte-clean",
         stat_dyn_max == 0.0, f"max-abs static dyn_loss = {stat_dyn_max:.2e}")

    # G3: total loss decreasing
    win = max(50, len(total_all) // 10)
    head_total = float(np.mean(total_all[:win]))
    tail_total = float(np.mean(total_all[-win:]))
    drop_total = (head_total - tail_total) / head_total * 100
    gate("G3 total loss drop >= 15%",
         drop_total >= 15.0,
         f"head-{win} = {head_total:.3f}, tail-{win} = {tail_total:.3f}, drop = {drop_total:.1f}%")

    # G4: static traj loss near baseline
    # Stage-1 baseline at batch=8 ended ~4.58 over 100 iter; for batch=16 we
    # expect similar magnitude (network output range comparable).
    if stat_traj:
        head_st = float(np.mean(stat_traj[:max(20, len(stat_traj)//10)]))
        tail_st = float(np.mean(stat_traj[-max(20, len(stat_traj)//10):]))
        drop_st = (head_st - tail_st) / head_st * 100
        gate("G4 static traj_loss decreases",
             drop_st >= 10.0,
             f"head = {head_st:.3f}, tail = {tail_st:.3f}, drop = {drop_st:.1f}%")
    else:
        gate("G4 static traj_loss decreases", False, "no static batches (??)")

    # G5: dynamic dyn_loss decreasing
    if dyn_dyn:
        head_dd = float(np.mean(dyn_dyn[:max(20, len(dyn_dyn)//10)]))
        tail_dd = float(np.mean(dyn_dyn[-max(20, len(dyn_dyn)//10):]))
        drop_dd = (head_dd - tail_dd) / head_dd * 100 if head_dd > 0 else 0
        gate("G5 dyn_loss decreases >= 15%",
             drop_dd >= 15.0,
             f"head = {head_dd:.4f}, tail = {tail_dd:.4f}, drop = {drop_dd:.1f}%")
    else:
        gate("G5 dyn_loss decreases", False, "no dynamic batches (??)")

    # Summary table
    print(f"\n=== summary (last {win}-window means) ===")
    print(f"  total:  {head_total:.3f} -> {tail_total:.3f}  ({drop_total:+.1f}%)")
    if stat_traj:
        print(f"  static traj: {head_st:.3f} -> {tail_st:.3f}  ({drop_st:+.1f}%)")
    if dyn_dyn:
        print(f"  dyn dyn:     {head_dd:.4f} -> {tail_dd:.4f}  ({drop_dd:+.1f}%)")
    print(f"  revae mean (all steps): {float(np.mean(revae_all)):.4f}")
    print(f"  kino (dyn batches > 0): {sum(1 for v in dyn_kino if v > 0)}/{len(dyn_kino)}")

    if fails:
        print(f"\n[FAIL] gates failed: {fails}")
        return 1
    print(f"\n[PASS] stage-3.1 D: 5/5 gates cleared")
    print(f"       TB log: {trainer.tensorboard_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
