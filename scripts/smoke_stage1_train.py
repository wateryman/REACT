"""Stage-1 PEMTRS-port 100-iter smoke training (REACT guide §6.1).

Reuses YopoTrainer's __init__ to wire up dataset/loss/policy/optimizer,
then manually walks the dataloader for max_steps to verify:

  (1) total_loss is finite and trending down,
  (2) reVAE parameters receive non-zero gradient,
  (3) trajectory_loss (original YOPO surface) also trends down.

If pass, stage-1 is gated through; if fail, we roll back per guide §6.0.3
without touching assertions.

Run from REACT/ root:
    python scripts/smoke_stage1_train.py [--max-steps 100] [--batch 8]
"""
import argparse
import os
import sys
from statistics import mean

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
YOPO_DIR = os.path.join(HERE, "..", "YOPO")
os.chdir(YOPO_DIR)
sys.path.insert(0, YOPO_DIR)

from config.config import cfg
from policy.yopo_trainer import YopoTrainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    args = ap.parse_args()

    log_dir = os.path.join(YOPO_DIR, "saved")
    os.makedirs(log_dir, exist_ok=True)

    print(f"== stage-1 smoke train | max_steps={args.max_steps} batch={args.batch} ==")

    trainer = YopoTrainer(
        learning_rate=args.lr,
        batch_size=args.batch,
        loss_weight=[1.0, 1.0],
        tensorboard_path=log_dir,
        checkpoint_path="",
        save_on_exit=False,
    )
    trainer.policy.train()

    lam_vae = float(cfg["loss_weights"]["lam_vae"])

    total_losses, traj_losses, score_losses, revae_losses = [], [], [], []
    revae_param = next(trainer.policy.revae.parameters())
    revae_grad_max = 0.0
    step = 0
    for depth, pos, rot, obs_b, map_id in trainer.train_dataloader:
        if depth.shape[0] != args.batch:
            continue
        trainer.optimizer.zero_grad()
        traj, score, revae, _, _, _, _ = trainer.forward_and_compute_loss(
            depth, pos, rot, obs_b, map_id
        )
        loss = (
            trainer.loss_weight[0] * traj
            + trainer.loss_weight[1] * score
            + lam_vae * revae
        )
        loss.backward()
        trainer.optimizer.step()

        if revae_param.grad is not None:
            revae_grad_max = max(revae_grad_max, revae_param.grad.abs().max().item())

        total_losses.append(loss.item())
        traj_losses.append(traj.item())
        score_losses.append(score.item())
        revae_losses.append(revae.item())

        if step % 10 == 0:
            print(f"  step {step:3d}  total={loss.item():.4f}  "
                  f"traj={traj.item():.4f}  score={score.item():.4f}  "
                  f"revae={revae.item():.4f}")
        step += 1
        if step >= args.max_steps:
            break

    assert step >= args.max_steps, f"dataloader exhausted at step {step} before max_steps"

    print()
    head_total = mean(total_losses[:20])
    tail_total = mean(total_losses[-20:])
    head_traj = mean(traj_losses[:20])
    tail_traj = mean(traj_losses[-20:])
    head_revae = mean(revae_losses[:20])
    tail_revae = mean(revae_losses[-20:])
    print(f"  first-20 mean: total={head_total:.4f} traj={head_traj:.4f} revae={head_revae:.4f}")
    print(f"  last-20  mean: total={tail_total:.4f} traj={tail_traj:.4f} revae={tail_revae:.4f}")
    print(f"  reVAE grad max-abs across run: {revae_grad_max:.4e}")

    # --- assertions ---
    for name, series in [("total", total_losses), ("traj", traj_losses),
                         ("score", score_losses), ("revae", revae_losses)]:
        for v in series:
            assert v == v, f"{name} loss NaN encountered"  # NaN check via self-inequality
            assert v != float("inf") and v != float("-inf"), f"{name} loss inf"

    # 5% improvement on total + trajectory branches is a generous floor
    # for 100 iters at lr=1.5e-4 on a network with mostly fresh reVAE weights.
    assert tail_total < head_total * 0.95, (
        f"total loss not decreasing >=5%: head={head_total:.4f} tail={tail_total:.4f}"
    )
    assert tail_traj < head_traj * 0.95, (
        f"trajectory loss not decreasing >=5%: head={head_traj:.4f} tail={tail_traj:.4f}"
    )
    # reVAE must actually be receiving gradient (not a frozen passenger)
    assert revae_grad_max > 1e-6, f"reVAE grad essentially zero: {revae_grad_max:.4e}"

    print()
    print(f"[PASS] stage-1 over {args.max_steps} iters: total {head_total:.4f} -> {tail_total:.4f}")


if __name__ == "__main__":
    main()
