"""Stage-3.1 fine-tune from upstream YOPO baseline checkpoint.

D1 of the 4-week sprint plan (see /home/wxs/.claude/plans/reflective-enchanting-steele.md).

Strategy:
  1. Build a REACT-shaped network (use_revae=True, use_dca=False,
     use_temporal=False) -- same architecture as train_stage_3_1_v2.py.
  2. Load YOPO_1/epoch50.pth (upstream baseline, 11.3 M params, NO reVAE keys)
     with strict=False.  Missing reVAE keys stay at random init; existing
     backbone/head/state_backbone keys get the well-trained baseline weights.
  3. Train 5 epoch (warmup) + 20 epoch (full) with z_floor + dyn loss on v3.
  4. Save checkpoints; we'll test C1-FT vs baseline on closed-loop SR.

Hypothesis (from §3 of the plan): fine-tuning preserves the baseline's
strong static-obstacle handling (7 % static_col vs C1-v2's 24 %) while
the dyn loss adds dynamic-obstacle awareness on top, hopefully landing
the SR at 35-45 % (vs baseline 31 %, vs C1-v2 31 %).

Gate at D1 evening: if C1-FT Pass-2 SR < 35 %, pivot to score-head
decoupling (D4 task moved up).

ETA: ~25 min for 5-epoch warmup; ~1.5 h if full 25 epoch.
"""
import os
import sys
import random
import argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REACT_ROOT = os.path.dirname(HERE)
YOPO_DIR = os.path.join(REACT_ROOT, "YOPO")
os.chdir(YOPO_DIR)
sys.path.insert(0, YOPO_DIR)

# Mutate cfg BEFORE importing yopo_trainer (which captures the singleton).
from config.config import cfg
cfg._data["revae"]["enable"]                  = True
cfg._data["dynamic_attention"]["enable"]      = False
cfg._data["frame_buffer"]["enable_temporal"]  = False
cfg._data["dynamic_ratio"]                    = 0.5
cfg._data["loss_weights"]["lam_dyn"]          = 3.0
cfg._data["loss_weights"]["lam_kino"]         = 0.5
# z_floor on (matches train_stage_3_1_v2.py)
cfg._data["z_floor"]["enable"]                = True
cfg._data["z_floor"]["z_floor_m"]             = 0.3
cfg._data["z_floor"]["lam_floor"]             = 5.0

from policy.yopo_trainer import YopoTrainer
from policy.yopo_network import YopoNetwork


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25,
                    help="total fine-tune epochs (5 warmup + 20 full)")
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--lr",     type=float, default=5e-5,
                    help="fine-tune LR; 3x lower than train_stage_3_1_v2's 1.5e-4")
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--save-interval", type=int, default=5)
    ap.add_argument("--base-ckpt", type=str,
                    default="saved/YOPO_1/epoch50.pth",
                    help="path (under YOPO/) to upstream baseline checkpoint")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    log_dir = os.path.join(YOPO_DIR, "saved")
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("stage-3.1 FINE-TUNE from upstream YOPO baseline")
    print(f"  base ckpt:     {args.base_ckpt}")
    print(f"  epochs:        {args.epochs}")
    print(f"  lr:            {args.lr}  (lower than train_stage_3_1_v2 1.5e-4)")
    print(f"  z_floor:       enable={cfg['z_floor']['enable']}, m={cfg['z_floor']['z_floor_m']}, lam={cfg['z_floor']['lam_floor']}")
    print(f"  dynamic_ratio: {cfg['dynamic_ratio']}")
    print(f"  lam_dyn:       {cfg['loss_weights']['lam_dyn']}")
    print(f"  lam_kino:      {cfg['loss_weights']['lam_kino']}")
    print("=" * 60)

    # We can't reuse YopoTrainer's load_state_dict (it's strict=True) because
    # the baseline ckpt has no reVAE keys.  We construct YopoTrainer with
    # checkpoint_path="" (skip its loader), then manually load with strict=False.
    trainer = YopoTrainer(
        learning_rate=args.lr,
        batch_size=args.batch,
        loss_weight=[1.0, 1.0],
        tensorboard_path=log_dir,
        checkpoint_path="",        # do NOT let trainer's strict loader run
        save_on_exit=True,
    )

    # Manual non-strict load with input-channel zero-padding for yopo_head.
    # The baseline ckpt has head_in = 64 (depth) + 9 (obs) = 73, but our
    # REACT network adds the 128-dim reVAE latent on top -> head_in = 201.
    # yopo_head.model.0 is a 1x1 conv whose weight shape is (256, head_in, 1, 1)
    # so the first weight tensor mismatches.  We zero-pad the extra 128
    # reVAE channels: the head retains baseline's learned filters on
    # (obs, depth) channels and initially ignores reVAE; training learns
    # the reVAE channel weights from scratch.
    base_path = args.base_ckpt
    if not os.path.isabs(base_path):
        base_path = os.path.join(YOPO_DIR, base_path)
    print(f"\nLoading base checkpoint (strict=False): {base_path}")
    state = torch.load(base_path, weights_only=True, map_location=trainer.device)

    head_key = "yopo_head.model.0.weight"
    if head_key in state:
        old_w = state[head_key]                              # (256, 73, 1, 1)
        target_shape = trainer.policy.state_dict()[head_key].shape  # (256, 201, 1, 1)
        if old_w.shape != target_shape:
            print(f"  yopo_head.model.0.weight zero-pad: {tuple(old_w.shape)} -> {tuple(target_shape)}")
            new_w = torch.zeros(target_shape, dtype=old_w.dtype, device=old_w.device)
            new_w[:, : old_w.shape[1]] = old_w               # baseline channels at front
            state[head_key] = new_w

    missing, unexpected = trainer.policy.load_state_dict(state, strict=False)
    print(f"  loaded; {len(missing)} missing keys (random-init, mostly reVAE/state_backbone),")
    print(f"          {len(unexpected)} unexpected keys (should be 0).")
    if unexpected:
        print(f"  unexpected sample: {unexpected[:3]}")
    if missing:
        print(f"  missing sample:    {missing[:5]}")

    trainer.train(epoch=args.epochs, save_interval=args.save_interval)
    print(f"\nRun YOPO Finish! Final checkpoint at {trainer.tensorboard_path}/epoch{args.epochs}.pth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
