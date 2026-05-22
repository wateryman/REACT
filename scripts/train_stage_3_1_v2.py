"""Stage-3.1 v2 production training run -- adds the z-floor loss.

Same config as scripts/train_stage_3_1.py, plus:
  cfg["z_floor"]["enable"] = True

The z-floor cost prevents the model from exploiting the SafetyLoss/ESDF gap
below z=0.  See REACT_MATH_Derivations/01 §4 and the stage-5.B closed-loop
debug log: the C1-v1 trained model (no z-floor) consistently scored
"look-down" anchors lowest and dove the drone into the ground at SR=0/100.

Output: YOPO/saved/YOPO_<next>/epoch50.pth (plus epoch10/20/30/40 ckpts).

ETA on RTX 3070, batch=16: ~2.5 h.
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
# 🟦 stage-3.1 v2: turn on z-floor penalty
cfg._data["z_floor"]["enable"]                = True
cfg._data["z_floor"]["z_floor_m"]             = 0.3
cfg._data["z_floor"]["lam_floor"]             = 5.0

from policy.yopo_trainer import YopoTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--lr",     type=float, default=1.5e-4)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--save-interval", type=int, default=10)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    log_dir = os.path.join(YOPO_DIR, "saved")
    os.makedirs(log_dir, exist_ok=True)
    print("=" * 60)
    print("stage-3.1 v2 production training (with z-floor loss)")
    print(f"  epochs:        {args.epochs}")
    print(f"  z_floor.enable: {cfg['z_floor']['enable']}")
    print(f"  z_floor_m:      {cfg['z_floor']['z_floor_m']}")
    print(f"  lam_floor:      {cfg['z_floor']['lam_floor']}")
    print("=" * 60)

    trainer = YopoTrainer(
        learning_rate=args.lr,
        batch_size=args.batch,
        loss_weight=[1.0, 1.0],
        tensorboard_path=log_dir,
        checkpoint_path="",
        save_on_exit=True,
    )
    trainer.train(epoch=args.epochs, save_interval=args.save_interval)
    print(f"Run YOPO Finish! Checkpoint at {trainer.tensorboard_path}/epoch{args.epochs}.pth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
