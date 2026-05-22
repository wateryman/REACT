"""Stage-3.1 production training run (50 epoch, save checkpoint).

This is the headline stage-3.1 config (path-c, loss-only):
  reVAE on, DCA off, temporal off, dynamic_ratio 0.5,
  lam_dyn=3.0, lam_kino=0.5  (cfg defaults under v3 dataset)

Output:
  YOPO/saved/YOPO_<N>/                     <- next free counter under saved/
    events.out.tfevents.*                  <- TensorBoard scalars
    epoch50.pth                            <- final checkpoint
    epoch10.pth, epoch20.pth, ...          <- intermediate (every 10)

The trainer's atexit hook (save_on_exit=True) also drops a final ckpt
if interrupted.

ETA on RTX 3070, batch=16: ~2.5 h.

Run from REACT/ root:
  python scripts/train_stage_3_1.py 2>&1 | tee /tmp/train_stage_3_1.log

Inspect:
  tensorboard --logdir YOPO/saved --port 6006
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
cfg._data["dynamic_attention"]["enable"]      = False         # stage-3.1 has no DCA
cfg._data["frame_buffer"]["enable_temporal"]  = False         # stage-3.1 single-frame
cfg._data["dynamic_ratio"]                    = 0.5
cfg._data["loss_weights"]["lam_dyn"]          = 3.0
cfg._data["loss_weights"]["lam_kino"]         = 0.5

from policy.yopo_trainer import YopoTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--lr",     type=float, default=1.5e-4)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--save-interval", type=int, default=10,
                    help="save a checkpoint every N epochs (default 10)")
    args = ap.parse_args()

    # Determinism (best-effort; dataloader workers add nondeterminism)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    log_dir = os.path.join(YOPO_DIR, "saved")
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("stage-3.1 production training")
    print(f"  epochs:        {args.epochs}")
    print(f"  batch:         {args.batch}")
    print(f"  lr:            {args.lr}")
    print(f"  save_interval: {args.save_interval}")
    print(f"  dataset:       {cfg['dataset_path']} (static)")
    print(f"               + {cfg['dataset_dynamic_path']} (dynamic, ratio {cfg['dynamic_ratio']})")
    print(f"  reVAE:         {cfg['revae']['enable']}")
    print(f"  DCA:           {cfg['dynamic_attention']['enable']}  (stage-3.1 off)")
    print(f"  temporal:      {cfg['frame_buffer'].get('enable_temporal', False)}  (stage-3.1 off)")
    print(f"  lam_dyn:       {cfg['loss_weights']['lam_dyn']}")
    print(f"  lam_kino:      {cfg['loss_weights']['lam_kino']}")
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
