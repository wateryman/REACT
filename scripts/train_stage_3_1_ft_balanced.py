"""Stage-3.1 fine-tune balanced (D5): lower lam_dyn to trade dyn for static recovery.

D4 result analysis:
  C1-FT v3 (lam_dyn=3.0, v3 data):  goal 36%, dyn 2%, static 10%   (best goal)
  C1-FT v4 (lam_dyn=3.0, v4 data):  goal 29%, dyn 0%, static 19%   (best dyn, but slow)

Hypothesis: with v4's strong dyn gradient signal (88% FOV-presence), lam_dyn=3.0
is too aggressive -- the model over-prioritizes ball avoidance at the cost of
goal pursuit and static obstacles.  Halving lam_dyn to 1.5 should rebalance:
expected dyn_col 0-1%, goal SR 33-38%.

All other cfg identical to train_stage_3_1_ft.py (z_floor on, dyn_ratio 0.5,
lam_kino 0.5, 25 epoch fine-tune from YOPO_1/epoch50.pth, LR 5e-5).
v4 data is already wired into traj_opt.yaml.

ETA ~1.3 h.
"""
import os, sys, random, argparse
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
REACT_ROOT = os.path.dirname(HERE)
YOPO_DIR = os.path.join(REACT_ROOT, "YOPO")
os.chdir(YOPO_DIR)
sys.path.insert(0, YOPO_DIR)

from config.config import cfg
cfg._data["revae"]["enable"]                  = True
cfg._data["dynamic_attention"]["enable"]      = False
cfg._data["frame_buffer"]["enable_temporal"]  = False
cfg._data["dynamic_ratio"]                    = 0.5
cfg._data["loss_weights"]["lam_dyn"]          = 1.5     # <-- D5 change: was 3.0
cfg._data["loss_weights"]["lam_kino"]         = 0.5
cfg._data["z_floor"]["enable"]                = True
cfg._data["z_floor"]["z_floor_m"]             = 0.3
cfg._data["z_floor"]["lam_floor"]             = 5.0

from policy.yopo_trainer import YopoTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--lr",     type=float, default=5e-5)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--save-interval", type=int, default=5)
    ap.add_argument("--base-ckpt", type=str, default="saved/YOPO_1/epoch50.pth")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    log_dir = os.path.join(YOPO_DIR, "saved")
    os.makedirs(log_dir, exist_ok=True)
    print("=" * 60)
    print("D5: stage-3.1 fine-tune BALANCED (lam_dyn 3.0 -> 1.5)")
    print(f"  base ckpt:     {args.base_ckpt}")
    print(f"  epochs:        {args.epochs}")
    print(f"  lr:            {args.lr}")
    print(f"  lam_dyn:       {cfg['loss_weights']['lam_dyn']}  (D4 used 3.0)")
    print(f"  lam_kino:      {cfg['loss_weights']['lam_kino']}")
    print(f"  dataset:       {cfg['dataset_dynamic_path']}")
    print("=" * 60)

    trainer = YopoTrainer(
        learning_rate=args.lr, batch_size=args.batch,
        loss_weight=[1.0, 1.0],
        tensorboard_path=log_dir, checkpoint_path="",
        save_on_exit=True,
    )

    # zero-pad baseline yopo_head (same trick as train_stage_3_1_ft.py)
    base_path = args.base_ckpt
    if not os.path.isabs(base_path):
        base_path = os.path.join(YOPO_DIR, base_path)
    print(f"\nLoading base checkpoint (strict=False): {base_path}")
    state = torch.load(base_path, weights_only=True, map_location=trainer.device)
    head_key = "yopo_head.model.0.weight"
    if head_key in state:
        old_w = state[head_key]
        target_shape = trainer.policy.state_dict()[head_key].shape
        if old_w.shape != target_shape:
            print(f"  yopo_head zero-pad: {tuple(old_w.shape)} -> {tuple(target_shape)}")
            new_w = torch.zeros(target_shape, dtype=old_w.dtype, device=old_w.device)
            new_w[:, : old_w.shape[1]] = old_w
            state[head_key] = new_w
    missing, unexpected = trainer.policy.load_state_dict(state, strict=False)
    print(f"  loaded; {len(missing)} missing, {len(unexpected)} unexpected.")

    trainer.train(epoch=args.epochs, save_interval=args.save_interval)
    print(f"\nRun YOPO Finish! Final checkpoint at {trainer.tensorboard_path}/epoch{args.epochs}.pth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
