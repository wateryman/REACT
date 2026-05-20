"""Run N full epochs of the REACT stage-1 (PEMTRS reVAE) training and
write tensorboard logs to YOPO/saved/.

This is a thin shim around train_yopo.py's logic so we can run any N
without editing the original entry point. Tensorboard event files end
up at YOPO/saved/YOPO_<next_index>/.

Usage (from REACT/ root):
    python scripts/run_one_epoch.py --epochs 1 --batch 16
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
YOPO_DIR = os.path.join(HERE, "..", "YOPO")
os.chdir(YOPO_DIR)
sys.path.insert(0, YOPO_DIR)

from policy.yopo_trainer import YopoTrainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    args = ap.parse_args()

    log_dir = os.path.join(YOPO_DIR, "saved")
    os.makedirs(log_dir, exist_ok=True)

    print(f"== run_one_epoch | epochs={args.epochs} batch={args.batch} lr={args.lr} ==")
    trainer = YopoTrainer(
        learning_rate=args.lr,
        batch_size=args.batch,
        loss_weight=[1.0, 1.0],
        tensorboard_path=log_dir,
        checkpoint_path="",
        save_on_exit=False,
    )
    trainer.train(epoch=args.epochs)
    print(f"[DONE] tb_dir={trainer.tensorboard_path}")


if __name__ == "__main__":
    main()
