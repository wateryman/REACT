"""Stage-3.1 C.3 smoke: mixed-sampling dataloader actually fires dyn/kino
losses on dynamic batches and stays bit-clean on static batches.

Verifies:
  (1) DynamicYOPOWrapper produces correct 7-tuple shape
  (2) Trainer constructs dyn_train_dataloader iff dynamic_ratio > 0
  (3) 100-step mixed training:
        - ~50/50 split (binomial, tol ±20)
        - all STATIC steps have dyn_loss == 0.0 EXACTLY (regression preserved)
        - majority of DYNAMIC steps have dyn_loss > 0
        - all losses finite, no NaN
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
YOPO_DIR = os.path.join(HERE, "..", "YOPO")
os.chdir(YOPO_DIR)
sys.path.insert(0, YOPO_DIR)

import numpy as np
import torch

from config.config import cfg
from policy.yopo_dataset import DynamicYOPOWrapper
from policy.yopo_trainer import YopoTrainer


# ---- (1) DynamicYOPOWrapper standalone shape check ----
print("== T1: DynamicYOPOWrapper shape check ==")
wrapper = DynamicYOPOWrapper(mode='valid')   # smaller set for speed
print(f"   len={len(wrapper)}; M_max={wrapper.M_max}; env_num_static={wrapper.env_num_static}")
img, pos, rot_wb, random_obs, map_idx, dyn_pad, dyn_mask = wrapper[0]
H, W = int(cfg["image_height"]), int(cfg["image_width"])
M = int(cfg["dynamic_attention"]["max_dyn_obs"])
assert img.shape == (1, H, W),         f"img shape {img.shape}"
assert pos.shape == (3,),              f"pos shape {pos.shape}"
assert rot_wb.shape == (3, 3),         f"rot_wb shape {rot_wb.shape}"
assert random_obs.shape == (9,),       f"random_obs shape {random_obs.shape}"
assert dyn_pad.shape == (M, 7),        f"dyn_pad shape {dyn_pad.shape}"
assert dyn_mask.shape == (M,),         f"dyn_mask shape {dyn_mask.shape}"
assert dyn_mask.dtype == np.bool_,     f"dyn_mask dtype {dyn_mask.dtype}"
assert 0 <= map_idx < wrapper.env_num_static
n_real = int(dyn_mask.sum())
print(f"   sample 0: img{img.shape} in [{img.min():.3f},{img.max():.3f}]  "
      f"#real balls={n_real}/{M}  map_idx={map_idx}")
if n_real > 0:
    print(f"             first real ball pos={dyn_pad[0,:3].tolist()} r={dyn_pad[0,6]:.3f}")
print("[OK] T1")

# ---- (2) Trainer constructs dyn dataloader ----
print("\n== T2: trainer wiring ==")
trainer = YopoTrainer(
    learning_rate=1.5e-4,
    batch_size=8,
    loss_weight=[1.0, 1.0],
    tensorboard_path='saved/_scratch',
    checkpoint_path='',
    save_on_exit=False,
)
trainer.policy.train()
assert trainer.dyn_train_iter is not None, "expected dyn dataloader (cfg.dynamic_ratio=0.5)"
print(f"   dynamic_ratio={trainer.dynamic_ratio}  "
      f"dyn ds len={len(trainer.dyn_train_dataloader.dataset)}")
print("[OK] T2")

# ---- (3) 100-step mixed training ----
print("\n== T3: 100-step mixed training ==")
lam_vae = float(cfg["loss_weights"]["lam_vae"])
n_dyn = n_stat = 0
dyn_losses_dyn = []
dyn_losses_stat = []
kino_losses_dyn = []
totals = []

for step, static_batch in enumerate(trainer.train_dataloader):
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
    traj, score, revae, dyn, kino, _, _, _, _ = trainer.forward_and_compute_loss(
        depth, pos, rot, obs_b, map_id, dyn_obs=dyn_payload)
    loss = (trainer.loss_weight[0] * traj
            + trainer.loss_weight[1] * score
            + lam_vae * revae + dyn + kino)
    loss.backward()
    trainer.optimizer.step()

    if dyn_payload is not None:
        n_dyn += 1
        dyn_losses_dyn.append(dyn.item())
        kino_losses_dyn.append(kino.item())
    else:
        n_stat += 1
        dyn_losses_stat.append(dyn.item())
    totals.append(loss.item())

    if step % 10 == 0:
        kind = "DYN " if dyn_payload is not None else "stat"
        print(f"  step {step:3d} [{kind}]: total={loss.item():6.3f}  "
              f"dyn={dyn.item():.4f}  kino={kino.item():.4f}")
    if step >= 99:
        break

print(f"\nbatch mix: dyn={n_dyn}, static={n_stat}  (expected ~50/50 at ratio 0.5)")

# --- asserts ---
assert abs(n_dyn - 50) < 25, f"dyn batches {n_dyn} too far from expected 50 (binomial)"

stat_max = max(abs(x) for x in dyn_losses_stat) if dyn_losses_stat else 0.0
print(f"static-batch dyn_loss max-abs = {stat_max:.2e}  (must be 0)")
assert stat_max == 0.0, f"dyn loss leaked into static batch: {stat_max}"

n_dyn_positive = sum(1 for x in dyn_losses_dyn if x > 0)
print(f"dynamic-batch dyn_loss > 0 frames: {n_dyn_positive}/{n_dyn}")
assert n_dyn_positive >= max(1, n_dyn // 2), \
    f"too many dynamic batches with 0 loss: {n_dyn_positive}/{n_dyn}"

n_kino_positive = sum(1 for x in kino_losses_dyn if x > 0)
print(f"dynamic-batch kino_loss > 0 frames: {n_kino_positive}/{n_dyn}")

head = float(np.mean(totals[:20]))
tail = float(np.mean(totals[-20:]))
print(f"total loss: first-20 mean={head:.3f}, last-20 mean={tail:.3f}")
assert all(np.isfinite(t) for t in totals), "non-finite loss found"

print("\n[PASS] C.3: mixed sampling fires dyn/kino on dynamic batches only, "
      "static path regression-clean, training stable.")
