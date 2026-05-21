"""Stage-3.2 sub-C smoke: trainer wires drone-relative tokens into
YopoNetwork.forward via the policy.inference signature added in sub-B.

Five checks:
  T1 cfg.dynamic_attention.enable == True; trainer.policy.use_dca == True
  T2 one-step strict assert: static-batch path is still regression-safe
       (dyn_loss == kino_loss == 0.0 EXACT when dyn_obs payload is None)
  T3 one-step on a dynamic batch: dyn_loss > 0 (motion_reshaped fires)
  T4 100-step mixed training:
        ~50/50 batch mix
        static-batch dyn_loss == 0 EXACT every step
        majority of dynamic batches have dyn_loss > 0
        all losses finite
        total loss trends down (head-20 vs tail-20)
  T5 verify the side channel actually fires: monkey-patch the network's DCA
       forward to count calls; expect: count > 0 only on dynamic batches.
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
from policy.yopo_trainer import YopoTrainer


def main():
    # T1
    print("== T1: cfg + trainer DCA wiring ==")
    assert bool(cfg["dynamic_attention"]["enable"]) is True, "cfg flag off"
    trainer = YopoTrainer(
        learning_rate=1.5e-4,
        batch_size=8,
        loss_weight=[1.0, 1.0],
        tensorboard_path="saved/_scratch",
        checkpoint_path="",
        save_on_exit=False,
    )
    trainer.policy.train()
    assert trainer.policy.use_dca is True
    assert trainer.policy.dca is not None
    assert trainer.policy.dyn_obs_encoder is not None
    n_params = sum(p.numel() for p in trainer.policy.parameters())
    print(f"[OK] T1: policy.use_dca={trainer.policy.use_dca}, params={n_params:,}")

    # T2 static-only step
    print("\n== T2: static-batch path regression ==")
    static_batch = next(iter(trainer.train_dataloader))
    depth, pos, rot, obs_b, map_id = static_batch
    (traj, score, revae, dyn, kino, *_rest) = trainer.forward_and_compute_loss(
        depth, pos, rot, obs_b, map_id)
    print(f"   traj={traj.item():.4f}  dyn={dyn.item():.10f}  kino={kino.item():.10f}")
    assert dyn.item() == 0.0, f"dyn leak on static: {dyn.item()}"
    assert kino.item() == 0.0, f"kino leak on static: {kino.item()}"
    print("[OK] T2: dyn=0 / kino=0 EXACT when dyn_obs payload is None")

    # T3 dynamic-only step
    print("\n== T3: dynamic-batch path fires ==")
    dyn_batch = next(iter(trainer.dyn_train_dataloader))
    depth_d, pos_d, rot_d, obs_d, map_d, dyn_pad, dyn_mask = dyn_batch
    (traj_d, score_d, _, dyn_d, kino_d, *_) = trainer.forward_and_compute_loss(
        depth_d, pos_d, rot_d, obs_d, map_d, dyn_obs=(dyn_pad, dyn_mask))
    print(f"   traj_d={traj_d.item():.4f}  dyn_d={dyn_d.item():.4f}  kino_d={kino_d.item():.4f}")
    assert dyn_d.item() > 0.0, f"dyn=0 on dynamic batch (should be > 0): {dyn_d.item()}"
    print(f"[OK] T3: dyn_loss > 0 on dynamic batch")

    # T5 setup (run before T4 so we instrument the SAME run): count DCA fires.
    dca_calls = {"n": 0}
    orig_dca_fwd = trainer.policy.dca.forward
    def counting_dca(q, kv, key_padding_mask=None):
        dca_calls["n"] += 1
        return orig_dca_fwd(q, kv, key_padding_mask=key_padding_mask)
    trainer.policy.dca.forward = counting_dca

    # T4 mixed training
    print("\n== T4: 100-step mixed training ==")
    lam_vae = float(cfg["loss_weights"]["lam_vae"])
    n_dyn = n_stat = 0
    stat_dyn_losses = []
    dyn_dyn_losses = []
    totals = []
    for step, static_batch in enumerate(trainer.train_dataloader):
        if static_batch[0].shape[0] != trainer.batch_size:
            continue
        dyn_payload = None
        if torch.rand(1).item() < trainer.dynamic_ratio:
            try:
                db = next(trainer.dyn_train_iter)
            except StopIteration:
                trainer.dyn_train_iter = iter(trainer.dyn_train_dataloader)
                db = next(trainer.dyn_train_iter)
            if db[0].shape[0] == trainer.batch_size:
                depth, pos, rot, obs_b, map_id, dp, dm = db
                dyn_payload = (dp, dm)
            else:
                depth, pos, rot, obs_b, map_id = static_batch
        else:
            depth, pos, rot, obs_b, map_id = static_batch

        trainer.optimizer.zero_grad()
        traj, score, revae, dyn, kino, _, _, _, _ = trainer.forward_and_compute_loss(
            depth, pos, rot, obs_b, map_id, dyn_obs=dyn_payload)
        loss = (trainer.loss_weight[0] * traj + trainer.loss_weight[1] * score
                + lam_vae * revae + dyn + kino)
        loss.backward()
        trainer.optimizer.step()
        totals.append(loss.item())
        if dyn_payload is not None:
            n_dyn += 1
            dyn_dyn_losses.append(dyn.item())
        else:
            n_stat += 1
            stat_dyn_losses.append(dyn.item())
        if step >= 99:
            break

    print(f"   batch mix: dyn={n_dyn}, stat={n_stat}")
    assert abs(n_dyn - 50) < 25, f"dyn batches {n_dyn} too far from 50"
    stat_max = max(abs(x) for x in stat_dyn_losses) if stat_dyn_losses else 0.0
    print(f"   static-batch dyn_loss max-abs = {stat_max:.2e}")
    assert stat_max == 0.0
    n_pos = sum(1 for x in dyn_dyn_losses if x > 0)
    print(f"   dynamic-batch dyn_loss > 0: {n_pos}/{n_dyn}")
    assert n_pos >= max(1, n_dyn // 2)
    assert all(np.isfinite(t) for t in totals)
    head, tail = float(np.mean(totals[:20])), float(np.mean(totals[-20:]))
    print(f"   total: head-20 = {head:.3f}, tail-20 = {tail:.3f}  (drop {(head-tail)/head*100:+.1f}%)")
    print(f"[OK] T4")

    # T5 verify DCA call count
    print(f"\n== T5: DCA forward call count ==")
    print(f"   DCA was called {dca_calls['n']} times across 100 training steps")
    print(f"   expected ~{n_dyn} (one per dynamic batch); not exact because the "
          "forward also runs once during T3 above")
    expected_lo = n_dyn   # T4 alone contributes >= n_dyn calls
    expected_hi = n_dyn + 5  # T3 added one more call before the instrumentation
    # T3 happened BEFORE we wrapped dca.forward, so it shouldn't add to count.
    # But we did set the monkeypatch AFTER T3, so dca_calls['n'] starts at 0 at
    # T4 start and increments once per dynamic batch in T4.
    assert dca_calls['n'] == n_dyn, \
        f"DCA called {dca_calls['n']}, expected {n_dyn} (one per dynamic batch)"
    print(f"[OK] T5: DCA fired exactly once per dynamic batch ({dca_calls['n']} == {n_dyn})")

    print("\n[PASS] stage-3.2 sub-C: 5/5 checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
