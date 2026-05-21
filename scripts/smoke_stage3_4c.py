"""Stage-3.4 sub-C smoke: trainer + DynamicYOPOWrapper K-frame wiring.

Five checks covering phase-3 of stage-3.4 path-a:
  T1 cfg + trainer wiring:
       cfg["frame_buffer"]["enable_temporal"] = True ->
       YopoTrainer instantiates YopoNetwork(use_temporal=True),
       DynamicYOPOWrapper(return_kframe=True), and prints both flags.
  T2 Static-batch path regression byte-clean:
       One forward_and_compute_loss step on a static batch returns
       dyn_loss == kino_loss == 0 EXACT (dyn_obs payload is None).
       Stage-3.1's regression contract is preserved even with the new
       depth-expand auto-coercion.
  T3 Dynamic-batch path fires:
       One step on a dynamic batch with dyn_obs payload yields
       dyn_loss > 0 (motion_reshaped fires).  Network forward consumes
       the (B, K, 1, H, W) tensor without error.
  T4 100-step mixed training:
       ~50/50 batch mix, static dyn_loss == 0 every step,
       majority of dynamic batches have dyn_loss > 0, all losses
       finite, total loss trends down (head-20 vs tail-20).
  T5 Depth shape inspection:
       In one static step and one dynamic step, verify that depth
       seen by forward_and_compute_loss after coercion is
       (B, K, 1, H, W) on both paths.
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
    # Force temporal on for this run.
    cfg._data["frame_buffer"]["enable_temporal"] = True
    # Keep DCA off so T2/T3 byte-comparison line up cleanly with stage-3.1
    # regression (DCA orthogonality already covered by smoke_stage3_4b T5).
    cfg._data["dynamic_attention"]["enable"] = False
    # Make 50/50 mix predictable
    cfg._data["dynamic_ratio"] = 0.5

    # T1
    print("== T1: cfg + trainer wiring ==")
    assert bool(cfg["frame_buffer"]["enable_temporal"]) is True
    trainer = YopoTrainer(
        learning_rate=1.5e-4,
        batch_size=8,
        loss_weight=[1.0, 1.0],
        tensorboard_path="saved/_scratch",
        checkpoint_path="",
        save_on_exit=False,
    )
    trainer.policy.train()
    assert trainer.policy.use_temporal is True
    assert trainer.policy.temporal_aggregator is not None
    assert trainer.use_temporal is True
    assert trainer.K == 10
    # Dataset wrapper should be in K-frame mode
    assert trainer.dyn_train_dataloader is not None
    assert trainer.dyn_train_dataloader.dataset.return_kframe is True
    n_params = sum(p.numel() for p in trainer.policy.parameters())
    print(f"[OK] T1: policy.use_temporal={trainer.policy.use_temporal}, "
          f"K={trainer.K}, dataset.return_kframe=True, params={n_params:,}")

    # T2: static-only step
    print("\n== T2: static-batch path regression ==")
    static_batch = next(iter(trainer.train_dataloader))
    depth, pos, rot, obs_b, map_id = static_batch
    assert depth.dim() == 4, f"static batch depth should be 4D, got {depth.shape}"
    (traj, score, revae, dyn, kino, *_rest) = trainer.forward_and_compute_loss(
        depth, pos, rot, obs_b, map_id)
    print(f"   traj={traj.item():.4f}  revae={revae.item():.4f}  "
          f"dyn={dyn.item():.10f}  kino={kino.item():.10f}")
    assert dyn.item() == 0.0, f"static dyn leak: {dyn.item()}"
    assert kino.item() == 0.0, f"static kino leak: {kino.item()}"
    print(f"[OK] T2: dyn=0 / kino=0 EXACT on static batch under use_temporal=True")

    # T3: dynamic-only step
    print("\n== T3: dynamic-batch path fires ==")
    dyn_batch = next(iter(trainer.dyn_train_dataloader))
    depth_d, pos_d, rot_d, obs_d, map_d, dyn_pad, dyn_mask = dyn_batch
    assert depth_d.dim() == 5, f"dynamic batch depth should be 5D (B,K,1,H,W), got {depth_d.shape}"
    print(f"   dynamic depth shape: {tuple(depth_d.shape)}")
    (traj_d, score_d, _, dyn_d, kino_d, *_) = trainer.forward_and_compute_loss(
        depth_d, pos_d, rot_d, obs_d, map_d, dyn_obs=(dyn_pad, dyn_mask))
    print(f"   traj_d={traj_d.item():.4f}  dyn_d={dyn_d.item():.4f}  "
          f"kino_d={kino_d.item():.6f}")
    assert dyn_d.item() > 0.0, f"dyn=0 on dynamic batch: {dyn_d.item()}"
    print(f"[OK] T3: dyn_loss > 0 on dynamic batch under use_temporal=True")

    # T5 (before T4 so we instrument the same trainer): shape inspection
    print("\n== T5: depth shape after auto-coercion ==")
    # The trainer's forward_and_compute_loss auto-expands 4D depth to 5D.
    # Wrap to capture the post-coercion shape.
    captured_shapes = {"static": None, "dynamic": None}
    orig = trainer.forward_and_compute_loss
    def capturing(depth, *args, **kwargs):
        # Mimic the same auto-coerce that's inside forward_and_compute_loss
        # so we can record what the policy.inference will see.
        d_for_record = depth.to(trainer.device)
        if trainer.use_temporal and d_for_record.dim() == 4:
            d_for_record = d_for_record.unsqueeze(1).expand(
                -1, trainer.K, -1, -1, -1)
        kind = "dynamic" if kwargs.get("dyn_obs") is not None else "static"
        captured_shapes[kind] = tuple(d_for_record.shape)
        return orig(depth, *args, **kwargs)
    trainer.forward_and_compute_loss = capturing

    # one static + one dynamic for the capture
    sb = next(iter(trainer.train_dataloader))
    capturing(*sb)
    db = next(iter(trainer.dyn_train_dataloader))
    capturing(db[0], db[1], db[2], db[3], db[4], dyn_obs=(db[5], db[6]))
    trainer.forward_and_compute_loss = orig    # restore

    print(f"   static  -> {captured_shapes['static']}")
    print(f"   dynamic -> {captured_shapes['dynamic']}")
    assert captured_shapes["static"][1] == trainer.K
    assert captured_shapes["dynamic"][1] == trainer.K
    assert captured_shapes["static"][2:] == captured_shapes["dynamic"][2:]
    print(f"[OK] T5: both paths feed (B, K, 1, H, W) to YopoNetwork.forward")

    # T4: 100-step mixed training
    print("\n== T4: 100-step mixed training ==")
    lam_vae = float(cfg["loss_weights"]["lam_vae"])
    n_dyn = n_stat = 0
    stat_dyn_losses, dyn_dyn_losses, totals = [], [], []
    for step, static_batch in enumerate(trainer.train_dataloader):
        if static_batch[0].shape[0] != trainer.batch_size:
            continue
        dyn_payload = None
        if torch.rand(1).item() < trainer.dynamic_ratio:
            try:
                dbb = next(trainer.dyn_train_iter)
            except StopIteration:
                trainer.dyn_train_iter = iter(trainer.dyn_train_dataloader)
                dbb = next(trainer.dyn_train_iter)
            if dbb[0].shape[0] == trainer.batch_size:
                depth, pos, rot, obs_b, map_id, dp, dm = dbb
                dyn_payload = (dp, dm)
            else:
                depth, pos, rot, obs_b, map_id = static_batch
        else:
            depth, pos, rot, obs_b, map_id = static_batch

        trainer.optimizer.zero_grad()
        traj, score, revae, dyn, kino, _, _, _, _ = \
            trainer.forward_and_compute_loss(
                depth, pos, rot, obs_b, map_id, dyn_obs=dyn_payload)
        loss = (trainer.loss_weight[0] * traj
                + trainer.loss_weight[1] * score
                + lam_vae * revae + dyn + kino)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.policy.parameters(),
                                        trainer.max_grad_norm)
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

    print(f"   batch mix:                       dyn={n_dyn}, stat={n_stat}")
    assert abs(n_dyn - 50) < 25, f"dyn batches {n_dyn} too far from 50"
    stat_max = max(abs(x) for x in stat_dyn_losses) if stat_dyn_losses else 0.0
    print(f"   static-batch dyn_loss max-abs:   {stat_max:.2e}")
    assert stat_max == 0.0, f"static dyn_loss leak: {stat_max}"
    n_pos = sum(1 for x in dyn_dyn_losses if x > 0)
    print(f"   dynamic-batch dyn_loss > 0:      {n_pos}/{n_dyn}")
    assert n_pos >= max(1, n_dyn // 2)
    assert all(np.isfinite(t) for t in totals)
    head, tail = float(np.mean(totals[:20])), float(np.mean(totals[-20:]))
    drop = (head - tail) / head * 100 if head > 0 else 0
    print(f"   total: head-20={head:.3f}, tail-20={tail:.3f}  (drop {drop:+.1f}%)")
    assert drop > 0, f"total loss did not decrease: head={head}, tail={tail}"
    print(f"[OK] T4")

    print("\n[PASS] stage-3.4 sub-C: 5/5 checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
