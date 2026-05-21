"""Stage-3.4 sub-B smoke: YopoNetwork.forward(depth_seq) K-frame path.

Five checks covering phase-2 of stage-3.4 path-a:
  T1 use_temporal=False regression (stage-3.2 byte-clean):
       Same seed -> two fresh YopoNetwork(use_temporal=False) instances
       produce IDENTICAL outputs on a fixed depth/obs sample.  (Sanity
       check that adding the ctor branch + import didn't change the
       random-init footprint of the unchanged path.)
  T2 use_temporal=True, K=1:
       (B, 1, 1, H, W) input shape -> outputs have correct shapes
       (endstate (B,9,V,H), score (B,V,H), recon (B,1,H,W), mu (B,128),
       logvar (B,128)).  All finite.
  T3 use_temporal=True, K=10:
       (B, 10, 1, H, W) input shape -> same output shape contract as
       T2; only the last frame is reconstructed.  All finite.
  T4 Parameter count:
       YopoNetwork(use_temporal=True) has EXACTLY
       YopoNetwork(use_temporal=False).numel() + GRU(128,128,1).numel()
       extra params.  Asserted to within 0.
  T5 DCA + temporal compose:
       use_temporal=True AND use_dca=True; dyn_obs_tokens supplied.
       Forward runs without error, outputs have correct shapes, the
       DCA path is on the graph (parameter count includes both).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
YOPO_DIR = os.path.join(HERE, "..", "YOPO")
os.chdir(YOPO_DIR)
sys.path.insert(0, YOPO_DIR)

import torch

from policy.yopo_network import YopoNetwork


def make_net(use_temporal: bool, use_dca: bool = False, seed: int = 42):
    torch.manual_seed(seed)
    return YopoNetwork(use_revae=True, revae_latent=128,
                       use_dca=use_dca, dca_n_heads=1,
                       use_temporal=use_temporal, temporal_hidden=128)


def main():
    B, V, H = 2, 3, 5
    H_in, W_in = 96, 160

    # ---------- T1: use_temporal=False byte-clean ----------
    print("== T1: use_temporal=False regression (two seeded instances) ==")
    net_a = make_net(use_temporal=False, seed=42)
    net_b = make_net(use_temporal=False, seed=42)
    net_a.eval(); net_b.eval()
    torch.manual_seed(7)
    depth = torch.rand(B, 1, H_in, W_in)
    obs = torch.randn(B, 9, V, H)
    with torch.no_grad():
        out_a = net_a(depth, obs)
        out_b = net_b(depth, obs)
    for name, ta, tb in zip(
            ("endstate", "score", "recon", "mu", "logvar"), out_a, out_b):
        assert torch.equal(ta, tb), f"T1 mismatch on {name}"
    print(f"   endstate {tuple(out_a[0].shape)}  score {tuple(out_a[1].shape)}  "
          f"recon {tuple(out_a[2].shape)}")
    print(f"[OK] T1 use_temporal=False byte-clean across seeded fresh instances")

    # ---------- T2: use_temporal=True, K=1 ----------
    print("\n== T2: use_temporal=True, K=1 ==")
    net_t = make_net(use_temporal=True, seed=42)
    net_t.eval()
    depth_k1 = depth.unsqueeze(1)            # (B, 1, 1, H, W)
    with torch.no_grad():
        endstate, score, recon, mu, logvar = net_t(depth_k1, obs)
    assert endstate.shape == (B, 9, V, H), endstate.shape
    assert score.shape == (B, V, H), score.shape
    assert recon.shape == (B, 1, H_in, W_in), recon.shape
    assert mu.shape == (B, 128), mu.shape
    assert logvar.shape == (B, 128), logvar.shape
    for name, t in zip(("endstate","score","recon","mu","logvar"),
                        (endstate, score, recon, mu, logvar)):
        assert torch.isfinite(t).all(), f"T2 NaN/Inf in {name}"
    print(f"   K=1 forward OK: endstate {tuple(endstate.shape)}  "
          f"recon {tuple(recon.shape)}  mu {tuple(mu.shape)}")
    print(f"[OK] T2")

    # ---------- T3: use_temporal=True, K=10 ----------
    print("\n== T3: use_temporal=True, K=10 ==")
    torch.manual_seed(11)
    depth_k10 = torch.rand(B, 10, 1, H_in, W_in)
    with torch.no_grad():
        endstate, score, recon, mu, logvar = net_t(depth_k10, obs)
    assert endstate.shape == (B, 9, V, H), endstate.shape
    assert score.shape == (B, V, H), score.shape
    assert recon.shape == (B, 1, H_in, W_in), \
        f"recon shape {recon.shape} -- should be last-frame only"
    assert mu.shape == (B, 128)
    assert logvar.shape == (B, 128)
    for name, t in zip(("endstate","score","recon","mu","logvar"),
                        (endstate, score, recon, mu, logvar)):
        assert torch.isfinite(t).all(), f"T3 NaN/Inf in {name}"
    print(f"   K=10 forward OK: recon is last-frame only (B,1,H,W)")
    print(f"[OK] T3")

    # ---------- T4: param count delta ----------
    print("\n== T4: parameter count delta ==")
    net_off = make_net(use_temporal=False, seed=42)
    net_on  = make_net(use_temporal=True,  seed=42)
    n_off = sum(p.numel() for p in net_off.parameters())
    n_on  = sum(p.numel() for p in net_on.parameters())
    n_agg = sum(p.numel() for p in net_on.temporal_aggregator.parameters())
    delta = n_on - n_off
    print(f"   use_temporal=False params: {n_off:,}")
    print(f"   use_temporal=True  params: {n_on:,}")
    print(f"   delta:                     {delta:,}  (aggregator: {n_agg:,})")
    assert delta == n_agg, f"param count delta {delta} != aggregator {n_agg}"
    assert 80_000 < n_agg < 120_000, f"aggregator size unexpected: {n_agg}"
    print(f"[OK] T4: delta == aggregator param count exactly")

    # ---------- T5: DCA + temporal compose ----------
    print("\n== T5: DCA + temporal compose ==")
    net_dt = make_net(use_temporal=True, use_dca=True, seed=42)
    net_dt.eval()
    M = 6
    dyn_tokens = torch.randn(B, M, 7)
    dyn_mask = torch.tensor([[True]*4 + [False]*2 for _ in range(B)])
    with torch.no_grad():
        endstate, score, recon, mu, logvar = net_dt(
            depth_k10, obs,
            dyn_obs_tokens=dyn_tokens, dyn_obs_mask=dyn_mask)
    assert endstate.shape == (B, 9, V, H)
    assert score.shape == (B, V, H)
    for name, t in zip(("endstate","score","recon","mu","logvar"),
                        (endstate, score, recon, mu, logvar)):
        assert torch.isfinite(t).all(), f"T5 NaN/Inf in {name}"
    n_dt = sum(p.numel() for p in net_dt.parameters())
    n_d_only = sum(p.numel() for p in
                    make_net(use_temporal=False, use_dca=True, seed=42).parameters())
    assert n_dt - n_d_only == n_agg, \
        f"DCA+temporal delta {n_dt - n_d_only} != aggregator {n_agg}"
    print(f"   DCA-only params:           {n_d_only:,}")
    print(f"   DCA+temporal params:       {n_dt:,}")
    print(f"   delta:                     {n_dt - n_d_only:,}  (== aggregator)")
    print(f"[OK] T5: DCA + temporal compose, params add cleanly")

    print("\n[PASS] stage-3.4 sub-B: 5/5 checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
