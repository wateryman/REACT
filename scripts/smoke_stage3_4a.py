"""Stage-3.4 sub-A smoke: TemporalAggregator unit + reVAE batched encode.

Four checks covering phase-1 of stage-3.4 path-a:
  T1 reVAE batched encode round-trip:
       (B*K, 1, H, W)  --reVAE.encode-->  (B*K, latent)
       --view-->  (B, K, latent)
       No NaN, finite, deterministic.
  T2 TemporalAggregator shape:
       (B, K=1, latent)   ->  (B, hidden)
       (B, K=10, latent)  ->  (B, hidden)
       Output finite, no NaN.
  T3 Static-batch fallback semantics:
       z_single (B, 1, latent)  --expand-->  (B, K, latent) constant across K
       GRU output is bounded (no exploding hidden) and approximately
       repeats if we feed K=20 instead of K=10 (fixed-point convergence).
  T4 Gradient flow end-to-end:
       depth (B*K, 1, H, W) requires_grad
       -> revae.encode -> reshape -> gru -> sum().backward()
       Both reVAE encoder params and GRU params receive non-zero grad.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
YOPO_DIR = os.path.join(HERE, "..", "YOPO")
os.chdir(YOPO_DIR)
sys.path.insert(0, YOPO_DIR)

import torch

torch.manual_seed(42)


def main():
    from policy.models.revae import ReVAE
    from policy.models.temporal_aggregator import TemporalAggregator

    B, K, H, W = 4, 10, 96, 160
    latent_dim = 128
    hidden = 128

    revae = ReVAE(latent_dim=latent_dim)
    revae.eval()  # disable reparam stochasticity so T1 is deterministic
    aggregator = TemporalAggregator(latent_dim=latent_dim, hidden=hidden)

    # ---------- T1: reVAE batched encode round-trip ----------
    print("== T1: reVAE batched encode round-trip ==")
    depth = torch.rand(B, K, 1, H, W)                       # in [0, 1]
    depth_flat = depth.view(B * K, 1, H, W)
    with torch.no_grad():
        mu, logvar = revae.encode(depth_flat)
    assert mu.shape == (B * K, latent_dim), mu.shape
    assert torch.isfinite(mu).all() and torch.isfinite(logvar).all()
    z_flat = mu                                              # use mu in eval mode
    z_seq = z_flat.view(B, K, latent_dim)
    assert z_seq.shape == (B, K, latent_dim)
    # Determinism: second call must match first call exactly.
    with torch.no_grad():
        mu2, _ = revae.encode(depth_flat)
    assert torch.allclose(mu, mu2), "reVAE encode not deterministic in eval mode"
    print(f"   depth {tuple(depth.shape)} -> z_seq {tuple(z_seq.shape)}, "
          f"|mu| mean = {mu.abs().mean():.4f}")
    print(f"[OK] T1 reVAE batched encode passes")

    # ---------- T2: aggregator shape ----------
    print("\n== T2: TemporalAggregator shape ==")
    for k in (1, 10):
        z = torch.randn(B, k, latent_dim)
        with torch.no_grad():
            h = aggregator(z)
        assert h.shape == (B, hidden), (k, h.shape)
        assert torch.isfinite(h).all()
        print(f"   K={k:2d}: z{tuple(z.shape)} -> h{tuple(h.shape)} "
              f"|h| mean = {h.abs().mean():.4f}")
    print(f"[OK] T2 aggregator shape passes")

    # ---------- T3: static-fallback fixed-point ----------
    print("\n== T3: static-fallback fixed-point convergence ==")
    z_single = torch.randn(B, 1, latent_dim)
    z_K10 = z_single.expand(-1, 10, -1)                     # stride-0
    z_K20 = z_single.expand(-1, 20, -1)
    with torch.no_grad():
        h_K10 = aggregator(z_K10)
        h_K20 = aggregator(z_K20)
    # Fixed-point: as K grows, h_K converges; the K=10 vs K=20 delta should
    # be small relative to |h|.  We just require BOUNDED output (no
    # divergence) and that delta isn't huge.
    delta = (h_K20 - h_K10).abs().mean().item()
    h_mag = h_K20.abs().mean().item()
    rel = delta / max(h_mag, 1e-8)
    print(f"   |h_K10| mean = {h_K10.abs().mean():.4f}")
    print(f"   |h_K20 - h_K10| / |h_K20| = {rel:.4f}  (small => fixed-point reached)")
    assert torch.isfinite(h_K10).all() and torch.isfinite(h_K20).all()
    assert h_mag < 100.0, f"GRU output exploded under constant input: |h|={h_mag}"
    assert rel < 0.5, f"Fixed-point not reached: delta/|h| = {rel}"
    print(f"[OK] T3 constant input is bounded + fixed-point converges")

    # ---------- T4: gradient flow ----------
    print("\n== T4: gradient flow reVAE -> GRU ==")
    revae.train()                                            # re-enable reparam
    aggregator.train()
    revae.zero_grad()
    aggregator.zero_grad()
    depth_g = torch.rand(B, K, 1, H, W, requires_grad=True)
    depth_g_flat = depth_g.view(B * K, 1, H, W)
    mu_g, logvar_g = revae.encode(depth_g_flat)
    # Use mu (deterministic forward) so gradient is unambiguous
    z_g = mu_g.view(B, K, latent_dim)
    h_g = aggregator(z_g)
    loss = h_g.pow(2).mean()
    loss.backward()

    # Both modules should have non-zero grads on at least one param.
    enc_max = max(p.grad.abs().max().item() for p in revae.encoder.parameters()
                   if p.grad is not None)
    fcmu_max = revae.fc_mu.weight.grad.abs().max().item()
    gru_max = max(p.grad.abs().max().item() for p in aggregator.parameters()
                   if p.grad is not None)
    print(f"   revae.encoder grad max-abs = {enc_max:.3e}")
    print(f"   revae.fc_mu   grad max-abs = {fcmu_max:.3e}")
    print(f"   aggregator    grad max-abs = {gru_max:.3e}")
    assert enc_max > 0.0, "reVAE encoder got no gradient"
    assert fcmu_max > 0.0, "reVAE fc_mu got no gradient"
    assert gru_max > 0.0, "aggregator GRU got no gradient"
    print(f"[OK] T4 gradient flows through reVAE.encoder + fc_mu + aggregator.gru")

    # Param count report
    n_revae = sum(p.numel() for p in revae.parameters())
    n_aggr = sum(p.numel() for p in aggregator.parameters())
    print(f"\n   reVAE  params: {n_revae:,}")
    print(f"   aggr   params: {n_aggr:,}  (~99K expected for GRU(128,128,1))")
    assert 80_000 < n_aggr < 120_000, f"aggregator param count surprising: {n_aggr}"

    print("\n[PASS] stage-3.4 sub-A: 4/4 checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
