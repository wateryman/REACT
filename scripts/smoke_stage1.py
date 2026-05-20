"""Stage-1 PEMTRS-port smoke test (REACT guide §6.1).

Verifies forward shapes for ReVAE / TemporalRegionSelector / GRUDecoder
and the FrameBuffer rolling window. Does NOT touch flightgym -- safe to
run before the Flightmare environment is ready.

Run from REACT/ root:  python scripts/smoke_stage1.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "YOPO"))

from policy.models.gru_decoder import GRUDecoder
from policy.models.revae import ReVAE
from policy.models.temporal_selector import TemporalRegionSelector
from policy.utils.frame_buffer import FrameBuffer


B, K = 2, 10
LATENT = 128
N_FUTURE = 5

# 1) reVAE forward shapes
depth = torch.randn(B * K, 1, 96, 160)
z, recon, mu, logvar = ReVAE(latent_dim=LATENT)(depth)
assert z.shape == (B * K, LATENT), z.shape
assert recon.shape == (B * K, 1, 96, 160), recon.shape
assert mu.shape == logvar.shape == (B * K, LATENT)
print(f"[OK] ReVAE       z={tuple(z.shape)}  recon={tuple(recon.shape)}")

# 2) TemporalRegionSelector future-N ROI
z_seq = z.view(B, K, LATENT)
roi = TemporalRegionSelector(d_model=LATENT, future_horizon=N_FUTURE)(z_seq)
assert roi.shape == (B, N_FUTURE, 4), roi.shape
print(f"[OK] Selector    roi={tuple(roi.shape)}")

# 3) GRUDecoder three-head output (no dyn obs -> cross-attn skipped)
state_dim = 16
roi_dim = N_FUTURE * 4
feat = torch.cat([z_seq, torch.randn(B, K, state_dim + roi_dim)], dim=-1)
dec = GRUDecoder(
    feat_dim=LATENT + state_dim + roi_dim,
    hidden=256, gru_layers=2, n_anchors=9, n_waypoints=5,
)
wp, score, T = dec(feat)
assert wp.shape == (B, 9, 5, 9), wp.shape
assert score.shape == (B, 9), score.shape
assert T.shape == (B, 9, 2), T.shape
print(f"[OK] GRUDecoder  wp={tuple(wp.shape)} score={tuple(score.shape)} T={tuple(T.shape)}")

# 4) GRUDecoder with dyn obs tokens (stage-3 path)
dyn = torch.randn(B, 6, 256)
wp2, _, _ = dec(feat, dyn_obs_tokens=dyn)
assert wp2.shape == wp.shape
print("[OK] GRUDecoder  dyn_obs path runs")

# 5) FrameBuffer warmup + rolling
fb = FrameBuffer(K=5)
fb.push(torch.zeros(1, 96, 160))
window = fb.get()
assert window.shape == (5, 1, 96, 160), window.shape
fb.push(torch.ones(1, 96, 160))
window = fb.get()
assert window.shape == (5, 1, 96, 160)
assert torch.equal(window[-1], torch.ones(1, 96, 160))
print(f"[OK] FrameBuffer warmup+rolling shape={tuple(window.shape)}")

# 6) backward finiteness on a real loss surface (recon + KL)
recon_loss = torch.nn.functional.mse_loss(recon, depth)
kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
total = recon_loss + 1e-3 * kl
total.backward()
assert torch.isfinite(total), f"loss non-finite: {total}"
print(f"[OK] backward    finite (loss={total.item():.4f})")

print()
print("[PASS] stage-1 smoke test")
