"""Stage-1 integration smoke test [🟩 PEMTRS].

Verifies YopoNetwork + reVAE forward shapes, reVAE loss computation, and
that gradients flow back into the reVAE parameters. Does NOT require
flightgym, ROS, ESDF maps, or any training data -- pure tensor ops on a
random batch.

Run from REACT/ root:
    python scripts/smoke_stage1_integration.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
YOPO_DIR = os.path.join(HERE, "..", "YOPO")
os.chdir(YOPO_DIR)
sys.path.insert(0, YOPO_DIR)

import torch

from config.config import cfg
from loss.loss_function import YOPOLoss
from policy.yopo_network import YopoNetwork


device = "cuda" if torch.cuda.is_available() else "cpu"

B = 4
H, W = int(cfg["image_height"]), int(cfg["image_width"])
V, Hh = int(cfg["vertical_num"]), int(cfg["horizon_num"])
latent = int(cfg["revae"]["latent_dim"])

# 1) Instantiate REACT-flavored YopoNetwork
net = YopoNetwork(use_revae=True, revae_latent=latent).to(device)
net.train()

depth = torch.rand(B, 1, H, W, device=device)
obs = torch.randn(B, 9, device=device)

endstate, score, recon, mu, logvar = net.inference(depth, obs)
assert endstate.shape == (B, 9, V, Hh), endstate.shape
assert score.shape == (B, V, Hh), score.shape
assert recon.shape == (B, 1, H, W), recon.shape
assert mu.shape == (B, latent), mu.shape
print(f"[OK] inference  endstate={tuple(endstate.shape)} score={tuple(score.shape)} recon={tuple(recon.shape)}")

# 2) reVAE loss (static -- avoids ESDF load that YOPOLoss() would do)
revae_loss = YOPOLoss.revae_loss(
    recon, depth, mu, logvar,
    lam_recon=float(cfg["revae"]["lam_recon"]),
    lam_kl=float(cfg["revae"]["lam_kl"]),
)
assert torch.isfinite(revae_loss), revae_loss
print(f"[OK] revae_loss = {revae_loss.item():.4f}")

# 3) Backward -- combine reVAE loss with a synthetic surface so we exercise
#    gradient through both the reVAE and the head branches.
total = revae_loss + endstate.abs().mean() + score.mean()
total.backward()
revae_grad = next(net.revae.parameters()).grad
assert revae_grad is not None and torch.isfinite(revae_grad).all(), "reVAE params no grad"
print(f"[OK] backward finite, reVAE grad max-abs = {revae_grad.abs().max().item():.4e}")

# 4) Ablation: use_revae=False path returns None for VAE tensors
net2 = YopoNetwork(use_revae=False).to(device)
endstate2, score2, recon2, mu2, logvar2 = net2.inference(depth, obs)
assert recon2 is None and mu2 is None and logvar2 is None
assert endstate2.shape == endstate.shape
print("[OK] use_revae=False path returns None for VAE tensors")

print()
print("[PASS] stage-1 integration smoke test")
