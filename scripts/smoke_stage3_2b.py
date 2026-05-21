"""Stage-3.2 sub-B smoke: YopoNetwork DCA side channel wiring.

Six checks:
  T1 use_dca=False           ctor does NOT instantiate DCA modules
                              (param count == stage-3.1 baseline)
  T2 use_dca=False forward    output shapes unchanged
  T3 use_dca=True, no tokens  forward skips DCA, shapes unchanged
  T4 use_dca=True, w/ tokens  forward applies DCA; shapes unchanged
  T5 use_dca=True             tokens DO change the output (DCA actually bites)
  T6 grad flows from output back into dyn_obs_tokens leaf
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
YOPO_DIR = os.path.join(HERE, "..", "YOPO")
os.chdir(YOPO_DIR)
sys.path.insert(0, YOPO_DIR)

import torch

torch.manual_seed(0)


def main():
    from policy.yopo_network import YopoNetwork

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, M = 4, 8
    V, H = 3, 5    # 15 anchors per the YOPO grid (cfg)
    depth = torch.rand(B, 1, 96, 160, device=device)
    obs = torch.randn(B, 9, device=device)

    # T1: use_dca=False has no DCA params
    net_off = YopoNetwork(use_dca=False).to(device)
    net_off.eval()
    n_off = sum(p.numel() for p in net_off.parameters())
    assert net_off.dca is None and net_off.dyn_obs_encoder is None
    print(f"[OK] T1 use_dca=False: dca={net_off.dca}, params={n_off:,}")

    # T2: use_dca=False forward unchanged
    endstate0, score0, recon0, mu0, logvar0 = net_off.inference(depth, obs)
    assert endstate0.shape == (B, 9, V, H), endstate0.shape
    assert score0.shape == (B, V, H)
    assert recon0.shape == (B, 1, 96, 160)
    print(f"[OK] T2 use_dca=False inference: endstate{tuple(endstate0.shape)} "
          f"score{tuple(score0.shape)} recon{tuple(recon0.shape)}")

    # T3: use_dca=True, no tokens -> DCA skipped
    net_on = YopoNetwork(use_dca=True).to(device)
    net_on.eval()
    n_on = sum(p.numel() for p in net_on.parameters())
    assert net_on.dca is not None and net_on.dyn_obs_encoder is not None
    print(f"   use_dca=True params={n_on:,}  (delta = {n_on - n_off:+,})")
    endstate1, score1, recon1, mu1, logvar1 = net_on.inference(depth, obs)
    assert endstate1.shape == (B, 9, V, H)
    print(f"[OK] T3 use_dca=True, tokens=None: same shapes; DCA skipped at None check")

    # T4: use_dca=True with tokens
    tokens = torch.randn(B, M, 7, device=device)
    mask = torch.zeros(B, M, dtype=torch.bool, device=device)
    mask[:, :5] = True   # first 5 slots real
    endstate2, score2, _, _, _ = net_on.inference(depth, obs,
                                                   dyn_obs_tokens=tokens,
                                                   dyn_obs_mask=mask)
    assert endstate2.shape == (B, 9, V, H)
    assert score2.shape == (B, V, H)
    print(f"[OK] T4 use_dca=True, tokens given: endstate{tuple(endstate2.shape)} "
          f"score{tuple(score2.shape)}")

    # T5: DCA actually changes the output vs same network with tokens=None
    diff_e = (endstate2 - endstate1).abs().max().item()
    diff_s = (score2 - score1).abs().max().item()
    print(f"[OK] T5 tokens-on vs tokens-off delta: endstate max-abs = {diff_e:.4e}, "
          f"score max-abs = {diff_s:.4e}")
    assert diff_e > 1e-4 and diff_s > 1e-4, "DCA had no effect"

    # T6: gradient flows back to dyn_obs_tokens
    net_on.train()
    tokens_g = torch.randn(B, M, 7, device=device, requires_grad=True)
    endstate3, score3, _, _, _ = net_on.inference(depth, obs,
                                                   dyn_obs_tokens=tokens_g,
                                                   dyn_obs_mask=mask)
    loss = endstate3.pow(2).mean() + score3.pow(2).mean()
    loss.backward()
    g = tokens_g.grad.abs().max().item()
    assert g > 1e-9, f"grad to dyn_obs_tokens is essentially 0: {g}"
    print(f"[OK] T6 grad to dyn_obs_tokens: max-abs = {g:.3e}")

    print("\n[PASS] stage-3.2 sub-B: 6/6 checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
