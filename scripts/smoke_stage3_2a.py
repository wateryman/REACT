"""Stage-3.2 sub-A smoke: dynamic_attention module + gru_decoder import refactor.

Six checks covering:
  T1 DynObsEncoder output shape (B, M, 7) -> (B, M, hidden)
  T2 DCA no-mask forward shape
  T3 DCA with key_padding_mask: masked output != unmasked output
  T4 GRUDecoder still imports/uses DCA (re-export from gru_decoder.py)
  T5 GRUDecoder with dyn_obs_tokens path forwards correctly
  T6 gradient flows from MSE-style loss back into both Q and K/V branches
     (threshold 1e-9 -- gradients are ~1e-7 with random init + LayerNorm +
      mean reduction; the assertion just proves they are non-zero, not that
      they are large.  Real training amplifies via meaningful targets.)
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
    from policy.models.dynamic_attention import DynObsEncoder, DynamicCrossAttention

    # T1
    enc = DynObsEncoder(7, 128)
    B, M = 4, 8
    obs = torch.randn(B, M, 7)
    out = enc(obs)
    assert out.shape == (B, M, 128), out.shape
    print(f"[OK] T1 DynObsEncoder: {tuple(obs.shape)} -> {tuple(out.shape)}")

    # T2
    dca = DynamicCrossAttention(hidden=128, n_heads=4)
    q = torch.randn(B, 1, 128)
    refined = dca(q, out)
    assert refined.shape == (B, 1, 128), refined.shape
    print(f"[OK] T2 DCA no-mask: q{tuple(q.shape)} kv{tuple(out.shape)} -> {tuple(refined.shape)}")

    # T3
    mask = torch.zeros(B, M, dtype=torch.bool)
    mask[:, 5:] = True
    refined_m = dca(q, out, key_padding_mask=mask)
    diff = (refined_m - refined).abs().max().item()
    assert diff > 1e-4, f"mask had no effect: diff = {diff}"
    print(f"[OK] T3 DCA mask vs no-mask delta = {diff:.4e}")

    # T4
    from policy.models.gru_decoder import GRUDecoder, DynamicCrossAttention as DCA_re
    assert DCA_re is DynamicCrossAttention, "re-export identity broken"
    dec = GRUDecoder(feat_dim=64, hidden=128, gru_layers=2, n_anchors=9, n_waypoints=5)
    wp, sc, T = dec(torch.randn(2, 10, 64))
    assert wp.shape == (2, 9, 5, 9)
    print(f"[OK] T4 GRUDecoder regression: re-exported DCA is same class; "
          f"wp{tuple(wp.shape)} score{tuple(sc.shape)} T{tuple(T.shape)}")

    # T5
    wp2, _, _ = dec(torch.randn(2, 10, 64),
                    dyn_obs_tokens=torch.randn(2, 6, 128))
    assert wp2.shape == wp.shape
    print(f"[OK] T5 GRUDecoder with dyn_obs_tokens path runs")

    # T6 -- gradient flow
    enc2 = DynObsEncoder(7, 128)
    dca2 = DynamicCrossAttention(128, 4)
    obs_in = torch.randn(2, 5, 7, requires_grad=True)
    q_in = torch.randn(2, 3, 128, requires_grad=True)
    tok = enc2(obs_in)
    out = dca2(q_in, tok)
    loss = out.pow(2).mean()
    loss.backward()
    g_obs = obs_in.grad.abs().max().item()
    g_q = q_in.grad.abs().max().item()
    assert g_obs > 1e-9 and g_q > 1e-9, f"grad essentially zero: obs={g_obs}, q={g_q}"
    print(f"[OK] T6 gradient flows: d(loss)/d(obs)={g_obs:.3e}  d(loss)/d(q)={g_q:.3e}")

    print("\n[PASS] stage-3.2 sub-A: 6/6 checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
