"""REACT stage-3.2 [🟧] dynamic-obstacle side-channel modules.

Two small nn.Modules used to inject "where are the moving obstacles" info
into the planner's forward path without changing the single-frame
anchor-grid output shape:

  - DynObsEncoder: per-obstacle MLP (7 -> hidden_dim).  Input is the
    7-tuple [px, py, pz, vx, vy, vz, radius] for one obstacle in a
    drone-relative frame (the trainer subtracts drone world position
    before packing).
  - DynamicCrossAttention: thin wrapper around nn.MultiheadAttention with
    a residual + LayerNorm; supports the standard `key_padding_mask`
    so padded obstacle slots don't contaminate the attention output.

Both live here (rather than in gru_decoder.py where DCA was originally
sketched in stage-1) so stage-3.2 path (b) -- single-frame forward with
the side channel -- can use them without dragging in the (still unwired)
GRU decoder.  gru_decoder.py imports DynamicCrossAttention from this
module so the stage-1 module on disk continues to construct identically.
"""
import torch
import torch.nn as nn


class DynObsEncoder(nn.Module):
    """Per-obstacle MLP.  Maps each obstacle row (7 dims) to a token of
    `hidden_dim`.  Two hidden layers with ReLU + a final LayerNorm; the
    output is then ready to be used as Keys/Values in a cross-attention.

    Input  : (B, M, 7)  -- packed [px, py, pz, vx, vy, vz, radius]
                          (already in drone-relative frame; the trainer
                          subtracts the drone world position before feeding)
    Output : (B, M, hidden_dim)
    """
    def __init__(self, in_dim: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DynamicCrossAttention(nn.Module):
    """Multi-head cross-attention with residual + LayerNorm.

    Compared to the stage-1 stub in gru_decoder.py this version supports
    `key_padding_mask` (True positions are ignored), which stage-3.2 needs
    because the dyn_obs payload is right-padded to M_max slots.
    """
    def __init__(self, hidden: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        q : (B, N_q, hidden)
        kv: (B, M, hidden)
        key_padding_mask: (B, M) bool, True == ignore this slot
        """
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        return self.norm(q + out)
