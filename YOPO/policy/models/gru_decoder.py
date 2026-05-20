"""Trajectory decoder [🟩 PEMTRS Sec. III-C + 🟧 dynamic cross-attention upgrade].

Pipeline:
  fused_seq (B, K, feat_dim)
    -> GRU                                                          🟩
    -> take last hidden state (B, 1, hidden)
    -> optional cross-attention to dynamic-obstacle tokens          🟧
    -> multi-head self-attention                                    🟩
    -> three heads:
         waypoint_head -> (B, n_anchors, n_waypoints, 9)            🟩 pos+vel+acc
         score_head    -> (B, n_anchors)                            🟩 softplus
         time_head     -> (B, n_anchors, 2)  [T_min, T_max]         🟩 softplus

The dyn_cross_attn module is wired in stage 1 but bypassed at runtime
when dyn_obs_tokens=None; stage 3 will feed it real obstacle tokens
encoded from info["dyn_obs"].
"""
import torch
import torch.nn as nn


class DynamicCrossAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: (B, 1, hidden); kv: (B, M, hidden)
        out, _ = self.attn(q, kv, kv)
        return self.norm(q + out)


class GRUDecoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden: int = 256,
        gru_layers: int = 2,
        n_anchors: int = 9,
        n_waypoints: int = 5,
        n_heads: int = 4,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.n_anchors = n_anchors
        self.n_waypoints = n_waypoints

        self.gru = nn.GRU(feat_dim, hidden, num_layers=gru_layers, batch_first=True)
        self.dyn_cross_attn = DynamicCrossAttention(hidden, n_heads=n_heads)
        self.self_attn = nn.MultiheadAttention(hidden, num_heads=n_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden)

        self.waypoint_head = nn.Linear(hidden, n_anchors * n_waypoints * 9)
        self.score_head = nn.Sequential(nn.Linear(hidden, n_anchors), nn.Softplus())
        self.time_head = nn.Sequential(nn.Linear(hidden, n_anchors * 2), nn.Softplus())

    def forward(self, fused_seq: torch.Tensor, dyn_obs_tokens: torch.Tensor = None, h0=None):
        h_seq, _ = self.gru(fused_seq, h0)
        h_last = h_seq[:, -1:, :]
        if dyn_obs_tokens is not None:
            h_last = self.dyn_cross_attn(h_last, dyn_obs_tokens)
        sa_out, _ = self.self_attn(h_last, h_last, h_last)
        h = self.self_norm(h_last + sa_out).squeeze(1)

        waypoints = self.waypoint_head(h).view(-1, self.n_anchors, self.n_waypoints, 9)
        score = self.score_head(h)
        T = self.time_head(h).view(-1, self.n_anchors, 2)
        return waypoints, score, T
