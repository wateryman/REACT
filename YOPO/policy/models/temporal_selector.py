"""Temporal Region Selector [🟩 PEMTRS Sec. III-B + 🟧 future_horizon upgrade].

Input:  z_seq (B, K, d_model) -- encoded latents over the K-frame window.
Output: roi   (B, N, 4)       -- N-step ROI sequence; each row encodes
                                 [phi_h, phi_v, dphi_h, dphi_v].

future_horizon=1 reproduces the original PEMTRS behaviour (current-frame
ROI only). future_horizon>=2 is the REACT 🟧 upgrade that predicts ROIs
for the next N timesteps in a single shot, used by the decoder to attend
to where dynamic obstacles will be -- not just where they are now.
"""
import math

import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class TemporalRegionSelector(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        future_horizon: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.future_horizon = future_horizon
        self.pe = SinusoidalPE(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.roi_head = nn.Linear(d_model, future_horizon * 4)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        # z_seq: (B, K, d_model)
        x = z_seq + self.pe(z_seq)
        h = self.transformer(x)
        pooled = h.mean(dim=1)
        roi = self.roi_head(pooled)
        return roi.view(-1, self.future_horizon, 4)
