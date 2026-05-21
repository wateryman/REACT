"""Temporal aggregator [🟦 stage-3.4 path-a].

A thin wrapper around `nn.GRU` that consumes a sequence of reVAE latents
(B, K, latent_dim) and emits a single hidden state (B, hidden) used as
the "z_temporal" feature in YopoNetwork.forward.  Replaces stage-1's
single-frame `z = revae(depth_last)` with `z_temporal = gru_last_hidden`.

Design notes (see docs/stage_3_4_design_cn.md §3, §5 Q1):
- One layer is enough at K=10; stack only if the 5k A/B asks for it.
- On static-batch fallback, the trainer expands the single depth frame
  to K identical copies; GRU sees constant input and converges to a
  fixed-point hidden state.  Gradient flow is equivalent to a 1-step
  GRU on that single frame (cf. arXiv 2002.00025).
- `hidden` defaults to match the reVAE latent_dim so `head_in` stays
  bit-identical to stage-1's 64 + 9 + 128 = 201.
"""
import torch
from torch import nn


class TemporalAggregator(nn.Module):

    def __init__(
            self,
            latent_dim: int = 128,
            hidden: int = 128,
            num_layers: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """Reduce a K-step latent sequence to a single embedding.

        Parameters
        ----------
        z_seq : (B, K, latent_dim) float
            reVAE latents for K consecutive frames.

        Returns
        -------
        h_last : (B, hidden) float
            Last-step hidden state of the GRU.  Stage-3.4 broadcasts this
            to V x H before YopoHead, exactly where stage-1 placed the
            single-frame z.
        """
        # nn.GRU returns (output, h_n); output[:, -1] equals h_n[-1] when
        # num_layers == 1.  We use output[:, -1] for clarity at any depth.
        out, _ = self.gru(z_seq)
        return out[:, -1, :]
