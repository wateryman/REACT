"""
YOPO Network
forward, prediction, pre-processing, post-processing

REACT stage-1 [🟩 PEMTRS]: a reVAE auxiliary encoder is concatenated into
the depth feature map before the YopoHead. The anchor grid (V x H) and
the 1x1 conv head are unchanged, so the per-anchor loss flow stays
compatible with the original YOPO trainer.

REACT stage-3.2 [🟧 path b]: optional DynamicCrossAttention side channel.
When `use_dca=True` and `dyn_obs_tokens` is passed at forward time, the
(B, head_in, V, H) feature map is reshaped to (B, V*H, head_in) and
cross-attended against per-obstacle tokens before going into the head.
Defaults preserve stage-1/stage-3.1 byte-clean behaviour.
"""

import torch
from torch import nn
import numpy as np
from policy.models.backbone import YopoBackbone
from policy.models.head import YopoHead
from policy.models.revae import ReVAE
from policy.models.dynamic_attention import DynObsEncoder, DynamicCrossAttention
from policy.state_transform import *


class YopoNetwork(nn.Module):

    def __init__(
            self,
            observation_dim=9,  # 9: v_xyz, a_xyz, goal_xyz
            output_dim=10,  # 10: x_pva, y_pva, z_pva, score
            hidden_state=64,
            use_revae: bool = True,    # 🟩
            revae_latent: int = 128,   # 🟩
            use_dca: bool = False,     # 🟧 stage-3.2
            dca_n_heads: int = 1,      # 🟧 stage-3.2: head_in (201) is prime-ish;
                                       #             1 head avoids divisibility issues
    ):
        super(YopoNetwork, self).__init__()
        self.state_transform = StateTransform()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_backbone = YopoBackbone(hidden_state)
        self.state_backbone = nn.Sequential()

        self.use_revae = use_revae
        self.revae_latent = revae_latent
        if use_revae:
            self.revae = ReVAE(latent_dim=revae_latent)
            head_in = hidden_state + observation_dim + revae_latent
        else:
            self.revae = None
            head_in = hidden_state + observation_dim
        self.yopo_head = YopoHead(head_in, output_dim)

        # 🟧 stage-3.2 sub-B: side-channel modules.  Only instantiated when
        # use_dca=True so stage-1/stage-3.1 networks keep their exact parameter
        # count and random-init footprint.
        self.use_dca = use_dca
        self.head_in = head_in
        if use_dca:
            self.dyn_obs_encoder = DynObsEncoder(in_dim=7, hidden_dim=head_in)
            self.dca = DynamicCrossAttention(hidden=head_in, n_heads=dca_n_heads)
        else:
            self.dyn_obs_encoder = None
            self.dca = None

    def forward(self, depth: torch.Tensor, obs: torch.Tensor,
                dyn_obs_tokens: torch.Tensor = None,
                dyn_obs_mask: torch.Tensor = None):
        """forward propagation.

        Parameters
        ----------
        depth : (B, 1, H_in, W_in)
        obs   : (B, observation_dim, V, H) -- already broadcast to anchor grid
                by state_transform.prepare_input
        dyn_obs_tokens : (B, M, 7) drone-relative obstacle rows
                         [px, py, pz, vx, vy, vz, radius], or None to skip DCA.
                         Only consumed when use_dca=True; when use_dca=False the
                         tokens are silently ignored.
        dyn_obs_mask   : (B, M) bool, True for real obstacle slots.  When None
                         and tokens is given, all M slots are treated as real.

        Returns
        -------
        endstate : (B, 9, V, H)
        score    : (B, V, H)
        recon    : (B, 1, H_in, W_in) or None
        mu       : (B, revae_latent) or None
        logvar   : (B, revae_latent) or None
        """
        depth_feature = self.image_backbone(depth)
        obs_feature = self.state_backbone(obs)

        if self.use_revae:
            z, recon, mu, logvar = self.revae(depth)
            Vh, Wh = depth_feature.shape[-2], depth_feature.shape[-1]
            z_spatial = z[:, :, None, None].expand(-1, -1, Vh, Wh)
            input_tensor = torch.cat((obs_feature, depth_feature, z_spatial), dim=1)
        else:
            recon, mu, logvar = None, None, None
            input_tensor = torch.cat((obs_feature, depth_feature), dim=1)

        # 🟧 stage-3.2 sub-B: DCA refinement.  When dyn_obs_tokens is None (or
        # use_dca is False, which leaves self.dca as None), this block is
        # skipped entirely -> output is bit-identical to the stage-3.1 path
        # in that mode, which preserves all upstream regression assertions.
        if self.use_dca and dyn_obs_tokens is not None:
            B, C, Vh, Wh = input_tensor.shape
            # (B, C, V, H) -> (B, V*H, C) anchor-tokens to act as Q
            feat_tokens = input_tensor.permute(0, 2, 3, 1).reshape(B, Vh * Wh, C)
            # Encode the per-obstacle 7-tuples into the same channel size (C)
            kv = self.dyn_obs_encoder(dyn_obs_tokens)            # (B, M, C)
            # PyTorch MHA expects key_padding_mask True == ignore; our mask
            # is True == real obstacle, so invert.
            kpm = None
            if dyn_obs_mask is not None:
                kpm = ~dyn_obs_mask.to(dtype=torch.bool)
            refined = self.dca(feat_tokens, kv, key_padding_mask=kpm)  # (B, V*H, C)
            # (B, V*H, C) -> (B, C, V, H) for the conv head
            input_tensor = refined.reshape(B, Vh, Wh, C).permute(0, 3, 1, 2).contiguous()

        output = self.yopo_head(input_tensor)
        endstate = torch.tanh(output[:, :9])              # [batch, 9, vertical_num, horizon_num]
        score = torch.nn.functional.softplus(output[:, 9])  # [batch, vertical_num, horizon_num]
        return endstate, score, recon, mu, logvar

    def inference(self, depth: torch.Tensor, obs: torch.Tensor,
                  dyn_obs_tokens: torch.Tensor = None,
                  dyn_obs_mask: torch.Tensor = None):
        """
            For network training:
            (1) normalize the input state and transform to primitive frame
            (2) forward propagation (optionally with DCA side channel)
            (3) convert the prediction to endstate in body frame.
            obs: current state in the body frame.
            return: (endstate_b, score_pred, recon, mu, logvar)
        """
        obs = self.state_transform.normalize_obs(obs)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, score_pred, recon, mu, logvar = self.forward(
            depth, obs, dyn_obs_tokens=dyn_obs_tokens, dyn_obs_mask=dyn_obs_mask)
        endstate = self.state_transform.pred_to_endstate(endstate_pred)
        return endstate, score_pred, recon, mu, logvar

    def print_grad(self, grad):
        print("grad of hook: ", grad)
