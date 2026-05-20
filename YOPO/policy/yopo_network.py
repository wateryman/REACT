"""
YOPO Network
forward, prediction, pre-processing, post-processing

REACT stage-1 [🟩 PEMTRS]: a reVAE auxiliary encoder is concatenated into
the depth feature map before the YopoHead. The anchor grid (V x H) and
the 1x1 conv head are unchanged, so the per-anchor loss flow stays
compatible with the original YOPO trainer.
"""

import torch
from torch import nn
import numpy as np
from policy.models.backbone import YopoBackbone
from policy.models.head import YopoHead
from policy.models.revae import ReVAE
from policy.state_transform import *


class YopoNetwork(nn.Module):

    def __init__(
            self,
            observation_dim=9,  # 9: v_xyz, a_xyz, goal_xyz
            output_dim=10,  # 10: x_pva, y_pva, z_pva, score
            hidden_state=64,
            use_revae: bool = True,    # 🟩
            revae_latent: int = 128,   # 🟩
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

    def forward(self, depth: torch.Tensor, obs: torch.Tensor):
        """forward propagation.

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

        output = self.yopo_head(input_tensor)
        endstate = torch.tanh(output[:, :9])              # [batch, 9, vertical_num, horizon_num]
        score = torch.nn.functional.softplus(output[:, 9])  # [batch, vertical_num, horizon_num]
        return endstate, score, recon, mu, logvar

    def inference(self, depth: torch.Tensor, obs: torch.Tensor):
        """
            For network training:
            (1) normalize the input state and transform to primitive frame
            (2) forward propagation
            (3) convert the prediction to endstate in body frame.
            obs: current state in the body frame.
            return: (endstate_b, score_pred, recon, mu, logvar)
        """
        obs = self.state_transform.normalize_obs(obs)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, score_pred, recon, mu, logvar = self.forward(depth, obs)
        endstate = self.state_transform.pred_to_endstate(endstate_pred)
        return endstate, score_pred, recon, mu, logvar

    def print_grad(self, grad):
        print("grad of hook: ", grad)
