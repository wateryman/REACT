"""Residual VAE (reVAE) for depth-image encoding [🟩 PEMTRS Sec. III-A].

Encoder: 4 residual conv blocks (1 -> 16 -> 32 -> 64 -> 128 channels) with
stride-2 downsampling, then adaptive average pool to (128, 1, 1) followed by
fc_mu / fc_logvar projecting to latent_dim (default 128).

Decoder: linear back to (128, 6, 10), then 4 nearest-neighbour upsample +
residual conv blocks restoring (1, 96, 160).

Sobel-edge reconstruction loss lives in losses/sobel_loss.py (added stage 3).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.skip(x), inplace=True)


class UpResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.skip(x), inplace=True)


class ReVAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder: 4 downsampling stages take 96x160 -> 6x10
        self.encoder = nn.Sequential(
            ResBlock(1, 16),    # 48x80
            ResBlock(16, 32),   # 24x40
            ResBlock(32, 64),   # 12x20
            ResBlock(64, 128),  # 6x10
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        # Decoder: project latent back to (128, 6, 10), 4x upsample blocks
        self._dec_hw = (6, 10)
        self.fc_dec = nn.Linear(latent_dim, 128 * self._dec_hw[0] * self._dec_hw[1])
        self.decoder = nn.Sequential(
            UpResBlock(128, 64),  # 12x20
            UpResBlock(64, 32),   # 24x40
            UpResBlock(32, 16),   # 48x80
            UpResBlock(16, 8),    # 96x160
        )
        self.head = nn.Conv2d(8, 1, kernel_size=1)

    def encode(self, depth: torch.Tensor):
        h = self.encoder(depth)
        h = self.pool(h).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, 128, *self._dec_hw)
        h = self.decoder(h)
        return self.head(h)

    def forward(self, depth: torch.Tensor):
        mu, logvar = self.encode(depth)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return z, recon, mu, logvar
