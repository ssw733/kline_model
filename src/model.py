from __future__ import annotations

import torch
from torch import nn


class WindowAutoencoder(nn.Module):
    def __init__(self, window_size: int, num_features: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        input_dim = window_size * num_features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.window_size = window_size
        self.num_features = num_features

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1)
        return self.encoder(flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        decoded = self.decoder(latent)
        return decoded.view(x.size(0), self.window_size, self.num_features)
