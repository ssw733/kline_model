from __future__ import annotations

import torch
from torch import nn


class BaseWindowAutoencoder(nn.Module):
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MLPWindowAutoencoder(BaseWindowAutoencoder):
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


class CNNWindowAutoencoder(BaseWindowAutoencoder):
    def __init__(
        self,
        window_size: int,
        num_features: int,
        hidden_dim: int,
        latent_dim: int,
        cnn_channels: int,
        cnn_kernel_size: int,
    ) -> None:
        super().__init__()
        padding = cnn_kernel_size // 2
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(num_features, cnn_channels, kernel_size=cnn_kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=cnn_kernel_size, padding=padding),
            nn.ReLU(),
        )
        self.encoder_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_channels * window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, window_size * num_features),
        )
        self.window_size = window_size
        self.num_features = num_features

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.conv_encoder(x.transpose(1, 2))
        return self.encoder_head(encoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        decoded = self.decoder(latent)
        return decoded.view(x.size(0), self.window_size, self.num_features)


def build_autoencoder(
    model_type: str,
    window_size: int,
    num_features: int,
    hidden_dim: int,
    latent_dim: int,
    cnn_channels: int,
    cnn_kernel_size: int,
) -> BaseWindowAutoencoder:
    if model_type == "mlp":
        return MLPWindowAutoencoder(
            window_size=window_size,
            num_features=num_features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )
    if model_type == "cnn":
        return CNNWindowAutoencoder(
            window_size=window_size,
            num_features=num_features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            cnn_channels=cnn_channels,
            cnn_kernel_size=cnn_kernel_size,
        )
    raise ValueError(f"Unsupported MODEL_TYPE: {model_type!r}. Expected one of: mlp, cnn.")
