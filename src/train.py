from __future__ import annotations

import json
import os
import random
from dataclasses import asdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import DbConfig, SourceConfig, TrainConfig
from .data import build_windows, fetch_series
from .model import WindowAutoencoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_artifact_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_metadata(
    artifact_dir: str,
    metadata: list,
    train_cfg: TrainConfig,
    source_cfg: SourceConfig,
) -> None:
    payload = {
        "train_config": asdict(train_cfg),
        "source_config": asdict(source_cfg),
        "windows": [asdict(item) for item in metadata],
    }
    with open(os.path.join(artifact_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def train() -> None:
    db_cfg = DbConfig()
    source_cfg = SourceConfig()
    train_cfg = TrainConfig()

    set_seed(train_cfg.seed)
    ensure_artifact_dir(train_cfg.artifact_dir)

    rows = fetch_series(db_cfg, source_cfg)
    windows, metadata = build_windows(rows, train_cfg)

    tensor = torch.tensor(windows, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)

    model = WindowAutoencoder(
        window_size=windows.shape[1],
        num_features=windows.shape[2],
        hidden_dim=train_cfg.hidden_dim,
        latent_dim=train_cfg.latent_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(train_cfg.epochs):
        total_loss = 0.0
        batch_count = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1

        avg_loss = total_loss / max(batch_count, 1)
        print(f"epoch={epoch + 1}/{train_cfg.epochs} loss={avg_loss:.6f}")

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(tensor).cpu().numpy()

    torch.save(model.state_dict(), os.path.join(train_cfg.artifact_dir, "model.pt"))
    np.save(os.path.join(train_cfg.artifact_dir, "windows.npy"), windows)
    np.save(os.path.join(train_cfg.artifact_dir, "embeddings.npy"), embeddings)
    save_metadata(train_cfg.artifact_dir, metadata, train_cfg, source_cfg)

    print(f"saved artifacts to {train_cfg.artifact_dir}")
    print(f"windows={len(metadata)} shape={windows.shape}")


if __name__ == "__main__":
    train()
