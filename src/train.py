from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from dataclasses import replace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import DbConfig, SourceConfig, TrainConfig, get_model_timeframes, get_table_for_timeframe
from .data import fetch_series, write_windows_memmap
from .model import build_autoencoder


class WindowMemmapDataset(Dataset):
    def __init__(self, windows_path: str) -> None:
        self.windows = np.load(windows_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, idx: int) -> np.ndarray:
        return np.array(self.windows[idx], dtype=np.float32, copy=True)


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
    timeframe: str,
) -> None:
    payload = {
        "timeframe": timeframe,
        "train_config": asdict(train_cfg),
        "source_config": asdict(source_cfg),
        "windows": [asdict(item) for item in metadata],
    }
    with open(os.path.join(artifact_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _train_one_timeframe(
    db_cfg: DbConfig,
    source_cfg: SourceConfig,
    train_cfg: TrainConfig,
    artifact_dir: str,
    timeframe: str,
) -> None:
    ensure_artifact_dir(artifact_dir)
    rows = fetch_series(db_cfg, source_cfg)
    windows_path = os.path.join(artifact_dir, "windows.npy")
    shape, metadata = write_windows_memmap(rows, train_cfg, windows_path)
    dataset = WindowMemmapDataset(windows_path)
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)

    model = build_autoencoder(
        model_type=train_cfg.model_type,
        window_size=shape[1],
        num_features=shape[2],
        hidden_dim=train_cfg.hidden_dim,
        latent_dim=train_cfg.latent_dim,
        cnn_channels=train_cfg.cnn_channels,
        cnn_kernel_size=train_cfg.cnn_kernel_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(train_cfg.epochs):
        total_loss = 0.0
        batch_count = 0
        for batch in loader:
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1

        avg_loss = total_loss / max(batch_count, 1)
        print(f"model={train_cfg.model_type} epoch={epoch + 1}/{train_cfg.epochs} loss={avg_loss:.6f}")

    model.eval()
    encode_loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=False)
    embedding_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch in encode_loader:
            embedding_batches.append(model.encode(batch).cpu().numpy())
    embeddings = np.concatenate(embedding_batches, axis=0)

    torch.save(model.state_dict(), os.path.join(artifact_dir, "model.pt"))
    np.save(os.path.join(artifact_dir, "embeddings.npy"), embeddings)
    save_metadata(artifact_dir, metadata, train_cfg, source_cfg, timeframe)

    print(f"timeframe={timeframe} saved_artifacts={artifact_dir}")
    print(f"timeframe={timeframe} windows={len(metadata)} shape={shape}")


def train() -> None:
    db_cfg = DbConfig()
    base_source_cfg = SourceConfig()
    train_cfg = TrainConfig()

    set_seed(train_cfg.seed)
    ensure_artifact_dir(train_cfg.artifact_dir)

    for timeframe in get_model_timeframes():
        source_cfg = replace(
            base_source_cfg,
            table=get_table_for_timeframe(
                timeframe,
                source_kind=base_source_cfg.source_kind,
                fallback_table=base_source_cfg.table,
            ),
            timeframe=timeframe,
        )
        artifact_dir = os.path.join(train_cfg.artifact_dir, timeframe)
        print(f"timeframe={timeframe} source_table={source_cfg.table} artifact_dir={artifact_dir}")
        _train_one_timeframe(
            db_cfg=db_cfg,
            source_cfg=source_cfg,
            train_cfg=train_cfg,
            artifact_dir=artifact_dir,
            timeframe=timeframe,
        )


if __name__ == "__main__":
    train()
