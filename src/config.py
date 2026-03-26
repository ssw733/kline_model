from __future__ import annotations

import os
from dataclasses import dataclass


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True)
class DbConfig:
    host: str = os.getenv("PGHOST", "localhost")
    port: int = _get_env_int("PGPORT", 5432)
    dbname: str = os.getenv("PGDATABASE", "coins")
    user: str = os.getenv("PGUSER", "admin")
    password: str = os.getenv("PGPASSWORD", "admin")


@dataclass(frozen=True)
class SourceConfig:
    source_kind: str = os.getenv("COINS_SOURCE_KIND", "coingecko")
    table: str = os.getenv("COINS_TABLE", "coingecko_market_data")
    entity_col: str = os.getenv("COINS_ENTITY_COL", "token")
    time_col: str = os.getenv("COINS_TIME_COL", "timestamp")
    price_col: str = os.getenv("COINS_PRICE_COL", "price")
    aux_col: str = os.getenv("COINS_AUX_COL", "holders")
    custom_sql: str | None = os.getenv("COINS_SOURCE_SQL")


@dataclass(frozen=True)
class TrainConfig:
    window_size: int = _get_env_int("WINDOW_SIZE", 64)
    min_points_per_entity: int = _get_env_int("MIN_POINTS_PER_ENTITY", 96)
    batch_size: int = _get_env_int("BATCH_SIZE", 256)
    epochs: int = _get_env_int("EPOCHS", 20)
    learning_rate: float = _get_env_float("LEARNING_RATE", 1e-3)
    latent_dim: int = _get_env_int("LATENT_DIM", 32)
    hidden_dim: int = _get_env_int("HIDDEN_DIM", 128)
    artifact_dir: str = os.getenv("ARTIFACT_DIR", "artifacts")
    seed: int = _get_env_int("SEED", 42)
