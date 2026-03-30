from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os

import numpy as np
from psycopg import connect

from .config import DataConfig, DbConfig, SourceConfig, TrainConfig


@dataclass
class WindowMetadata:
    entity_id: str
    start_ts: float
    end_ts: float
    start_idx: int


def build_dsn(db: DbConfig) -> str:
    return (
        f"host={db.host} port={db.port} dbname={db.dbname} "
        f"user={db.user} password={db.password}"
    )


def fetch_series(db: DbConfig, source: SourceConfig) -> list[dict]:
    data_cfg = DataConfig()
    if source.custom_sql:
        sql = source.custom_sql
    elif source.source_kind == "coingecko":
        sql = f"""
            SELECT
                coin_id::text AS entity_id,
                EXTRACT(EPOCH FROM ts)::double precision AS ts,
                price::double precision AS price,
                COALESCE(total_volume::double precision, {data_cfg.default_volume}) AS volume,
                COALESCE(market_cap::double precision, {data_cfg.default_market_cap}) AS market_cap
            FROM {source.table}
            WHERE price IS NOT NULL
            ORDER BY coin_id, ts
        """
    elif source.source_kind == "binance":
        sql = f"""
            SELECT
                COALESCE(coingecko_id, base_asset)::text AS entity_id,
                base_asset::text AS base_asset,
                EXTRACT(EPOCH FROM open_time)::double precision AS ts,
                open::double precision AS open,
                high::double precision AS high,
                low::double precision AS low,
                close::double precision AS close,
                close::double precision AS price,
                COALESCE(volume::double precision, {data_cfg.default_volume}) AS volume,
                COALESCE(quote_asset_volume::double precision, {data_cfg.default_quote_volume}) AS quote_volume
            FROM {source.table}
            WHERE close IS NOT NULL
            ORDER BY entity_id, open_time
        """
    else:
        sql = f"""
            SELECT
                {source.entity_col}::text AS entity_id,
                EXTRACT(EPOCH FROM {source.time_col})::double precision AS ts,
                {source.price_col}::double precision AS price,
                COALESCE({source.aux_col}::double precision, {data_cfg.default_volume}) AS aux_value
            FROM {source.table}
            WHERE {source.price_col} IS NOT NULL
            ORDER BY {source.entity_col}, {source.time_col}
        """

    with connect(build_dsn(db)) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            columns = [desc.name for desc in cur.description]
            rows = cur.fetchall()

    data = [dict(zip(columns, row)) for row in rows]
    if not data:
        raise ValueError("No rows loaded from PostgreSQL. Check table name or SQL query.")

    required = {"entity_id", "ts", "price"}
    missing = required - set(data[0].keys())
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    for row in data:
        if isinstance(row.get("ts"), datetime):
            row["ts"] = row["ts"].timestamp()

    if "volume" not in data[0]:
        for row in data:
            row["volume"] = float(row.get("aux_value", data_cfg.default_volume) or data_cfg.default_volume)
    if "market_cap" not in data[0]:
        for row in data:
            row["market_cap"] = data_cfg.default_market_cap
    if "quote_volume" not in data[0]:
        for row in data:
            row["quote_volume"] = data_cfg.default_quote_volume
    for key in ("open", "high", "low", "close"):
        if key not in data[0]:
            for row in data:
                row[key] = float(row["price"])

    return data


def _window_zscore(values: np.ndarray) -> np.ndarray:
    data_cfg = DataConfig()
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / (std + data_cfg.epsilon_std)


def _safe_log_return(prices: np.ndarray) -> np.ndarray:
    data_cfg = DataConfig()
    clipped = np.clip(prices, data_cfg.epsilon_price, None)
    returns = np.diff(np.log(clipped), prepend=np.log(clipped[0]))
    returns[0] = data_cfg.zero_fill_value
    return returns


def _relative_price(prices: np.ndarray) -> np.ndarray:
    data_cfg = DataConfig()
    base = max(prices[0], data_cfg.epsilon_price)
    return prices / base - 1.0


def _growth_rate(values: np.ndarray) -> np.ndarray:
    data_cfg = DataConfig()
    prev = np.clip(np.roll(values, 1), data_cfg.epsilon_std, None)
    prev[0] = max(values[0], data_cfg.epsilon_std)
    growth = values / prev - 1.0
    growth[0] = data_cfg.zero_fill_value
    return growth


def group_by_entity(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        entity_id = str(row["entity_id"])
        grouped.setdefault(entity_id, []).append(row)
    return grouped


def _build_feature_matrix(
    entity_rows: list[dict],
    data_cfg: DataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    prices = np.array([float(row["price"]) for row in entity_rows], dtype=np.float32)
    volume = np.array([float(row.get("volume", data_cfg.default_volume)) for row in entity_rows], dtype=np.float32)
    market_cap = np.array(
        [float(row.get("market_cap", data_cfg.default_market_cap)) for row in entity_rows],
        dtype=np.float32,
    )
    quote_volume = np.array(
        [float(row.get("quote_volume", data_cfg.default_quote_volume)) for row in entity_rows],
        dtype=np.float32,
    )
    ts = np.array([float(row["ts"]) for row in entity_rows], dtype=np.float64)

    rel_price = _relative_price(prices)
    log_returns = _safe_log_return(prices)
    volatility = np.sqrt(
        np.convolve(
            log_returns ** 2,
            np.ones(data_cfg.volatility_window, dtype=np.float32) / data_cfg.volatility_window,
            mode="same",
        )
    )
    volume_log = np.log1p(np.clip(volume, data_cfg.zero_fill_value, None))
    volume_delta = np.diff(volume_log, prepend=volume_log[0])
    volume_rate = _growth_rate(volume_log)
    market_cap_log = np.log1p(np.clip(market_cap, data_cfg.zero_fill_value, None))
    quote_volume_log = np.log1p(np.clip(quote_volume, data_cfg.zero_fill_value, None))

    features = np.column_stack(
        [
            rel_price,
            log_returns,
            volatility,
            volume_delta,
            volume_rate,
            market_cap_log,
            quote_volume_log,
        ]
    ).astype(np.float32)
    return features, ts


def _iter_entity_windows(
    grouped: dict[str, list[dict]],
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
):
    for entity_id, entity_rows in grouped.items():
        if len(entity_rows) < max(train_cfg.min_points_per_entity, train_cfg.window_size):
            continue

        features, ts = _build_feature_matrix(entity_rows, data_cfg)
        window_size = train_cfg.window_size
        for start_idx in range(0, len(entity_rows) - window_size + 1):
            end_idx = start_idx + window_size
            chunk = features[start_idx:end_idx]
            normalized = _window_zscore(chunk)
            normalized = np.nan_to_num(
                normalized,
                nan=data_cfg.zero_fill_value,
                posinf=data_cfg.zero_fill_value,
                neginf=data_cfg.zero_fill_value,
            ).astype(np.float32)
            yield normalized, WindowMetadata(
                entity_id=entity_id,
                start_ts=float(ts[start_idx]),
                end_ts=float(ts[end_idx - 1]),
                start_idx=start_idx,
            )


def write_windows_memmap(
    rows: list[dict],
    train_cfg: TrainConfig,
    output_path: str,
) -> tuple[tuple[int, int, int], list[WindowMetadata]]:
    data_cfg = DataConfig()
    grouped = group_by_entity(rows)
    total_windows = 0
    num_features: int | None = None

    for entity_rows in grouped.values():
        if len(entity_rows) < max(train_cfg.min_points_per_entity, train_cfg.window_size):
            continue
        total_windows += len(entity_rows) - train_cfg.window_size + 1
        if num_features is None:
            features, _ = _build_feature_matrix(entity_rows, data_cfg)
            num_features = int(features.shape[1])

    if total_windows == 0 or num_features is None:
        raise ValueError(
            "No windows were created. Lower WINDOW_SIZE or MIN_POINTS_PER_ENTITY, "
            "or verify the source data."
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    windows_memmap = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_windows, train_cfg.window_size, num_features),
    )

    metadata: list[WindowMetadata] = []
    write_idx = 0
    for window, item in _iter_entity_windows(grouped, train_cfg, data_cfg):
        windows_memmap[write_idx] = window
        metadata.append(item)
        write_idx += 1

    windows_memmap.flush()
    return (total_windows, train_cfg.window_size, num_features), metadata
