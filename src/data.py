from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from psycopg import connect

from .config import DbConfig, SourceConfig, TrainConfig


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
    if source.custom_sql:
        sql = source.custom_sql
    elif source.source_kind == "coingecko":
        sql = """
            SELECT
                coin_id::text AS entity_id,
                EXTRACT(EPOCH FROM ts)::double precision AS ts,
                price::double precision AS price,
                COALESCE(total_volume::double precision, 0.0) AS volume,
                COALESCE(market_cap::double precision, 0.0) AS market_cap
            FROM coingecko_market_data
            WHERE price IS NOT NULL
            ORDER BY coin_id, ts
        """
    elif source.source_kind == "binance":
        sql = """
            SELECT
                COALESCE(coingecko_id, base_asset)::text AS entity_id,
                EXTRACT(EPOCH FROM open_time)::double precision AS ts,
                open::double precision AS open,
                high::double precision AS high,
                low::double precision AS low,
                close::double precision AS close,
                close::double precision AS price,
                volume::double precision AS volume,
                quote_asset_volume::double precision AS quote_volume
            FROM binance_klines
            WHERE close IS NOT NULL
            ORDER BY entity_id, open_time
        """
    else:
        sql = f"""
            SELECT
                {source.entity_col}::text AS entity_id,
                EXTRACT(EPOCH FROM {source.time_col})::double precision AS ts,
                {source.price_col}::double precision AS price,
                COALESCE({source.aux_col}::double precision, 0.0) AS aux_value
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
            row["volume"] = float(row.get("aux_value", 0.0) or 0.0)
    if "market_cap" not in data[0]:
        for row in data:
            row["market_cap"] = 0.0
    if "quote_volume" not in data[0]:
        for row in data:
            row["quote_volume"] = 0.0
    for key in ("open", "high", "low", "close"):
        if key not in data[0]:
            for row in data:
                row[key] = float(row["price"])

    return data


def _window_zscore(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / (std + 1e-8)


def _safe_log_return(prices: np.ndarray) -> np.ndarray:
    clipped = np.clip(prices, 1e-12, None)
    returns = np.diff(np.log(clipped), prepend=np.log(clipped[0]))
    returns[0] = 0.0
    return returns


def _relative_price(prices: np.ndarray) -> np.ndarray:
    base = max(prices[0], 1e-12)
    return prices / base - 1.0


def _growth_rate(values: np.ndarray) -> np.ndarray:
    prev = np.clip(np.roll(values, 1), 1e-8, None)
    prev[0] = max(values[0], 1e-8)
    growth = values / prev - 1.0
    growth[0] = 0.0
    return growth


def group_by_entity(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        entity_id = str(row["entity_id"])
        grouped.setdefault(entity_id, []).append(row)
    return grouped


def build_windows(
    rows: list[dict],
    train_cfg: TrainConfig,
) -> tuple[np.ndarray, list[WindowMetadata]]:
    grouped = group_by_entity(rows)
    windows: list[np.ndarray] = []
    metadata: list[WindowMetadata] = []

    for entity_id, entity_rows in grouped.items():
        if len(entity_rows) < max(train_cfg.min_points_per_entity, train_cfg.window_size):
            continue

        prices = np.array([float(row["price"]) for row in entity_rows], dtype=np.float32)
        volume = np.array([float(row.get("volume", 0.0)) for row in entity_rows], dtype=np.float32)
        market_cap = np.array([float(row.get("market_cap", 0.0)) for row in entity_rows], dtype=np.float32)
        quote_volume = np.array([float(row.get("quote_volume", 0.0)) for row in entity_rows], dtype=np.float32)
        ts = np.array([float(row["ts"]) for row in entity_rows], dtype=np.float64)

        rel_price = _relative_price(prices)
        log_returns = _safe_log_return(prices)
        volatility = np.sqrt(
            np.convolve(log_returns ** 2, np.ones(8, dtype=np.float32) / 8.0, mode="same")
        )
        volume_log = np.log1p(np.clip(volume, 0.0, None))
        volume_delta = np.diff(volume_log, prepend=volume_log[0])
        volume_rate = _growth_rate(volume_log)
        market_cap_log = np.log1p(np.clip(market_cap, 0.0, None))
        quote_volume_log = np.log1p(np.clip(quote_volume, 0.0, None))

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

        window_size = train_cfg.window_size
        for start_idx in range(0, len(entity_rows) - window_size + 1):
            end_idx = start_idx + window_size
            chunk = features[start_idx:end_idx]
            normalized = _window_zscore(chunk)
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
            windows.append(normalized)
            metadata.append(
                WindowMetadata(
                    entity_id=entity_id,
                    start_ts=float(ts[start_idx]),
                    end_ts=float(ts[end_idx - 1]),
                    start_idx=start_idx,
                )
            )

    if not windows:
        raise ValueError(
            "No windows were created. Lower WINDOW_SIZE or MIN_POINTS_PER_ENTITY, "
            "or verify the source data."
        )

    return np.stack(windows), metadata
