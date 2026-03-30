from __future__ import annotations

import argparse
import json
import os
from html import escape
from datetime import UTC, datetime
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import torch

from .config import DEFAULT_TIMEFRAMES, DataConfig, DbConfig, PlotConfig, SearchConfig, SourceConfig
from .data import _build_feature_matrix, _window_zscore, fetch_series, group_by_entity
from .model import build_autoencoder


def cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    search_cfg = SearchConfig()
    matrix_norm = np.linalg.norm(matrix, axis=1) + search_cfg.epsilon_similarity
    vector_norm = np.linalg.norm(vector) + search_cfg.epsilon_similarity
    return (matrix @ vector) / (matrix_norm * vector_norm)


def load_artifacts(artifact_dir: str) -> tuple[dict, np.ndarray]:
    with open(os.path.join(artifact_dir, "metadata.json"), "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    embeddings = np.load(os.path.join(artifact_dir, "embeddings.npy"))
    return metadata, embeddings


def _is_artifact_dir(path: str) -> bool:
    return all(
        os.path.exists(os.path.join(path, name))
        for name in ("metadata.json", "embeddings.npy", "model.pt")
    )


def _resolve_artifact_dirs(artifact_dir: str) -> list[tuple[str, str]]:
    if _is_artifact_dir(artifact_dir):
        metadata, _ = load_artifacts(artifact_dir)
        timeframe = str(metadata.get("timeframe") or os.path.basename(os.path.abspath(artifact_dir)) or "default")
        return [(timeframe, artifact_dir)]

    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError(
            f"Artifact directory '{artifact_dir}' was not found. Run training first."
        )

    matches: list[tuple[str, str]] = []
    for timeframe in DEFAULT_TIMEFRAMES:
        candidate = os.path.join(artifact_dir, timeframe)
        if _is_artifact_dir(candidate):
            matches.append((timeframe, candidate))

    if matches:
        return matches

    for name in sorted(os.listdir(artifact_dir)):
        candidate = os.path.join(artifact_dir, name)
        if not os.path.isdir(candidate) or not _is_artifact_dir(candidate):
            continue
        metadata, _ = load_artifacts(candidate)
        timeframe = str(metadata.get("timeframe") or name)
        matches.append((timeframe, candidate))

    if matches:
        return matches

    raise FileNotFoundError(
        f"No trained artifact subdirectories were found inside '{artifact_dir}'."
    )


def _load_series_lookup(source_cfg_payload: dict) -> dict[str, dict[str, np.ndarray | str]]:
    source_cfg = SourceConfig(**source_cfg_payload)
    rows = fetch_series(DbConfig(), source_cfg)
    grouped = group_by_entity(rows)
    lookup: dict[str, dict[str, np.ndarray | str]] = {}
    for entity_id, entity_rows in grouped.items():
        lookup[entity_id] = {
            "price": np.array([float(row["price"]) for row in entity_rows], dtype=np.float64),
            "ts": np.array([float(row["ts"]) for row in entity_rows], dtype=np.float64),
            "open": np.array([float(row.get("open", row["price"])) for row in entity_rows], dtype=np.float64),
            "high": np.array([float(row.get("high", row["price"])) for row in entity_rows], dtype=np.float64),
            "low": np.array([float(row.get("low", row["price"])) for row in entity_rows], dtype=np.float64),
            "close": np.array([float(row.get("close", row["price"])) for row in entity_rows], dtype=np.float64),
            "base_asset": str(entity_rows[-1].get("base_asset") or entity_id).upper(),
        }
    return lookup


def _load_search_model(artifact_dir: str, train_config: dict):
    model = build_autoencoder(
        model_type=train_config["model_type"],
        window_size=int(train_config["window_size"]),
        num_features=7,
        hidden_dim=int(train_config["hidden_dim"]),
        latent_dim=int(train_config["latent_dim"]),
        cnn_channels=int(train_config.get("cnn_channels", 64)),
        cnn_kernel_size=int(train_config.get("cnn_kernel_size", 5)),
    )
    state = torch.load(os.path.join(artifact_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _fetch_live_binance_rows(entity_id: str, base_asset: str, timeframe: str, window_size: int) -> list[dict]:
    search_cfg = SearchConfig()
    symbol = f"{base_asset.upper()}{search_cfg.binance_query_quote_asset.upper()}"
    params = urlencode({"symbol": symbol, "interval": timeframe, "limit": window_size})
    url = f"{search_cfg.binance_api_base_url.rstrip('/')}" + f"/api/v3/klines?{params}"
    with urlopen(url, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, list) or len(payload) < window_size:
        raise ValueError(
            f"Binance API returned insufficient klines for entity={entity_id} symbol={symbol} timeframe={timeframe}."
        )

    rows: list[dict] = []
    for item in payload:
        rows.append(
            {
                "entity_id": entity_id,
                "base_asset": base_asset.upper(),
                "ts": float(item[0]) / 1000.0,
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "price": float(item[4]),
                "volume": float(item[5]),
                "quote_volume": float(item[7]),
                "market_cap": 0.0,
            }
        )
    return rows


def _build_live_query(
    model,
    entity_id: str,
    timeframe: str,
    window_size: int,
    series_lookup: dict[str, dict[str, np.ndarray | str]],
) -> tuple[dict, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    base_asset = str(series_lookup.get(entity_id, {}).get("base_asset") or entity_id).upper()
    rows = _fetch_live_binance_rows(entity_id=entity_id, base_asset=base_asset, timeframe=timeframe, window_size=window_size)
    features, ts = _build_feature_matrix(rows, DataConfig())
    window = _window_zscore(features[-window_size:]).astype(np.float32)
    tensor = torch.tensor(window[np.newaxis, ...], dtype=torch.float32)
    with torch.no_grad():
        query_vector = model.encode(tensor).cpu().numpy()[0]

    query_series = {
        "price": np.array([float(row["price"]) for row in rows], dtype=np.float64),
        "ts": np.array([float(row["ts"]) for row in rows], dtype=np.float64),
        "open": np.array([float(row["open"]) for row in rows], dtype=np.float64),
        "high": np.array([float(row["high"]) for row in rows], dtype=np.float64),
        "low": np.array([float(row["low"]) for row in rows], dtype=np.float64),
        "close": np.array([float(row["close"]) for row in rows], dtype=np.float64),
    }
    query_item = {
        "entity_id": entity_id,
        "start_ts": float(ts[-window_size]),
        "end_ts": float(ts[-1]),
        "start_idx": 0,
        "source": "binance_api",
    }
    query_close = query_series["close"][:window_size].astype(np.float64)
    return query_item, query_series, query_vector, query_close


def _future_return(
    prices_by_entity: dict[str, np.ndarray],
    entity_id: str,
    start_idx: int,
    window_size: int,
    horizon: int,
) -> float | None:
    series = prices_by_entity.get(entity_id)
    if series is None:
        return None

    end_idx = start_idx + window_size - 1
    future_idx = end_idx + horizon
    if future_idx >= len(series):
        return None

    current_price = float(series[end_idx])
    future_price = float(series[future_idx])
    if current_price <= 0:
        return None
    return future_price / current_price - 1.0


def _format_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _verdict(mean_value: float, positive_rate: float) -> str:
    search_cfg = SearchConfig()
    if (
        mean_value >= search_cfg.bullish_mean_threshold
        and positive_rate >= search_cfg.bullish_positive_rate_threshold
    ):
        return "bullish"
    if (
        mean_value <= search_cfg.bearish_mean_threshold
        and positive_rate <= search_cfg.bearish_positive_rate_threshold
    ):
        return "bearish"
    return "neutral"


def _relative_series(values: np.ndarray) -> np.ndarray:
    search_cfg = SearchConfig()
    base = max(float(values[0]), search_cfg.epsilon_price)
    return values / base - 1.0


def _zscore_1d(values: np.ndarray) -> np.ndarray:
    search_cfg = SearchConfig()
    values = values.astype(np.float64)
    return (values - values.mean()) / (values.std() + search_cfg.epsilon_similarity)


def _window_close_series(
    series_lookup: dict[str, dict[str, np.ndarray]],
    item: dict,
    window_size: int,
) -> np.ndarray:
    start_idx = int(item["start_idx"])
    end_idx = start_idx + window_size
    return series_lookup[item["entity_id"]]["close"][start_idx:end_idx].astype(np.float64)


def _shape_features(close_prices: np.ndarray) -> dict[str, np.ndarray | float]:
    search_cfg = SearchConfig()
    rel = _relative_series(close_prices)
    log_returns = np.diff(
        np.log(np.clip(close_prices, search_cfg.epsilon_price, None)),
        prepend=np.log(max(close_prices[0], search_cfg.epsilon_price)),
    )
    log_returns[0] = 0.0
    amplitude = float(np.max(rel) - np.min(rel))
    volatility = float(np.std(log_returns))
    return {
        "rel": rel,
        "zrel": _zscore_1d(rel),
        "amplitude": amplitude,
        "volatility": volatility,
    }


def _safe_ratio(a: float, b: float) -> float:
    search_cfg = SearchConfig()
    return a / max(b, search_cfg.epsilon_similarity)


def _window_similarity(
    query_close: np.ndarray,
    match_close: np.ndarray,
    embedding_score: float,
) -> dict[str, float]:
    search_cfg = SearchConfig()
    query_feat = _shape_features(query_close)
    match_feat = _shape_features(match_close)
    shape_cosine = float(
        np.dot(query_feat["zrel"], match_feat["zrel"])
        / (
            (np.linalg.norm(query_feat["zrel"]) + search_cfg.epsilon_similarity)
            * (np.linalg.norm(match_feat["zrel"]) + search_cfg.epsilon_similarity)
        )
    )
    path_mae = float(np.mean(np.abs(query_feat["rel"] - match_feat["rel"])))
    path_score = 1.0 / (1.0 + path_mae * search_cfg.path_score_scale)
    final_score = (
        search_cfg.final_score_embedding_weight * embedding_score
        + search_cfg.final_score_shape_weight * shape_cosine
        + search_cfg.final_score_path_weight * path_score
    )
    return {
        "embedding_score": embedding_score,
        "shape_score": shape_cosine,
        "path_score": path_score,
        "final_score": final_score,
        "query_amplitude": float(query_feat["amplitude"]),
        "match_amplitude": float(match_feat["amplitude"]),
        "query_volatility": float(query_feat["volatility"]),
        "match_volatility": float(match_feat["volatility"]),
    }


def _deduplicate_candidates(
    candidates: list[tuple[float, int, dict, dict[str, float]]],
    window_size: int,
    candle_seconds: float,
) -> list[tuple[float, int, dict, dict[str, float]]]:
    selected: list[tuple[float, int, dict, dict[str, float]]] = []
    selected_starts_by_entity: dict[str, list[int]] = {}
    selected_start_timestamps: list[float] = []
    min_time_gap = max(candle_seconds * window_size, 0.0)

    for candidate in candidates:
        _, _, item, _ = candidate
        entity_id = str(item["entity_id"])
        start_idx = int(item["start_idx"])
        start_ts = float(item.get("start_ts", 0.0))

        prior_starts = selected_starts_by_entity.get(entity_id, [])
        if any(abs(start_idx - prior_start) < window_size for prior_start in prior_starts):
            continue
        if min_time_gap > 0.0 and any(abs(start_ts - prior_ts) < min_time_gap for prior_ts in selected_start_timestamps):
            continue

        selected.append(candidate)
        selected_starts_by_entity.setdefault(entity_id, []).append(start_idx)
        selected_start_timestamps.append(start_ts)

    return selected


def _score_to_weight(score: float) -> float:
    search_cfg = SearchConfig()
    return max(float(score), 0.0) ** search_cfg.analog_forecast_weight_power


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = quantile * float(np.sum(sorted_weights))
    idx = int(np.searchsorted(cumulative, threshold, side="left"))
    idx = min(max(idx, 0), len(sorted_values) - 1)
    return float(sorted_values[idx])


def _weighted_positive_rate(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    positive_mask = (values > 0).astype(np.float64)
    return float(np.sum(positive_mask * weights) / np.sum(weights))


def _effective_sample_size(weights: np.ndarray) -> float:
    if len(weights) == 0:
        return 0.0
    denom = float(np.sum(weights ** 2))
    if denom <= 0.0:
        return 0.0
    total = float(np.sum(weights))
    return (total * total) / denom


def _confidence_label(match_count: int, effective_n: float) -> str:
    search_cfg = SearchConfig()
    if match_count < search_cfg.analog_forecast_min_matches or effective_n < search_cfg.analog_forecast_min_matches:
        return "low"
    if match_count >= search_cfg.analog_forecast_high_confidence_matches and effective_n >= search_cfg.analog_forecast_high_confidence_matches:
        return "high"
    return "medium"


def _build_plot_series_from_series(
    entity_series: dict[str, np.ndarray],
    start_idx: int,
    window_size: int,
    max_horizon: int,
) -> dict[str, np.ndarray | int]:
    search_cfg = SearchConfig()
    prices = entity_series["price"]
    end_idx = start_idx + window_size
    plot_end_idx = min(len(prices), end_idx + max_horizon)
    slice_prices = prices[start_idx:plot_end_idx]
    x = np.arange(len(slice_prices))

    close_rel = _relative_series(slice_prices)
    open_rel = entity_series["open"][start_idx:plot_end_idx] / max(float(slice_prices[0]), search_cfg.epsilon_price) - 1.0
    high_rel = entity_series["high"][start_idx:plot_end_idx] / max(float(slice_prices[0]), search_cfg.epsilon_price) - 1.0
    low_rel = entity_series["low"][start_idx:plot_end_idx] / max(float(slice_prices[0]), search_cfg.epsilon_price) - 1.0

    return {
        "x": x,
        "close": close_rel,
        "open": open_rel,
        "high": high_rel,
        "low": low_rel,
        "split_idx": window_size,
    }


def _build_plot_series(
    series_lookup: dict[str, dict[str, np.ndarray | str]],
    item: dict,
    window_size: int,
    max_horizon: int,
) -> dict[str, np.ndarray | int]:
    entity_series = series_lookup[item["entity_id"]]
    return _build_plot_series_from_series(entity_series, int(item["start_idx"]), window_size, max_horizon)


def _build_absolute_plot_series_from_series(
    entity_series: dict[str, np.ndarray],
    start_idx: int,
    window_size: int,
    max_horizon: int,
) -> dict[str, np.ndarray | int]:
    prices = entity_series["price"]
    end_idx = start_idx + window_size
    plot_end_idx = min(len(prices), end_idx + max_horizon)
    x = np.arange(plot_end_idx - start_idx)
    return {
        "x": x,
        "close": entity_series["close"][start_idx:plot_end_idx].astype(np.float64),
        "open": entity_series["open"][start_idx:plot_end_idx].astype(np.float64),
        "high": entity_series["high"][start_idx:plot_end_idx].astype(np.float64),
        "low": entity_series["low"][start_idx:plot_end_idx].astype(np.float64),
        "split_idx": window_size,
    }


def _build_absolute_plot_series(
    series_lookup: dict[str, dict[str, np.ndarray | str]],
    item: dict,
    window_size: int,
    max_horizon: int,
) -> dict[str, np.ndarray | int]:
    entity_series = series_lookup[item["entity_id"]]
    return _build_absolute_plot_series_from_series(entity_series, int(item["start_idx"]), window_size, max_horizon)


def _has_future_candles(
    series_lookup: dict[str, dict[str, np.ndarray | str]],
    item: dict,
    window_size: int,
) -> bool:
    entity_series = series_lookup[item["entity_id"]]
    end_idx = int(item["start_idx"]) + window_size
    return end_idx < len(entity_series["close"])


def _build_forecast_overlay(
    query_plot: dict[str, np.ndarray | int],
    match_plots: list[dict[str, np.ndarray | int]],
    forecast_len: int,
    match_scores: list[float],
    clamp_min: float | None = None,
    relative_mode: bool = False,
) -> dict[str, np.ndarray] | None:
    split_idx = int(query_plot["split_idx"])
    if not match_plots:
        return None

    future_paths: list[np.ndarray] = []
    weights: list[float] = []
    for match_plot, score in zip(match_plots, match_scores):
        match_close = np.asarray(match_plot["close"], dtype=np.float64)
        if len(match_close) <= split_idx:
            continue
        if relative_mode:
            base_price = float(match_close[split_idx - 1])
            if base_price <= 0.0:
                continue
            anchored_future = match_close[split_idx : split_idx + forecast_len] / base_price - 1.0
        else:
            anchored_future = match_close[split_idx : split_idx + forecast_len] - float(match_close[split_idx - 1])
        if len(anchored_future) == 0:
            continue
        future_paths.append(anchored_future)
        weights.append(_score_to_weight(score))

    if not future_paths:
        return None

    max_len = min(max(len(path) for path in future_paths), forecast_len)
    if max_len <= 0:
        return None

    query_close = np.asarray(query_plot["close"], dtype=np.float64)
    base_value = float(query_close[split_idx - 1])
    future_x = np.arange(split_idx - 1, split_idx + max_len, dtype=np.float64)
    median_values = [base_value]
    low_values = [base_value]
    high_values = [base_value]

    for step_idx in range(max_len):
        step_values = []
        step_weights = []
        for path, weight in zip(future_paths, weights):
            if step_idx < len(path):
                step_values.append(path[step_idx])
                step_weights.append(weight)
        if not step_values:
            break
        step_arr = np.array(step_values, dtype=np.float64)
        weight_arr = np.array(step_weights, dtype=np.float64)
        if relative_mode:
            median_values.append(base_value * (1.0 + _weighted_quantile(step_arr, weight_arr, 0.5)))
            low_values.append(base_value * (1.0 + _weighted_quantile(step_arr, weight_arr, 0.25)))
            high_values.append(base_value * (1.0 + _weighted_quantile(step_arr, weight_arr, 0.75)))
        else:
            median_values.append(base_value + _weighted_quantile(step_arr, weight_arr, 0.5))
            low_values.append(base_value + _weighted_quantile(step_arr, weight_arr, 0.25))
            high_values.append(base_value + _weighted_quantile(step_arr, weight_arr, 0.75))

    length = min(len(median_values), len(future_x))
    median_arr = np.array(median_values[:length], dtype=np.float64)
    low_arr = np.array(low_values[:length], dtype=np.float64)
    high_arr = np.array(high_values[:length], dtype=np.float64)
    if clamp_min is not None:
        median_arr = np.maximum(median_arr, clamp_min)
        low_arr = np.maximum(low_arr, clamp_min)
        high_arr = np.maximum(high_arr, clamp_min)
    return {
        "x": future_x[:length],
        "median": median_arr,
        "low": low_arr,
        "high": high_arr,
    }


def _save_plot(
    plot_dir: str,
    query_item: dict,
    matches: list[tuple[int, dict, float]],
    series_lookup: dict[str, dict[str, np.ndarray | str]],
    window_size: int,
    horizons: list[int],
    timeframe: str,
    query_series: dict[str, np.ndarray] | None = None,
    include_relative_plots: bool = False,
) -> str:
    plot_cfg = PlotConfig()
    os.makedirs(plot_dir, exist_ok=True)
    forecast_len = window_size
    plot_extension_len = forecast_len
    if plot_cfg.show_query_actual_future:
        plot_extension_len = max(plot_extension_len, max(horizons) if horizons else 0)

    if query_series is not None:
        query_plot = _build_plot_series_from_series(query_series, 0, window_size, plot_extension_len)
        query_plot_raw = _build_absolute_plot_series_from_series(query_series, 0, window_size, plot_extension_len)
    else:
        query_plot = _build_plot_series(series_lookup, query_item, window_size, plot_extension_len)
        query_plot_raw = _build_absolute_plot_series(series_lookup, query_item, window_size, plot_extension_len)

    query_x = query_plot["x"]
    query_close = query_plot["close"]
    query_x_raw = query_plot_raw["x"]
    query_close_raw = query_plot_raw["close"]
    split_idx = int(query_plot["split_idx"])
    colors = plot_cfg.colors

    plot_matches = matches
    match_plot_series = [
        _build_plot_series(series_lookup, item, window_size, plot_extension_len) for _, item, _ in plot_matches
    ]
    match_plot_series_raw = [
        _build_absolute_plot_series(series_lookup, item, window_size, plot_extension_len) for _, item, _ in plot_matches
    ]

    scores = [score for _, _, score in matches]
    forecast_overlay = _build_forecast_overlay(
        query_plot,
        match_plot_series,
        forecast_len=forecast_len,
        match_scores=scores,
    )
    forecast_overlay_raw = _build_forecast_overlay(
        query_plot_raw,
        match_plot_series_raw,
        forecast_len=forecast_len,
        match_scores=scores,
        clamp_min=0.0,
        relative_mode=True,
    )

    panels: list[dict] = [
        {
            "title": (
                f"Query Original Price: {query_item['entity_id']} | "
                f"{_format_ts(query_item['start_ts'])} -> {_format_ts(query_item['end_ts'])}"
            ),
            "y_axis_format": "price",
            "group": "query",
            "series": [
                {
                    "x": query_x_raw[:split_idx],
                    "open": query_plot_raw["open"][:split_idx],
                    "high": query_plot_raw["high"][:split_idx],
                    "low": query_plot_raw["low"][:split_idx],
                    "close": query_close_raw[:split_idx],
                    "color": "#111111",
                    "label": "query candles",
                    "kind": "candles",
                },
                {
                    "x": forecast_overlay_raw["x"],
                    "low": forecast_overlay_raw["low"],
                    "high": forecast_overlay_raw["high"],
                    "color": plot_cfg.forecast_band_color,
                    "label": "scenario IQR",
                    "opacity": plot_cfg.forecast_band_opacity,
                    "kind": "band",
                }
                if forecast_overlay_raw is not None
                else None,
                {
                    "x": forecast_overlay_raw["x"],
                    "y": forecast_overlay_raw["median"],
                    "color": plot_cfg.forecast_line_color,
                    "label": "scenario median",
                    "dashed": True,
                    "kind": "line",
                }
                if forecast_overlay_raw is not None
                else None,
            ],
            "split_idx": split_idx,
            "x_max_override": (window_size * 2) - 1,
        }
    ]

    if include_relative_plots:
        panels.insert(
            0,
            {
                "title": (
                    f"Query: {query_item['entity_id']} | "
                    f"{_format_ts(query_item['start_ts'])} -> {_format_ts(query_item['end_ts'])}"
                ),
                "y_axis_format": "percent",
                "group": "query",
                "series": [
                    {
                        "x": query_x[:split_idx],
                        "open": query_plot["open"][:split_idx],
                        "high": query_plot["high"][:split_idx],
                        "low": query_plot["low"][:split_idx],
                        "close": query_close[:split_idx],
                        "color": "#111111",
                        "label": "query candles",
                        "kind": "candles",
                    },
                    {
                        "x": query_x[split_idx - 1 :],
                        "y": query_close[split_idx - 1 :],
                        "color": "#111111",
                        "label": "query future",
                        "dashed": True,
                        "kind": "line",
                    }
                    if plot_cfg.show_query_actual_future and len(query_x) > split_idx
                    else None,
                    {
                        "x": forecast_overlay["x"],
                        "low": forecast_overlay["low"],
                        "high": forecast_overlay["high"],
                        "color": plot_cfg.forecast_band_color,
                        "label": "scenario IQR",
                        "opacity": plot_cfg.forecast_band_opacity,
                        "kind": "band",
                    }
                    if forecast_overlay is not None
                    else None,
                    {
                        "x": forecast_overlay["x"],
                        "y": forecast_overlay["median"],
                        "color": plot_cfg.forecast_line_color,
                        "label": "scenario median",
                        "dashed": True,
                        "kind": "line",
                    }
                    if forecast_overlay is not None
                    else None,
                ],
                "split_idx": split_idx,
                "x_max_override": (window_size * 2) - 1,
            },
        )

    for plot_idx, ((rank_idx, item, score), match_plot_raw) in enumerate(
        zip(plot_matches, match_plot_series_raw), start=1
    ):
        match_x_raw = match_plot_raw["x"]
        match_close_raw = match_plot_raw["close"]
        split_idx_raw = int(match_plot_raw["split_idx"])
        color = colors[(plot_idx - 1) % len(colors)]
        group_name = f"match_{rank_idx}"

        if include_relative_plots:
            match_plot = match_plot_series[plot_idx - 1]
            match_x = match_plot["x"]
            match_close = match_plot["close"]
            split_idx_rel = int(match_plot["split_idx"])
            panels.append(
                {
                    "title": (
                        f"rank={rank_idx} score={score:.4f} entity={item['entity_id']} | "
                        f"{_format_ts(item['start_ts'])} -> {_format_ts(item['end_ts'])}"
                    ),
                    "y_axis_format": "percent",
                    "group": group_name,
                    "series": [
                        {
                            "x": query_x[:window_size],
                            "y": query_close[:window_size],
                            "color": "#111111",
                            "label": "query",
                            "kind": "line",
                        },
                        {
                            "x": match_x[:window_size],
                            "open": match_plot["open"][:window_size],
                            "high": match_plot["high"][:window_size],
                            "low": match_plot["low"][:window_size],
                            "close": match_close[:window_size],
                            "color": color,
                            "label": f"match #{rank_idx} candles",
                            "kind": "candles",
                        },
                        {
                            "x": match_x[split_idx_rel: split_idx_rel + forecast_len],
                            "open": match_plot["open"][split_idx_rel: split_idx_rel + forecast_len],
                            "high": match_plot["high"][split_idx_rel: split_idx_rel + forecast_len],
                            "low": match_plot["low"][split_idx_rel: split_idx_rel + forecast_len],
                            "close": match_close[split_idx_rel: split_idx_rel + forecast_len],
                            "color": color,
                            "label": "match future candles",
                            "kind": "candles",
                        }
                        if len(match_x) > split_idx_rel
                        else None,
                    ],
                    "split_idx": split_idx_rel,
                    "x_max_override": (window_size * 2) - 1,
                }
            )

        panels.append(
            {
                "title": (
                    f"rank={rank_idx} score={score:.4f} original price entity={item['entity_id']} | "
                    f"{_format_ts(item['start_ts'])} -> {_format_ts(item['end_ts'])}"
                ),
                "y_axis_format": "price",
                "group": group_name,
                "series": [
                    {
                        "x": match_x_raw[:window_size],
                        "open": match_plot_raw["open"][:window_size],
                        "high": match_plot_raw["high"][:window_size],
                        "low": match_plot_raw["low"][:window_size],
                        "close": match_close_raw[:window_size],
                        "color": color,
                        "label": f"match #{rank_idx} candles",
                        "kind": "candles",
                    },
                    {
                        "x": match_x_raw[split_idx_raw: split_idx_raw + forecast_len],
                        "open": match_plot_raw["open"][split_idx_raw: split_idx_raw + forecast_len],
                        "high": match_plot_raw["high"][split_idx_raw: split_idx_raw + forecast_len],
                        "low": match_plot_raw["low"][split_idx_raw: split_idx_raw + forecast_len],
                        "close": match_close_raw[split_idx_raw: split_idx_raw + forecast_len],
                        "color": color,
                        "label": "match future candles",
                        "kind": "candles",
                    }
                    if len(match_x_raw) > split_idx_raw
                    else None,
                ],
                "split_idx": split_idx_raw,
                "x_max_override": (window_size * 2) - 1,
            }
        )


    panels = [
        {
            **panel,
            "series": [series for series in panel["series"] if series is not None],
        }
        for panel in panels
    ]

    outer_width = plot_cfg.outer_width
    panel_height = plot_cfg.panel_height
    padding_left = plot_cfg.padding_left
    padding_right = plot_cfg.padding_right
    padding_top = plot_cfg.padding_top
    padding_bottom = plot_cfg.padding_bottom
    legend_y_offset = plot_cfg.legend_y_offset
    total_height = len(panels) * panel_height

    def wrap_title(title: str) -> list[str]:
        words = title.split()
        if not words:
            return [title]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= plot_cfg.title_max_chars:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def format_y_tick(value: float, axis_format: str) -> str:
        if axis_format == "price":
            return f"{value:.4g}"
        return f"{value:.1%}"

    def panel_y_bounds(panel: dict) -> tuple[float, float]:
        panel_values = np.concatenate(
            [
                values
                for s in panel["series"]
                for values in (
                    [s["y"]]
                    if s.get("kind") == "line"
                    else [s["low"], s["high"]]
                    if s.get("kind") == "band"
                    else [s["open"], s["high"], s["low"], s["close"]]
                )
            ]
        )
        y_min = float(np.min(panel_values))
        y_max = float(np.max(panel_values))
        if abs(y_max - y_min) < 1e-9:
            base = max(abs(y_max), 1.0)
            if panel.get("y_axis_format") == "price":
                y_max += base * 0.01
            else:
                y_min -= base * 0.01
                y_max += base * 0.01
        pad = (y_max - y_min) * 0.05
        if panel.get("y_axis_format") == "price":
            bounded_min = max(0.0, y_min)
            bounded_max = y_max + pad
        else:
            bounded_min = y_min - pad
            bounded_max = y_max + pad
        return bounded_min, bounded_max

    def map_y(y_val: float, content_top: float, content_height: float, y_min: float, y_max: float) -> float:
        return content_top + (1.0 - ((float(y_val) - y_min) / (y_max - y_min))) * content_height

    def to_polyline(
        xs: np.ndarray,
        ys: np.ndarray,
        width: int,
        panel_x_max: float,
        content_top: float,
        content_height: float,
        y_min: float,
        y_max: float,
    ) -> str:
        plot_width = width - padding_left - padding_right
        points = []
        for x_val, y_val in zip(xs, ys):
            px = padding_left + (float(x_val) / max(panel_x_max, 1.0)) * plot_width
            py = map_y(float(y_val), content_top, content_height, y_min, y_max)
            points.append(f"{px:.2f},{py:.2f}")
        return " ".join(points)

    def to_band_polygon(
        xs: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        width: int,
        panel_x_max: float,
        content_top: float,
        content_height: float,
        y_min: float,
        y_max: float,
    ) -> str:
        plot_width = width - padding_left - padding_right
        upper_points = []
        lower_points = []
        for x_val, low_val, high_val in zip(xs, low, high):
            px = padding_left + (float(x_val) / max(panel_x_max, 1.0)) * plot_width
            upper_points.append(f"{px:.2f},{map_y(float(high_val), content_top, content_height, y_min, y_max):.2f}")
            lower_points.append(f"{px:.2f},{map_y(float(low_val), content_top, content_height, y_min, y_max):.2f}")
        return " ".join(upper_points + list(reversed(lower_points)))

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{outer_width}" height="{total_height}" viewBox="0 0 {outer_width} {total_height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827} .small{font-size:12px} .title{font-size:16px;font-weight:bold} .axis{stroke:#9ca3af;stroke-width:1} .grid{stroke:#e5e7eb;stroke-width:1} .split{stroke:#6b7280;stroke-width:1;stroke-dasharray:4 4}</style>',
    ]

    panel_tops = [idx * panel_height for idx in range(len(panels))]
    group_ranges: list[tuple[int, int, str]] = []
    if panels:
        group_start = 0
        current_group = str(panels[0].get("group", "group_0"))
        for idx in range(1, len(panels)):
            panel_group = str(panels[idx].get("group", f"group_{idx}"))
            if panel_group != current_group:
                group_ranges.append((group_start, idx - 1, current_group))
                group_start = idx
                current_group = panel_group
        group_ranges.append((group_start, len(panels) - 1, current_group))

    for start_idx, end_idx, _group_name in group_ranges:
        y = panel_tops[start_idx] + 6
        height = (panel_tops[end_idx] + panel_height) - panel_tops[start_idx] - 12
        svg_parts.append(
            f'<rect x="6" y="{y:.2f}" width="{outer_width - 12}" height="{height:.2f}" rx="10" ry="10" fill="#f8fafc" stroke="#dbe4f0" stroke-width="1"/>'
        )

    for panel_idx, panel in enumerate(panels):
        top = panel_idx * panel_height
        plot_width = outer_width - padding_left - padding_right
        panel_x_max = float(
            panel.get(
                "x_max_override",
                max(float(np.max(series["x"])) for series in panel["series"] if len(series["x"]) > 0),
            )
        )
        y_min, y_max = panel_y_bounds(panel)
        title_lines = wrap_title(panel["title"])
        title_y = top + 22
        title_last_y = title_y + (len(title_lines) - 1) * plot_cfg.title_line_height
        legend_y = max(top + legend_y_offset, title_last_y + 14)
        content_top = max(top + padding_top, legend_y + 12)
        content_height = panel_height - (content_top - top) - padding_bottom
        content_bottom = content_top + content_height
        tspans = "".join(
            f'<tspan x="{padding_left}" y="{title_y + idx * plot_cfg.title_line_height}">{escape(line)}</tspan>'
            for idx, line in enumerate(title_lines)
        )
        svg_parts.append(f'<text class="title">{tspans}</text>')

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = content_top + frac * content_height
            svg_parts.append(
                f'<line x1="{padding_left}" y1="{y:.2f}" x2="{padding_left + plot_width}" y2="{y:.2f}" class="grid"/>'
            )
            value = y_max - frac * (y_max - y_min)
            svg_parts.append(
                f'<text x="8" y="{y + 4:.2f}" class="small">{format_y_tick(value, panel.get("y_axis_format", "percent"))}</text>'
            )

        svg_parts.append(
            f'<line x1="{padding_left}" y1="{content_top:.2f}" x2="{padding_left}" y2="{content_bottom:.2f}" class="axis"/>'
        )
        svg_parts.append(
            f'<line x1="{padding_left}" y1="{content_bottom:.2f}" x2="{padding_left + plot_width}" y2="{content_bottom:.2f}" class="axis"/>'
        )

        split_x = padding_left + ((panel["split_idx"] - 1) / max(panel_x_max, 1.0)) * plot_width
        svg_parts.append(
            f'<line x1="{split_x:.2f}" y1="{content_top:.2f}" x2="{split_x:.2f}" y2="{content_bottom:.2f}" class="split"/>'
        )

        legend_x = padding_left
        for series in panel["series"]:
            dash = ' stroke-dasharray="6 4"' if series.get("dashed") else ""
            if series.get("kind") == "candles":
                candle_body = max(
                    min(plot_width / max(window_size * plot_cfg.candle_body_divisor, 1), plot_cfg.candle_body_max),
                    plot_cfg.candle_body_min,
                )
                for x_val, open_val, high_val, low_val, close_val in zip(
                    series["x"], series["open"], series["high"], series["low"], series["close"]
                ):
                    px = padding_left + (float(x_val) / max(panel_x_max, 1.0)) * plot_width
                    wick_y1 = map_y(float(high_val), content_top, content_height, y_min, y_max)
                    wick_y2 = map_y(float(low_val), content_top, content_height, y_min, y_max)
                    open_y = map_y(float(open_val), content_top, content_height, y_min, y_max)
                    close_y = map_y(float(close_val), content_top, content_height, y_min, y_max)
                    body_y = min(open_y, close_y)
                    body_h = max(abs(close_y - open_y), plot_cfg.candle_body_min_height)
                    body_color = plot_cfg.up_candle_color if close_val >= open_val else plot_cfg.down_candle_color
                    svg_parts.append(
                        f'<line x1="{px:.2f}" y1="{wick_y1:.2f}" x2="{px:.2f}" y2="{wick_y2:.2f}" stroke="{body_color}" stroke-width="1.2"/>'
                    )
                    svg_parts.append(
                        f'<rect x="{px - candle_body / 2:.2f}" y="{body_y:.2f}" width="{candle_body:.2f}" height="{body_h:.2f}" fill="{body_color}" opacity="0.85"/>'
                    )
            elif series.get("kind") == "band":
                polygon = to_band_polygon(series["x"], series["low"], series["high"], outer_width, panel_x_max, content_top, content_height, y_min, y_max)
                svg_parts.append(
                    f'<polygon points="{polygon}" fill="{series["color"]}" opacity="{float(series.get("opacity", 0.2)):.2f}"/>'
                )
            else:
                polyline = to_polyline(series["x"], series["y"], outer_width, panel_x_max, content_top, content_height, y_min, y_max)
                svg_parts.append(
                    f'<polyline fill="none" stroke="{series["color"]}" stroke-width="2"{dash} points="{polyline}"/>'
                )
            svg_parts.append(
                f'<line x1="{legend_x}" y1="{legend_y:.2f}" x2="{legend_x + 18}" y2="{legend_y:.2f}" stroke="{series["color"]}" stroke-width="2"{dash}/>'
            )
            svg_parts.append(
                f'<text x="{legend_x + 24}" y="{legend_y + 4:.2f}" class="small">{escape(series["label"])}</text>'
            )
            legend_x += 180

        x_axis_label = "Price timeline from window start" if panel.get("y_axis_format") == "price" else "Candles from window start"
        svg_parts.append(
            f'<text x="{padding_left}" y="{top + panel_height - 10}" class="small">{x_axis_label}</text>'
        )

    svg_parts.append("</svg>")
    avg_score = 0.0
    if matches:
        avg_score = float(sum(score for _, _, score in matches) / len(matches))
    output_path = os.path.join(
        plot_dir,
        f"{timeframe}_{query_item['entity_id']}_avgscore_{avg_score:.4f}.svg",
    )
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(svg_parts))
    return output_path

def search(
    entity_id: str | None,
    top_k: int,
    artifact_dir: str,
    horizons: list[int],
    plot_dir: str | None,
    only_past: bool,
    gap_candles: int,
    calendar_gap_candles: int,
    min_range_ratio: float,
    max_range_ratio: float,
    min_vol_ratio: float,
    max_vol_ratio: float,
    min_shape_score: float,
    min_final_score: float,
    exclude_empty_plots: bool,
    include_relative_plots: bool,
) -> None:
    for timeframe, resolved_artifact_dir in _resolve_artifact_dirs(artifact_dir):
        metadata, embeddings = load_artifacts(resolved_artifact_dir)
        windows = metadata["windows"]
        train_config = metadata["train_config"]
        source_cfg_payload = metadata.get("source_config", {"source_kind": "binance"})
        series_lookup = _load_series_lookup(source_cfg_payload)
        prices_by_entity = {
            entity_name: values["price"] for entity_name, values in series_lookup.items()
        } if horizons else {}
        window_size = int(train_config["window_size"])

        if entity_id is None:
            raise ValueError("--entity is required because query is fetched from live Binance API.")

        candidate_indexes = [idx for idx, item in enumerate(windows) if item["entity_id"] == entity_id]
        if not candidate_indexes:
            print(f"timeframe={timeframe} entity={entity_id} status=not_found")
            continue

        model = _load_search_model(resolved_artifact_dir, train_config)
        query_item, query_series, query_vector, query_close = _build_live_query(
            model=model,
            entity_id=entity_id,
            timeframe=timeframe,
            window_size=window_size,
            series_lookup=series_lookup,
        )
        query_start_ts = float(query_item["start_ts"])
        query_end_ts = float(query_item["end_ts"])
        candle_seconds = 0.0
        if window_size > 1:
            candle_seconds = max((query_end_ts - query_start_ts) / (window_size - 1), 0.0)
        global_cutoff_ts = query_start_ts - (calendar_gap_candles * candle_seconds)

        sims = cosine_similarity(embeddings, query_vector)

        candidates: list[tuple[float, int, dict, dict[str, float]]] = []
        for idx in np.argsort(-sims):
            item = windows[int(idx)]
            if only_past:
                match_end_ts = float(item["end_ts"])
                if match_end_ts >= global_cutoff_ts:
                    continue

            match_close = _window_close_series(series_lookup, item, window_size)
            similarity = _window_similarity(query_close, match_close, float(sims[idx]))
            amplitude_ratio = _safe_ratio(similarity["match_amplitude"], similarity["query_amplitude"])
            volatility_ratio = _safe_ratio(similarity["match_volatility"], similarity["query_volatility"])
            if not (min_range_ratio <= amplitude_ratio <= max_range_ratio):
                continue
            if not (min_vol_ratio <= volatility_ratio <= max_vol_ratio):
                continue
            if similarity["shape_score"] < min_shape_score:
                continue
            if similarity["final_score"] < min_final_score:
                continue
            candidates.append((similarity["final_score"], int(idx), item, similarity))

        candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = _deduplicate_candidates(candidates, window_size=window_size, candle_seconds=candle_seconds)
        printed = 0
        matches_for_plot: list[tuple[int, dict, float]] = []
        aggregate_outcomes: dict[int, list[tuple[float, float]]] = {horizon: [] for horizon in horizons}
        selected_scores: list[float] = []
        print(f"timeframe={timeframe} artifact_dir={resolved_artifact_dir}")
        print(
            f"timeframe={timeframe} query_source=binance_api entity={query_item['entity_id']} "
            f"pattern_size={window_size} candles "
            f"start_ts={query_item['start_ts']} ({_format_ts(query_item['start_ts'])}) "
            f"end_ts={query_item['end_ts']} ({_format_ts(query_item['end_ts'])})"
        )
        if only_past:
            print(
                f"timeframe={timeframe} search_mode=historical_only gap_candles={gap_candles} "
                f"calendar_gap_candles={calendar_gap_candles} "
                f"rule=match_end_before_query_start_minus_calendar_gap"
            )
        print(
            f"timeframe={timeframe} shape_filters=min_range_ratio={min_range_ratio} "
            f"max_range_ratio={max_range_ratio} min_vol_ratio={min_vol_ratio} "
            f"max_vol_ratio={max_vol_ratio} min_shape_score={min_shape_score} "
            f"min_final_score={min_final_score} dedup_gap_candles={window_size} "
            f"exclude_empty_plots={exclude_empty_plots}"
        )

        for final_score, idx, item, similarity in candidates:
            if exclude_empty_plots and not _has_future_candles(series_lookup, item, window_size):
                continue
            match_weight = _score_to_weight(final_score)
            line = (
                f"timeframe={timeframe} rank={printed + 1} idx={idx} score={final_score:.4f} "
                f"weight={match_weight:.4f} embed={similarity['embedding_score']:.4f} "
                f"shape={similarity['shape_score']:.4f} entity={item['entity_id']} "
                f"pattern_size={window_size} candles start_ts={item['start_ts']} ({_format_ts(item['start_ts'])}) "
                f"end_ts={item['end_ts']} ({_format_ts(item['end_ts'])})"
            )
            if horizons:
                outcome_parts = []
                for horizon in horizons:
                    value = _future_return(
                        prices_by_entity,
                        item["entity_id"],
                        int(item["start_idx"]),
                        window_size,
                        horizon,
                    )
                    if value is not None:
                        outcome_parts.append(f"+{horizon}={value:.2%}")
                        aggregate_outcomes[horizon].append((value, match_weight))
                line = f"{line} outcomes={' '.join(outcome_parts) if outcome_parts else 'n/a'}"
            print(line)
            matches_for_plot.append((printed + 1, item, float(final_score)))
            selected_scores.append(float(final_score))
            printed += 1
            if printed >= top_k:
                break

        analog_weights = np.array([_score_to_weight(score) for score in selected_scores], dtype=np.float64)
        effective_n = _effective_sample_size(analog_weights)
        confidence = _confidence_label(printed, effective_n)
        print(
            f"timeframe={timeframe} analog_forecast_summary matches={printed} "
            f"effective_matches={effective_n:.2f} confidence={confidence}"
        )

        if horizons:
            print(f"timeframe={timeframe} analog_forecast")
            for horizon in horizons:
                weighted_values = aggregate_outcomes.get(horizon, [])
                if not weighted_values:
                    print(f"timeframe={timeframe} +{horizon}: n/a")
                    continue
                arr = np.array([value for value, _ in weighted_values], dtype=np.float64)
                weights = np.array([weight for _, weight in weighted_values], dtype=np.float64)
                mean_value = _weighted_mean(arr, weights)
                median_value = _weighted_quantile(arr, weights, 0.5)
                low_value = _weighted_quantile(arr, weights, 0.25)
                high_value = _weighted_quantile(arr, weights, 0.75)
                positive_rate = _weighted_positive_rate(arr, weights)
                verdict = _verdict(mean_value, positive_rate)
                print(
                    f"timeframe={timeframe} +{horizon}: weighted_mean={mean_value:.2%} "
                    f"weighted_median={median_value:.2%} iqr=[{low_value:.2%},{high_value:.2%}] "
                    f"weighted_positive_rate={positive_rate:.0%} verdict={verdict} confidence={confidence}"
                )

        if plot_dir and matches_for_plot:
            timeframe_plot_dir = os.path.join(plot_dir, timeframe)
            output_path = _save_plot(
                plot_dir=timeframe_plot_dir,
                query_item=query_item,
                matches=matches_for_plot,
                series_lookup=series_lookup,
                window_size=window_size,
                horizons=horizons,
                timeframe=timeframe,
                query_series=query_series,
                include_relative_plots=include_relative_plots,
            )
            print(f"timeframe={timeframe} plot_saved={output_path}")
        elif plot_dir:
            print(f"timeframe={timeframe} plot_skipped=no_matches")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", dest="entity_id", default=None)
    search_cfg = SearchConfig()
    parser.add_argument("--top-k", type=int, default=search_cfg.top_k)
    parser.add_argument("--artifact-dir", default=search_cfg.artifact_dir)
    parser.add_argument("--horizons", default=search_cfg.horizons_raw)
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--only-past", action="store_true", default=search_cfg.only_past)
    parser.add_argument("--gap-candles", type=int, default=search_cfg.gap_candles)
    parser.add_argument("--calendar-gap-candles", type=int, default=search_cfg.calendar_gap_candles)
    parser.add_argument("--min-range-ratio", type=float, default=search_cfg.min_range_ratio)
    parser.add_argument("--max-range-ratio", type=float, default=search_cfg.max_range_ratio)
    parser.add_argument("--min-vol-ratio", type=float, default=search_cfg.min_vol_ratio)
    parser.add_argument("--max-vol-ratio", type=float, default=search_cfg.max_vol_ratio)
    parser.add_argument("--min-shape-score", type=float, default=search_cfg.min_shape_score)
    parser.add_argument("--min-final-score", type=float, default=search_cfg.min_final_score)
    parser.add_argument("--exclude-empty-plots", action="store_true")
    parser.add_argument("--include-relative-plots", action="store_true")
    args = parser.parse_args()
    horizons = [int(part) for part in args.horizons.split(",") if part.strip()]
    search(
        entity_id=args.entity_id,
        top_k=args.top_k,
        artifact_dir=args.artifact_dir,
        horizons=horizons,
        plot_dir=args.plot_dir,
        only_past=args.only_past,
        gap_candles=args.gap_candles,
        calendar_gap_candles=args.calendar_gap_candles,
        min_range_ratio=args.min_range_ratio,
        max_range_ratio=args.max_range_ratio,
        min_vol_ratio=args.min_vol_ratio,
        max_vol_ratio=args.max_vol_ratio,
        min_shape_score=args.min_shape_score,
        min_final_score=args.min_final_score,
        exclude_empty_plots=args.exclude_empty_plots,
        include_relative_plots=args.include_relative_plots,
    )


if __name__ == "__main__":
    main()
