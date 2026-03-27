from __future__ import annotations

import argparse
import json
import os
from html import escape
from datetime import UTC, datetime

import numpy as np

from .config import DEFAULT_TIMEFRAMES, DbConfig, PlotConfig, SearchConfig, SourceConfig
from .data import fetch_series, group_by_entity


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


def _load_price_lookup(source_cfg_payload: dict) -> dict[str, np.ndarray]:
    source_cfg = SourceConfig(**source_cfg_payload)
    rows = fetch_series(DbConfig(), source_cfg)
    grouped = group_by_entity(rows)
    return {
        entity_id: np.array([float(row["price"]) for row in entity_rows], dtype=np.float64)
        for entity_id, entity_rows in grouped.items()
    }


def _load_series_lookup(source_cfg_payload: dict) -> dict[str, dict[str, np.ndarray]]:
    source_cfg = SourceConfig(**source_cfg_payload)
    rows = fetch_series(DbConfig(), source_cfg)
    grouped = group_by_entity(rows)
    lookup: dict[str, dict[str, np.ndarray]] = {}
    for entity_id, entity_rows in grouped.items():
        lookup[entity_id] = {
            "price": np.array([float(row["price"]) for row in entity_rows], dtype=np.float64),
            "ts": np.array([float(row["ts"]) for row in entity_rows], dtype=np.float64),
            "open": np.array([float(row.get("open", row["price"])) for row in entity_rows], dtype=np.float64),
            "high": np.array([float(row.get("high", row["price"])) for row in entity_rows], dtype=np.float64),
            "low": np.array([float(row.get("low", row["price"])) for row in entity_rows], dtype=np.float64),
            "close": np.array([float(row.get("close", row["price"])) for row in entity_rows], dtype=np.float64),
        }
    return lookup


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


def _build_plot_series(
    series_lookup: dict[str, dict[str, np.ndarray]],
    item: dict,
    window_size: int,
    max_horizon: int,
) -> dict[str, np.ndarray | int]:
    search_cfg = SearchConfig()
    entity_series = series_lookup[item["entity_id"]]
    prices = entity_series["price"]
    start_idx = int(item["start_idx"])
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


def _save_plot(
    plot_dir: str,
    query_item: dict,
    matches: list[tuple[int, dict, float]],
    series_lookup: dict[str, dict[str, np.ndarray]],
    window_size: int,
    horizons: list[int],
    timeframe: str,
) -> str:
    plot_cfg = PlotConfig()
    os.makedirs(plot_dir, exist_ok=True)
    max_horizon = max(horizons) if horizons else 0
    query_plot = _build_plot_series(series_lookup, query_item, window_size, max_horizon)
    query_x = query_plot["x"]
    query_close = query_plot["close"]
    split_idx = int(query_plot["split_idx"])
    panels: list[dict] = [
        {
            "title": (
                f"Query: {query_item['entity_id']} | "
                f"{_format_ts(query_item['start_ts'])} -> {_format_ts(query_item['end_ts'])}"
            ),
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
                if len(query_x) > split_idx
                else None,
            ],
            "split_idx": split_idx,
        }
    ]

    colors = plot_cfg.colors
    for plot_idx, (rank_idx, item, score) in enumerate(matches, start=1):
        match_plot = _build_plot_series(series_lookup, item, window_size, max_horizon)
        match_x = match_plot["x"]
        match_close = match_plot["close"]
        split_idx = int(match_plot["split_idx"])
        panels.append(
            {
                "title": (
                    f"rank={rank_idx} score={score:.4f} entity={item['entity_id']} | "
                    f"{_format_ts(item['start_ts'])} -> {_format_ts(item['end_ts'])}"
                ),
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
                        "color": colors[(plot_idx - 1) % len(colors)],
                        "label": f"match #{rank_idx} candles",
                        "kind": "candles",
                    },
                    {
                        "x": match_x[split_idx - 1 :],
                        "y": match_close[split_idx - 1 :],
                        "color": colors[(plot_idx - 1) % len(colors)],
                        "label": "match future",
                        "dashed": True,
                        "kind": "line",
                    }
                    if len(match_x) > split_idx
                    else None,
                ],
                "split_idx": split_idx,
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

    def map_y(y_val: float, top: int) -> float:
        plot_height = panel_height - padding_top - padding_bottom
        return top + padding_top + (
            1.0 - ((float(y_val) - global_y_min) / (global_y_max - global_y_min))
        ) * plot_height

    def to_polyline(xs: np.ndarray, ys: np.ndarray, width: int, top: int) -> str:
        plot_width = width - padding_left - padding_right
        x_max = max(len(xs) - 1, 1)
        points = []
        for x_val, y_val in zip(xs, ys):
            px = padding_left + (float(x_val) / x_max) * plot_width
            py = map_y(float(y_val), top)
            points.append(f"{px:.2f},{py:.2f}")
        return " ".join(points)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{outer_width}" height="{total_height}" viewBox="0 0 {outer_width} {total_height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827} .small{font-size:12px} .title{font-size:16px;font-weight:bold} .axis{stroke:#9ca3af;stroke-width:1} .grid{stroke:#e5e7eb;stroke-width:1} .split{stroke:#6b7280;stroke-width:1;stroke-dasharray:4 4}</style>',
    ]

    all_y = np.concatenate(
        [
            values
            for panel in panels
            for s in panel["series"]
            for values in (
                [s["y"]]
                if s.get("kind") == "line"
                else [s["open"], s["high"], s["low"], s["close"]]
            )
        ]
    )
    global_y_min = float(np.min(all_y))
    global_y_max = float(np.max(all_y))
    if abs(global_y_max - global_y_min) < 1e-9:
        global_y_max = global_y_min + 1e-6

    for panel_idx, panel in enumerate(panels):
        top = panel_idx * panel_height
        plot_width = outer_width - padding_left - padding_right
        plot_height = panel_height - padding_top - padding_bottom
        title_lines = wrap_title(panel["title"])
        title_y = top + 22
        tspans = "".join(
            f'<tspan x="{padding_left}" y="{title_y + idx * plot_cfg.title_line_height}">{escape(line)}</tspan>'
            for idx, line in enumerate(title_lines)
        )
        svg_parts.append(f'<text class="title">{tspans}</text>')

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = top + padding_top + frac * plot_height
            svg_parts.append(
                f'<line x1="{padding_left}" y1="{y:.2f}" x2="{padding_left + plot_width}" y2="{y:.2f}" class="grid"/>'
            )
            value = global_y_max - frac * (global_y_max - global_y_min)
            svg_parts.append(f'<text x="8" y="{y + 4:.2f}" class="small">{value:.1%}</text>')

        svg_parts.append(
            f'<line x1="{padding_left}" y1="{top + padding_top}" x2="{padding_left}" y2="{top + padding_top + plot_height}" class="axis"/>'
        )
        svg_parts.append(
            f'<line x1="{padding_left}" y1="{top + padding_top + plot_height}" x2="{padding_left + plot_width}" y2="{top + padding_top + plot_height}" class="axis"/>'
        )

        split_x = padding_left + ((panel["split_idx"] - 1) / max(window_size - 1, 1)) * plot_width
        svg_parts.append(
            f'<line x1="{split_x:.2f}" y1="{top + padding_top}" x2="{split_x:.2f}" y2="{top + padding_top + plot_height}" class="split"/>'
        )

        legend_x = padding_left
        for series in panel["series"]:
            dash = ' stroke-dasharray="6 4"' if series.get("dashed") else ""
            if series.get("kind") == "candles":
                plot_width = outer_width - padding_left - padding_right
                x_max = max(len(series["x"]) - 1, 1)
                candle_body = max(
                    min(plot_width / max(window_size * plot_cfg.candle_body_divisor, 1), plot_cfg.candle_body_max),
                    plot_cfg.candle_body_min,
                )
                for x_val, open_val, high_val, low_val, close_val in zip(
                    series["x"], series["open"], series["high"], series["low"], series["close"]
                ):
                    px = padding_left + (float(x_val) / x_max) * plot_width
                    wick_y1 = map_y(float(high_val), top)
                    wick_y2 = map_y(float(low_val), top)
                    open_y = map_y(float(open_val), top)
                    close_y = map_y(float(close_val), top)
                    body_y = min(open_y, close_y)
                    body_h = max(abs(close_y - open_y), plot_cfg.candle_body_min_height)
                    body_color = plot_cfg.up_candle_color if close_val >= open_val else plot_cfg.down_candle_color
                    svg_parts.append(
                        f'<line x1="{px:.2f}" y1="{wick_y1:.2f}" x2="{px:.2f}" y2="{wick_y2:.2f}" '
                        f'stroke="{body_color}" stroke-width="1.2"/>'
                    )
                    svg_parts.append(
                        f'<rect x="{px - candle_body / 2:.2f}" y="{body_y:.2f}" width="{candle_body:.2f}" '
                        f'height="{body_h:.2f}" fill="{body_color}" opacity="0.85"/>'
                    )
            else:
                polyline = to_polyline(series["x"], series["y"], outer_width, top)
                svg_parts.append(
                    f'<polyline fill="none" stroke="{series["color"]}" stroke-width="2"{dash} points="{polyline}"/>'
                )
            svg_parts.append(
                f'<line x1="{legend_x}" y1="{top + legend_y_offset}" x2="{legend_x + 18}" y2="{top + legend_y_offset}" '
                f'stroke="{series["color"]}" stroke-width="2"{dash}/>'
            )
            svg_parts.append(
                f'<text x="{legend_x + 24}" y="{top + legend_y_offset + 4}" class="small">{escape(series["label"])}</text>'
            )
            legend_x += 180

        svg_parts.append(
            f'<text x="{padding_left}" y="{top + panel_height - 10}" class="small">Candles from window start</text>'
        )

    svg_parts.append("</svg>")
    output_path = os.path.join(plot_dir, f"pattern_search_{timeframe}_{query_item['entity_id']}.svg")
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

        if entity_id:
            candidate_indexes = [idx for idx, item in enumerate(windows) if item["entity_id"] == entity_id]
            if not candidate_indexes:
                print(f"timeframe={timeframe} entity={entity_id} status=not_found")
                continue
            query_idx = candidate_indexes[-1]
        else:
            query_idx = len(windows) - 1

        query_item = windows[query_idx]
        query_start_ts = float(query_item["start_ts"])
        query_end_ts = float(query_item["end_ts"])
        candle_seconds = 0.0
        if window_size > 1:
            candle_seconds = max((query_end_ts - query_start_ts) / (window_size - 1), 0.0)
        global_cutoff_ts = query_start_ts - (calendar_gap_candles * candle_seconds)

        query_vector = embeddings[query_idx]
        sims = cosine_similarity(embeddings, query_vector)
        query_close = _window_close_series(series_lookup, query_item, window_size)

        candidates: list[tuple[float, int, dict, dict[str, float]]] = []
        for idx in np.argsort(-sims):
            if idx == query_idx:
                continue
            item = windows[int(idx)]
            if only_past:
                match_end_ts = float(item["end_ts"])
                match_end_idx = int(item["start_idx"]) + window_size - 1
                query_start_idx = int(query_item["start_idx"])
                same_entity = item["entity_id"] == query_item["entity_id"]
                if match_end_ts >= global_cutoff_ts:
                    continue
                if same_entity and match_end_idx >= max(0, query_start_idx - gap_candles):
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
        printed = 0
        matches_for_plot: list[tuple[int, dict, float]] = []
        aggregate_outcomes: dict[int, list[float]] = {horizon: [] for horizon in horizons}
        print(f"timeframe={timeframe} artifact_dir={resolved_artifact_dir}")
        print(
            f"timeframe={timeframe} query_idx={query_idx} entity={query_item['entity_id']} "
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
            f"min_final_score={min_final_score}"
        )
        if horizons:
            query_outcomes = []
            for horizon in horizons:
                value = _future_return(
                    prices_by_entity,
                    query_item["entity_id"],
                    int(query_item["start_idx"]),
                    window_size,
                    horizon,
                )
                if value is not None:
                    query_outcomes.append(f"+{horizon}={value:.2%}")
            if query_outcomes:
                print(f"timeframe={timeframe} query_outcomes {' '.join(query_outcomes)}")

        for final_score, idx, item, similarity in candidates[:top_k]:
            line = (
                f"timeframe={timeframe} rank={printed + 1} idx={idx} score={final_score:.4f} "
                f"embed={similarity['embedding_score']:.4f} shape={similarity['shape_score']:.4f} "
                f"entity={item['entity_id']} pattern_size={window_size} candles "
                f"start_ts={item['start_ts']} ({_format_ts(item['start_ts'])}) "
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
                        aggregate_outcomes[horizon].append(value)
                line = f"{line} outcomes={' '.join(outcome_parts) if outcome_parts else 'n/a'}"
            print(line)
            matches_for_plot.append((printed + 1, item, float(final_score)))
            printed += 1

        if horizons:
            print(f"timeframe={timeframe} aggregate_signal")
            for horizon in horizons:
                values = aggregate_outcomes.get(horizon, [])
                if not values:
                    print(f"timeframe={timeframe} +{horizon}: n/a")
                    continue
                arr = np.array(values, dtype=np.float64)
                mean_value = float(arr.mean())
                median_value = float(np.median(arr))
                positive_rate = float((arr > 0).mean())
                verdict = _verdict(mean_value, positive_rate)
                print(
                    f"timeframe={timeframe} +{horizon}: mean={mean_value:.2%} median={median_value:.2%} "
                    f"positive_rate={positive_rate:.0%} verdict={verdict}"
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
    )


if __name__ == "__main__":
    main()
