from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_TIMEFRAMES = ("1d", "4h", "1h")


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _get_env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [part.strip() for part in value.split(",") if part.strip()] or default


@dataclass(frozen=True)
class DbConfig:
    host: str = os.getenv("PGHOST", "localhost")
    port: int = _get_env_int("PGPORT", 5432)
    dbname: str = os.getenv("PGDATABASE", "coins")
    user: str = os.getenv("PGUSER", "admin")
    password: str = os.getenv("PGPASSWORD", "admin")


@dataclass(frozen=True)
class SourceConfig:
    source_kind: str = os.getenv("COINS_SOURCE_KIND", "binance")
    table: str = os.getenv("COINS_TABLE", "binance_klines_1h")
    entity_col: str = os.getenv("COINS_ENTITY_COL", "token")
    time_col: str = os.getenv("COINS_TIME_COL", "timestamp")
    price_col: str = os.getenv("COINS_PRICE_COL", "price")
    aux_col: str = os.getenv("COINS_AUX_COL", "holders")
    custom_sql: str | None = os.getenv("COINS_SOURCE_SQL")
    timeframe: str | None = os.getenv("COINS_TIMEFRAME")


@dataclass(frozen=True)
class TrainConfig:
    model_type: str = os.getenv("MODEL_TYPE", "mlp")
    window_size: int = _get_env_int("WINDOW_SIZE", 64)
    min_points_per_entity: int = _get_env_int("MIN_POINTS_PER_ENTITY", 96)
    batch_size: int = _get_env_int("BATCH_SIZE", 256)
    epochs: int = _get_env_int("EPOCHS", 20)
    learning_rate: float = _get_env_float("LEARNING_RATE", 1e-3)
    latent_dim: int = _get_env_int("LATENT_DIM", 32)
    hidden_dim: int = _get_env_int("HIDDEN_DIM", 128)
    cnn_channels: int = _get_env_int("CNN_CHANNELS", 64)
    cnn_kernel_size: int = _get_env_int("CNN_KERNEL_SIZE", 5)
    artifact_dir: str = os.getenv("ARTIFACT_DIR", "artifacts")
    seed: int = _get_env_int("SEED", 42)


@dataclass(frozen=True)
class DataConfig:
    epsilon_std: float = _get_env_float("EPSILON_STD", 1e-8)
    epsilon_price: float = _get_env_float("EPSILON_PRICE", 1e-12)
    volatility_window: int = _get_env_int("VOLATILITY_WINDOW", 8)
    zero_fill_value: float = _get_env_float("ZERO_FILL_VALUE", 0.0)
    default_volume: float = _get_env_float("DEFAULT_VOLUME", 0.0)
    default_market_cap: float = _get_env_float("DEFAULT_MARKET_CAP", 0.0)
    default_quote_volume: float = _get_env_float("DEFAULT_QUOTE_VOLUME", 0.0)


@dataclass(frozen=True)
class SearchConfig:
    top_k: int = _get_env_int("SEARCH_TOP_K", 5)
    analyze_all_workers: int = _get_env_int("ANALYZE_ALL_WORKERS", 6)
    artifact_dir: str = os.getenv("SEARCH_ARTIFACT_DIR", "artifacts")
    horizons_raw: str = os.getenv("SEARCH_HORIZONS", "5,10,20")
    only_past: bool = os.getenv("SEARCH_ONLY_PAST", "").lower() in {"1", "true", "yes"}
    gap_candles: int = _get_env_int("SEARCH_GAP_CANDLES", 0)
    calendar_gap_candles: int = _get_env_int("SEARCH_CALENDAR_GAP_CANDLES", 0)
    min_range_ratio: float = _get_env_float("SEARCH_MIN_RANGE_RATIO", 0.35)
    max_range_ratio: float = _get_env_float("SEARCH_MAX_RANGE_RATIO", 3.0)
    min_vol_ratio: float = _get_env_float("SEARCH_MIN_VOL_RATIO", 0.35)
    max_vol_ratio: float = _get_env_float("SEARCH_MAX_VOL_RATIO", 3.0)
    min_shape_score: float = _get_env_float("SEARCH_MIN_SHAPE_SCORE", 0.55)
    min_final_score: float = _get_env_float("SEARCH_MIN_FINAL_SCORE", 0.9)
    bullish_mean_threshold: float = _get_env_float("BULLISH_MEAN_THRESHOLD", 0.02)
    bullish_positive_rate_threshold: float = _get_env_float("BULLISH_POSITIVE_RATE_THRESHOLD", 0.6)
    bearish_mean_threshold: float = _get_env_float("BEARISH_MEAN_THRESHOLD", -0.02)
    bearish_positive_rate_threshold: float = _get_env_float("BEARISH_POSITIVE_RATE_THRESHOLD", 0.4)
    epsilon_similarity: float = _get_env_float("SIMILARITY_EPSILON", 1e-8)
    epsilon_price: float = _get_env_float("SEARCH_PRICE_EPSILON", 1e-12)
    path_score_scale: float = _get_env_float("SEARCH_PATH_SCORE_SCALE", 25.0)
    final_score_embedding_weight: float = _get_env_float("FINAL_SCORE_EMBEDDING_WEIGHT", 0.55)
    final_score_shape_weight: float = _get_env_float("FINAL_SCORE_SHAPE_WEIGHT", 0.30)
    final_score_path_weight: float = _get_env_float("FINAL_SCORE_PATH_WEIGHT", 0.15)


@dataclass(frozen=True)
class PlotConfig:
    colors: list[str]
    outer_width: int = _get_env_int("PLOT_OUTER_WIDTH", 1200)
    panel_height: int = _get_env_int("PLOT_PANEL_HEIGHT", 240)
    padding_left: int = _get_env_int("PLOT_PADDING_LEFT", 70)
    padding_right: int = _get_env_int("PLOT_PADDING_RIGHT", 30)
    padding_top: int = _get_env_int("PLOT_PADDING_TOP", 45)
    padding_bottom: int = _get_env_int("PLOT_PADDING_BOTTOM", 40)
    legend_y_offset: int = _get_env_int("PLOT_LEGEND_Y_OFFSET", 40)
    title_line_height: int = _get_env_int("PLOT_TITLE_LINE_HEIGHT", 18)
    title_max_chars: int = _get_env_int("PLOT_TITLE_MAX_CHARS", 72)
    candle_body_divisor: float = _get_env_float("PLOT_CANDLE_BODY_DIVISOR", 1.8)
    candle_body_min: float = _get_env_float("PLOT_CANDLE_BODY_MIN", 3.0)
    candle_body_max: float = _get_env_float("PLOT_CANDLE_BODY_MAX", 10.0)
    candle_body_min_height: float = _get_env_float("PLOT_CANDLE_BODY_MIN_HEIGHT", 1.2)
    up_candle_color: str = os.getenv("PLOT_UP_CANDLE_COLOR", "#16a34a")
    down_candle_color: str = os.getenv("PLOT_DOWN_CANDLE_COLOR", "#dc2626")
    forecast_line_color: str = os.getenv("PLOT_FORECAST_LINE_COLOR", "#7c3aed")
    forecast_band_color: str = os.getenv("PLOT_FORECAST_BAND_COLOR", "#c4b5fd")
    forecast_band_opacity: float = _get_env_float("PLOT_FORECAST_BAND_OPACITY", 0.28)
    show_query_actual_future: bool = os.getenv("PLOT_SHOW_QUERY_ACTUAL_FUTURE", "").lower() in {"1", "true", "yes"}

    def __init__(self) -> None:
        object.__setattr__(
            self,
            "colors",
            _get_env_list(
                "PLOT_COLORS",
                ["#1d4ed8", "#ea580c", "#16a34a", "#dc2626", "#7c3aed", "#0891b2"],
            ),
        )
        object.__setattr__(self, "outer_width", _get_env_int("PLOT_OUTER_WIDTH", 1200))
        object.__setattr__(self, "panel_height", _get_env_int("PLOT_PANEL_HEIGHT", 240))
        object.__setattr__(self, "padding_left", _get_env_int("PLOT_PADDING_LEFT", 70))
        object.__setattr__(self, "padding_right", _get_env_int("PLOT_PADDING_RIGHT", 30))
        object.__setattr__(self, "padding_top", _get_env_int("PLOT_PADDING_TOP", 45))
        object.__setattr__(self, "padding_bottom", _get_env_int("PLOT_PADDING_BOTTOM", 40))
        object.__setattr__(self, "legend_y_offset", _get_env_int("PLOT_LEGEND_Y_OFFSET", 40))
        object.__setattr__(self, "title_line_height", _get_env_int("PLOT_TITLE_LINE_HEIGHT", 18))
        object.__setattr__(self, "title_max_chars", _get_env_int("PLOT_TITLE_MAX_CHARS", 72))
        object.__setattr__(self, "candle_body_divisor", _get_env_float("PLOT_CANDLE_BODY_DIVISOR", 1.8))
        object.__setattr__(self, "candle_body_min", _get_env_float("PLOT_CANDLE_BODY_MIN", 3.0))
        object.__setattr__(self, "candle_body_max", _get_env_float("PLOT_CANDLE_BODY_MAX", 10.0))
        object.__setattr__(self, "candle_body_min_height", _get_env_float("PLOT_CANDLE_BODY_MIN_HEIGHT", 1.2))
        object.__setattr__(self, "up_candle_color", os.getenv("PLOT_UP_CANDLE_COLOR", "#16a34a"))
        object.__setattr__(self, "down_candle_color", os.getenv("PLOT_DOWN_CANDLE_COLOR", "#dc2626"))
        object.__setattr__(self, "forecast_line_color", os.getenv("PLOT_FORECAST_LINE_COLOR", "#7c3aed"))
        object.__setattr__(self, "forecast_band_color", os.getenv("PLOT_FORECAST_BAND_COLOR", "#c4b5fd"))
        object.__setattr__(self, "forecast_band_opacity", _get_env_float("PLOT_FORECAST_BAND_OPACITY", 0.28))
        object.__setattr__(
            self,
            "show_query_actual_future",
            os.getenv("PLOT_SHOW_QUERY_ACTUAL_FUTURE", "").lower() in {"1", "true", "yes"},
        )


def get_model_timeframes() -> list[str]:
    return _get_env_list("MODEL_TIMEFRAMES", list(DEFAULT_TIMEFRAMES))


def get_table_for_timeframe(timeframe: str, source_kind: str, fallback_table: str | None = None) -> str:
    explicit = os.getenv(f"COINS_TABLE_{timeframe.upper()}")
    if explicit:
        return explicit

    if source_kind == "binance":
        defaults = {
            "1h": "binance_klines_1h",
            "4h": "binance_klines_4h",
            "1d": "binance_klines_1d",
        }
        return defaults.get(timeframe, fallback_table or f"binance_klines_{timeframe}")

    if fallback_table:
        return fallback_table
    return timeframe
