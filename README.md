# AI Trading Pattern Search MVP

Проект для поиска похожих исторических паттернов и построения analog forecast по крипто-таймсериям из PostgreSQL.

Сейчас пайплайн устроен так:

- обучает отдельную retrieval-модель для `1d`, `4h`, `1h`
- `query` всегда берёт из live Binance API
- исторические кандидаты и их фактические продолжения берёт из PostgreSQL
- строит прогноз не direct-моделью, а по фактическому future найденных аналогов

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
```

Если `torch` не ставится автоматически, можно явно установить CPU-версию:

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
```

## Быстрый старт

```bash
cp .env.example .env
set -a
source .env
set +a
python3 -m src.train
python3 -m src.search --entity bitcoin --artifact-dir artifacts --top-k 5 --plot-dir artifacts/plots
```

## Основные env-переменные

### Источник данных

- `COINS_SOURCE_KIND`
- `COINS_TABLE`
- `COINS_TABLE_1D`
- `COINS_TABLE_4H`
- `COINS_TABLE_1H`
- `MODEL_TIMEFRAMES`
- `COINS_ENTITY_COL`
- `COINS_TIME_COL`
- `COINS_PRICE_COL`
- `COINS_AUX_COL`
- `COINS_SOURCE_SQL`

### Обучение retrieval-модели

- `MODEL_TYPE`
- `WINDOW_SIZE`
- `MIN_POINTS_PER_ENTITY`
- `BATCH_SIZE`
- `EPOCHS`
- `LEARNING_RATE`
- `LATENT_DIM`
- `HIDDEN_DIM`
- `CNN_CHANNELS`
- `CNN_KERNEL_SIZE`
- `ARTIFACT_DIR`
- `SEED`

### Препроцессинг

- `EPSILON_STD`
- `EPSILON_PRICE`
- `VOLATILITY_WINDOW`
- `ZERO_FILL_VALUE`
- `DEFAULT_VOLUME`
- `DEFAULT_MARKET_CAP`
- `DEFAULT_QUOTE_VOLUME`

### Search / Analog forecast

- `SEARCH_TOP_K`
- `ANALYZE_ALL_WORKERS`
- `SEARCH_ARTIFACT_DIR`
- `SEARCH_HORIZONS`
- `SEARCH_ONLY_PAST`
- `BINANCE_API_BASE_URL`
- `BINANCE_QUERY_QUOTE_ASSET`
- `SEARCH_GAP_CANDLES`
- `SEARCH_CALENDAR_GAP_CANDLES`
- `SEARCH_MIN_RANGE_RATIO`
- `SEARCH_MAX_RANGE_RATIO`
- `SEARCH_MIN_VOL_RATIO`
- `SEARCH_MAX_VOL_RATIO`
- `SEARCH_MIN_SHAPE_SCORE`
- `SEARCH_MIN_FINAL_SCORE`
- `ANALOG_FORECAST_MIN_MATCHES`
- `ANALOG_FORECAST_HIGH_CONFIDENCE_MATCHES`
- `ANALOG_FORECAST_WEIGHT_POWER`
- `BULLISH_MEAN_THRESHOLD`
- `BULLISH_POSITIVE_RATE_THRESHOLD`
- `BEARISH_MEAN_THRESHOLD`
- `BEARISH_POSITIVE_RATE_THRESHOLD`
- `SIMILARITY_EPSILON`
- `SEARCH_PRICE_EPSILON`
- `SEARCH_PATH_SCORE_SCALE`
- `FINAL_SCORE_EMBEDDING_WEIGHT`
- `FINAL_SCORE_SHAPE_WEIGHT`
- `FINAL_SCORE_PATH_WEIGHT`

### Графики

- `PLOT_COLORS`
- `PLOT_OUTER_WIDTH`
- `PLOT_PANEL_HEIGHT`
- `PLOT_PADDING_LEFT`
- `PLOT_PADDING_RIGHT`
- `PLOT_PADDING_TOP`
- `PLOT_PADDING_BOTTOM`
- `PLOT_LEGEND_Y_OFFSET`
- `PLOT_TITLE_LINE_HEIGHT`
- `PLOT_TITLE_MAX_CHARS`
- `PLOT_CANDLE_BODY_DIVISOR`
- `PLOT_CANDLE_BODY_MIN`
- `PLOT_CANDLE_BODY_MAX`
- `PLOT_CANDLE_BODY_MIN_HEIGHT`
- `PLOT_UP_CANDLE_COLOR`
- `PLOT_DOWN_CANDLE_COLOR`
- `PLOT_FORECAST_LINE_COLOR`
- `PLOT_FORECAST_BAND_COLOR`
- `PLOT_FORECAST_BAND_OPACITY`
- `PLOT_SHOW_QUERY_ACTUAL_FUTURE`

## Источник данных

Для обучения и поиска используется `binance`.

Ожидаемые поля в таблицах Binance:

- `coingecko_id` или `base_asset`
- `open_time`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `quote_asset_volume`

По умолчанию используются таблицы:

- `binance_klines_1d`
- `binance_klines_4h`
- `binance_klines_1h`

## Как обучить retrieval-модели

```bash
python3 -m src.train
```

Можно быстро пробовать разные типы моделей:

```bash
export MODEL_TYPE=mlp
python3 -m src.train
```

или

```bash
export MODEL_TYPE=cnn
export CNN_CHANNELS=64
export CNN_KERNEL_SIZE=5
python3 -m src.train
```

После обучения будут сохранены артефакты:

- `artifacts/1d/model.pt`
- `artifacts/1d/embeddings.npy`
- `artifacts/1d/windows.npy`
- `artifacts/1d/metadata.json`

и аналогично для `4h` и `1h`.

## Как пользоваться search

Базовый запуск:

```bash
python3 -m src.search --entity bitcoin --artifact-dir artifacts --top-k 5
```

Важно:

- `query` всегда берётся из актуальных последних свечей Binance API
- исторические совпадения и их future-хвосты берутся из PostgreSQL
- direct forecast в проекте не используется

Полезные варианты:

```bash
python3 -m src.search --entity bitcoin --artifact-dir artifacts --top-k 5 --horizons 5,10,20,64
```

```bash
python3 -m src.search --entity bitcoin --artifact-dir artifacts --top-k 5 --horizons 5,10,20,64 --plot-dir artifacts/plots
```

```bash
python3 -m src.search --entity bitcoin --artifact-dir artifacts --top-k 5 --horizons 5,10,20,64 --plot-dir artifacts/plots
```

```bash
python3 -m src.search --entity bitcoin --artifact-dir artifacts --top-k 5 --horizons 5,10,20,64 --plot-dir artifacts/plots --include-relative-plots
```

```bash
python3 -m src.search --entity bitcoin --artifact-dir artifacts --top-k 5 --horizons 5,10,20,64 --only-past --gap-candles 20 --calendar-gap-candles 20 --plot-dir artifacts/plots
```

## Что выдаёт analog forecast

Для каждого таймфрейма поиск печатает:

- top-k лучших исторических аналогов
- их фактические outcomes по горизонтам
- `analog_forecast_summary` с числом матчей, effective matches и confidence
- по умолчанию рисует original price панели в реальной цене
- по флагу `--include-relative-plots` дополнительно добавляет нормализованные панели
- weighted summary по каждому горизонту:
  - `weighted_mean`
  - `weighted_median`
  - `iqr`
  - `weighted_positive_rate`
  - `verdict`
  - `confidence`

Confidence интерпретируется так:

- `low` — хороших аналогов мало
- `medium` — аналогов достаточно, но база ещё не очень широкая
- `high` — аналогов много и weighted evidence устойчивее

## Как пользоваться analyze_all

Для всех монет:

```bash
python3 -m src.analyze_all --artifact-dir artifacts --top-k 5 --horizons 5,10,20,64 --plot-dir artifacts/plots_all --workers 4
```

Если нужны только исторические совпадения до query:

```bash
python3 -m src.analyze_all --artifact-dir artifacts --top-k 5 --horizons 5,10,20,64 --only-past --gap-candles 20 --calendar-gap-candles 20 --plot-dir artifacts/plots_all --workers 4
```
