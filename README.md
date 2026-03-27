# AI Trading Pattern Search MVP

Минимальный MVP для поиска похожих паттернов в крипто-таймсериях из PostgreSQL.

Сейчас проект работает так:

- читает 3 binance-таймфрейма в порядке `1d`, `4h`, `1h`
- обучает отдельную модель на каждый таймфрейм
- сохраняет артефакты по подпапкам, например `artifacts/1h`
- ищет похожие исторические окна отдельно внутри каждого таймфрейма
- Coingecko не используется как источник свечей для обучения; оттуда берётся только список топ-токенов

## Что нужно для запуска

- Python 3.11+
- PostgreSQL с таблицей таймсерий
- зависимости из `requirements.txt` и `requirements-train.txt`

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

## Настройка подключения к БД

Минимальные переменные окружения:

```bash
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=coins
export PGUSER=admin
export PGPASSWORD=admin
```

Для удобства можно взять шаблон из `.env.example` и загрузить его перед запуском:

```bash
cp .env.example .env
set -a
source .env
set +a
```

По умолчанию, если ничего не экспортировать дополнительно, проект будет пытаться читать:

- `COINS_SOURCE_KIND=binance`
- `MODEL_TIMEFRAMES=1d,4h,1h`
- `COINS_TABLE_1D=binance_klines_1d`
- `COINS_TABLE_4H=binance_klines_4h`
- `COINS_TABLE_1H=binance_klines_1h`

## Основные env-переменные

Сейчас основные константы проекта вынесены в `env`.

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

### Обучение

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

### Численные константы препроцессинга

- `EPSILON_STD`
- `EPSILON_PRICE`
- `VOLATILITY_WINDOW`
- `ZERO_FILL_VALUE`
- `DEFAULT_VOLUME`
- `DEFAULT_MARKET_CAP`
- `DEFAULT_QUOTE_VOLUME`

### Поиск

- `SEARCH_TOP_K`
- `SEARCH_ARTIFACT_DIR`
- `SEARCH_HORIZONS`
- `SEARCH_ONLY_PAST`
- `SEARCH_GAP_CANDLES`
- `SEARCH_CALENDAR_GAP_CANDLES`
- `SEARCH_MIN_RANGE_RATIO`
- `SEARCH_MAX_RANGE_RATIO`
- `SEARCH_MIN_VOL_RATIO`
- `SEARCH_MAX_VOL_RATIO`
- `SEARCH_MIN_SHAPE_SCORE`
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
- `PLOT_CANDLE_BODY_DIVISOR`
- `PLOT_CANDLE_BODY_MIN`
- `PLOT_CANDLE_BODY_MAX`
- `PLOT_CANDLE_BODY_MIN_HEIGHT`
- `PLOT_UP_CANDLE_COLOR`
- `PLOT_DOWN_CANDLE_COLOR`

## Источник данных

Для обучения и поиска используется `binance`.

Coingecko в текущем процессе не является источником таймсерий для модели. Он нужен только как внешний источник списка топ-токенов, если вы используете его в отдельном ingestion-процессе.

### `binance`

Использует 3 таблицы по таймфреймам по умолчанию и обрабатывает их в порядке убывания таймфрейма:

- `binance_klines_1d`
- `binance_klines_4h`
- `binance_klines_1h`

Ожидаемые поля:

- `coingecko_id` или `base_asset`
- `open_time`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `quote_asset_volume`

Запуск по умолчанию:

```bash
export COINS_SOURCE_KIND=binance
```

Если таблицы называются иначе, задайте их явно:

```bash
export COINS_SOURCE_KIND=binance
export MODEL_TIMEFRAMES=1d,4h,1h
export COINS_TABLE_1D=binance_klines_1d
export COINS_TABLE_4H=binance_klines_4h
export COINS_TABLE_1H=binance_klines_1h
```

### `custom`

Если у вас своя таблица:

```bash
export COINS_SOURCE_KIND=custom
export COINS_TABLE=your_table
export COINS_ENTITY_COL=coin_id
export COINS_TIME_COL=ts
export COINS_PRICE_COL=price
export COINS_AUX_COL=total_volume
```

Либо можно передать SQL вручную:

```bash
export COINS_SOURCE_SQL="
SELECT
  coin_id AS entity_id,
  EXTRACT(EPOCH FROM ts) AS ts,
  price,
  total_volume AS volume,
  market_cap
FROM your_table
WHERE price IS NOT NULL
ORDER BY coin_id, ts
"
```

## Как обучить модели

Обучение выполняется одной командой:

```bash
python3 -m src.train
```

Можно быстро пробовать разные типы моделей без смены кода:

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

После обучения будут сохранены артефакты в папки:

- `artifacts/1d/`
- `artifacts/4h/`
- `artifacts/1h/`

Внутри каждой:

- `model.pt`
- `embeddings.npy`
- `windows.npy`
- `metadata.json`

## Как пользоваться скриптом поиска

Базовый запуск:

```bash
python3 -m src.search --entity bitcoin --artifact-dir artifacts --top-k 5
```

Что делает команда:

- берёт последнее сохранённое окно указанной монеты отдельно для каждого таймфрейма
- ищет самые похожие окна среди всех исторических окон этого таймфрейма
- выводит отдельный top-k для `1d`, `4h`, `1h`

### Полезные параметры

Поиск с анализом того, что происходило после найденных паттернов:

```bash
python3 -m src.search --entity bitcoin --artifact-dir artifacts --top-k 5 --horizons 5,10,20
```

Сохранить график query и найденных совпадений:

```bash
python3 -m src.search --entity bitcoin --artifact-dir artifacts --top-k 5 --horizons 5,10,20 --plot-dir artifacts/plots
```

Если совпадений нет, график для этого таймфрейма не сохраняется.

Искать только в прошлом относительно query:

```bash
python3 -m src.search \
  --entity bitcoin \
  --artifact-dir artifacts \
  --top-k 5 \
  --horizons 5,10,20 \
  --only-past \
  --gap-candles 20 \
  --calendar-gap-candles 20 \
  --plot-dir artifacts/plots
```

### Основные флаги `src.search`

- `--entity` - монета/сущность, например `bitcoin`
- `--top-k` - сколько лучших совпадений вернуть
- `--artifact-dir` - папка с обученной моделью, по умолчанию `artifacts`
- `--horizons` - горизонты в свечах, например `5,10,20`
- `--plot-dir` - куда сохранить SVG-график
- `--only-past` - брать только паттерны из прошлого
- `--gap-candles` - зазор по свечам для той же монеты
- `--calendar-gap-candles` - календарный зазор для всех монет
- `--min-range-ratio` - фильтр по схожести амплитуды
- `--max-range-ratio` - верхняя граница амплитуды
- `--min-vol-ratio` - фильтр по схожести волатильности
- `--max-vol-ratio` - верхняя граница волатильности
- `--min-shape-score` - минимальная схожесть формы

## Как запускать по разным таймфреймам

По умолчанию `train` уже обучает все 3 модели за один запуск.

Если хотите обучать не все таймфреймы, а только часть:

```bash
export MODEL_TIMEFRAMES=1d,4h
python3 -m src.train
```

## Что именно учит модель

Модель работает не с абсолютной ценой, а в основном с относительной динамикой внутри окна.

Используются признаки:

- `relative_price = price / price[0] - 1`
- `log_return`
- локальная волатильность
- динамика объёма
- `market_cap_log`
- `quote_volume_log`

Это значит, что движения `1 -> 2` и `5 -> 10` для модели близки по форме, если относительная динамика одинаковая.

## Что хранится в эмбеддингах

После обучения каждое окно превращается в latent-вектор размерности `LATENT_DIM` (по умолчанию `32`).

Файлы:

- `embeddings.npy` - массив эмбеддингов всех окон
- `metadata.json` - соответствие индекса эмбеддинга монете и временному диапазону окна
- `windows.npy` - нормализованные окна, которые подавались модели

## Формат ожидаемого датасета

После приведения SQL проект ожидает как минимум:

- `entity_id`
- `ts`
- `price`

Опционально:

- `volume`
- `market_cap`
- `quote_volume`
- `open`
- `high`
- `low`
- `close`

## Типовой сценарий работы

### Обучить 3 модели

```bash
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=coins
export PGUSER=admin
export PGPASSWORD=admin

export COINS_SOURCE_KIND=binance
export MODEL_TIMEFRAMES=1d,4h,1h
export COINS_TABLE_1D=binance_klines_1d
export COINS_TABLE_4H=binance_klines_4h
export COINS_TABLE_1H=binance_klines_1h
export ARTIFACT_DIR=artifacts

python3 -m src.train
```

### Найти похожие окна для bitcoin

```bash
python3 -m src.search --entity bitcoin --artifact-dir artifacts --top-k 5 --horizons 5,10,20 --plot-dir artifacts/plots
```

### Прогнать поиск по всем монетам

```bash
python3 -m src.analyze_all --artifact-dir artifacts --top-k 5 --horizons 5,10,20 --plot-dir artifacts/plots_all
```

В массовом режиме графики сохраняются только для тех монет, у которых реально нашлись совпадения.

## Ограничение текущей версии

В текущем состоянии репозитория:

- поиск выполняется отдельно внутри каждого таймфрейма
- итоговая агрегация между `1d`, `4h`, `1h` пока остаётся на уровне вывода, а не отдельного ансамблевого сигнала
