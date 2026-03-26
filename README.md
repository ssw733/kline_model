# AI Trading Pattern Search MVP

Минимальный MVP для поиска похожих паттернов в крипто-таймсериях из PostgreSQL.

Что умеет:

- Загружает таймсерию из PostgreSQL
- Строит скользящие окна по цене и дополнительным признакам
- Нормализует цену по паттерну, а не по абсолютному уровню
- Обучает автоэнкодер на PyTorch
- Ищет top-k похожих исторических окон по эмбеддингам

## Быстрый старт

1. Установить зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
```

2. Настроить подключение к БД и выбрать источник данных:

```bash
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=coins
export PGUSER=admin
export PGPASSWORD=admin

export COINS_SOURCE_KIND=coingecko
```

Поддерживаются готовые режимы:

- `COINS_SOURCE_KIND=coingecko` для таблицы `coingecko_market_data`
- `COINS_SOURCE_KIND=binance` для таблицы `binance_klines`

Если хотите взять свою таблицу, используйте `COINS_SOURCE_KIND=custom` и настройте колонки:

```bash
export COINS_SOURCE_KIND=custom
export COINS_TABLE=your_table
export COINS_ENTITY_COL=coin_id
export COINS_TIME_COL=ts
export COINS_PRICE_COL=price
export COINS_AUX_COL=total_volume
```

Если структура еще сложнее, можно передать SQL вручную:

```bash
export COINS_SOURCE_SQL="
SELECT
  coin_id AS entity_id,
  EXTRACT(EPOCH FROM ts) AS ts,
  price,
  total_volume AS volume,
  market_cap
FROM coingecko_market_data
WHERE price IS NOT NULL
ORDER BY coin_id, ts
"
```

3. Обучить модель:

```bash
python3 -m src.train
```

4. Найти похожие окна:

```bash
python3 -m src.search --entity YOUR_TOKEN --top-k 5
```

С оценкой того, что происходило после похожих паттернов:

```bash
PGHOST=127.0.0.1 python3 -m src.search --entity bitcoin --top-k 5 --horizons 5,10,20
```

С сохранением графика исходного окна и найденных совпадений:

```bash
PGHOST=127.0.0.1 python3 -m src.search --entity bitcoin --top-k 5 --horizons 5,10,20 --plot-dir artifacts/plots
```

Для прогнозирования лучше использовать только исторические совпадения:

```bash
PGHOST=127.0.0.1 python3 -m src.search --entity bitcoin --top-k 5 --horizons 5,10,20 --only-past --gap-candles 20 --calendar-gap-candles 20 --plot-dir artifacts/plots
```

## Формат источника

Скрипт ожидает, что итоговый датасет после SQL содержит колонки:

- `entity_id`
- `ts`
- `price`
- `volume` (опционально)
- `market_cap` (опционально)
- `quote_volume` (опционально)

Если вы используете режимы `coingecko`, `binance` или `custom`, преобразование в эти имена делается автоматически.

## Нормализация цены

Для поиска именно похожих паттернов используется комбинация:

- `log_return = log(price_t / price_t-1)`
- `relative_price = price / price[0] - 1`
- `log1p(volume)` и его производные
- `window z-score` для каждого окна отдельно

Это позволяет сети учить форму движения, а не абсолютный уровень цены.

## Что было потом

`src.search` умеет считать future return после каждого найденного совпадения.
Если указать `--horizons 5,10,20`, в выводе появятся доходности через 5, 10 и 20 свечей после конца окна.

Если дополнительно указать `--plot-dir`, команда сохранит SVG с графиком query-паттерна и top-k совпадений.

Если указать `--only-past`, поиск будет брать только окна из прошлого относительно query.
`--gap-candles` задает дополнительный зазор в свечах для той же монеты, чтобы не подбирать почти соседние окна.
`--calendar-gap-candles` задает календарный зазор в свечах для всех монет, чтобы не подбирать окна из того же рыночного режима на других активах.
