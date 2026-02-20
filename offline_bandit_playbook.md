# Bandit/CatBoost experiment plan for logged ad data (polars + tqdm)

## Формат входных данных
Ожидается таблица (CSV/Parquet) с колонками:
- `policy`
- `reward`
- `puid`
- `features` — строка чисел, разделённых `"\t"`
- `show`
- `candidates` — строка id, разделённых `"\t"`
- `date`

Парсинг делает `preprocess_bandit_dataframe(...)`:
- `candidates -> candidates_list: list[int]`
- `features -> features_list: list[float]`

## Что изменилось
- Основа переписана с pandas на **polars** для более быстрого I/O/обработки.
- Добавлен `tqdm`:
  - на уровне сценариев (`run_scenarios`)
  - внутри `evaluate_policy` по шагам оценки.
- Добавлен live-режим графиков по флагу:
  - `--live-plots`
  - `--plot-every N` — обновление графиков каждые N шагов.

## Сценарии параметрами
Вместо фиксированного раннера — `run_scenarios(train_df, test_df, policy_factories, scenarios, ...)`,
где `scenarios: list[ScenarioConfig]`.

## Возвращаемые результаты
`run_scenarios(...)` возвращает dict из polars-таблиц:
1. `metrics` — метрики по `(scenario, algo)`
2. `history` — история по шагам (`avg_reward`, `cumulative_regret`, `avg_regret`)

## Симуляция среды
`make_simulated_environment(proba_predictor, stochastic=True)` позволяет считать regret, когда доступна награда для любого действия.

## CLI запуск
Скрипт: `src/run_benchmark.py`

Примеры:
- `python src/run_benchmark.py --input data/events.csv --test-ratio 0.2`
- `python src/run_benchmark.py --input data/events.csv --simulate --stochastic-sim`
- `python src/run_benchmark.py --input data/events.csv --live-plots --plot-every 20`

Артефакты:
- `artifacts/metrics.csv`
- `artifacts/history.csv`
- `artifacts/plots/*.png` (если доступен matplotlib)
