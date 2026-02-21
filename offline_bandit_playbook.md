# Bandit experiment plan (polars + pandas metrics + tqdm)

## Формат входных данных
Ожидается таблица (TSV/Parquet) с колонками:
- `policy`
- `reward`
- `features` — строка чисел, разделённых `"\\t"` (возможны `null`)
- `show`
- `candidates` — строка id, разделённых `"\\t"`
- `date`

Парсинг делает `preprocess_bandit_dataframe(...)`:
- `candidates -> candidates_list: list[int]`
- `features -> features_list: list[float]`
- `null/none/nan` в features заменяется на `-1e-6`.

## Ключевые изменения
- CatBoost убран из раннера: только `epsilon_greedy`, `ucb`, `thompson_sampling`.
- Обработка данных на polars, а `metrics` и `history` возвращаются как pandas DataFrame.
- Статистика в политиках обновляется батчами: каждые ~10% шагов теста.
- `tqdm` обновляется не на каждом шаге, а каждые ~5% шагов, чтобы не засорять вывод.
- Live-графики во время проигрывания политик: `--live-plots --plot-every N`.

## Регрет
- Если есть `env_reward` (симуляция):
  - regret = best_reward_among_candidates - chosen_reward.
- Если среды нет:
  - regret считается как `max_a E[r|a] - E[r|a_chosen]`,
    где `E[r|a]` — эмпирическая оценка на train/pretrain.

## Сценарии параметрами
`run_scenarios(train_df, test_df, policy_factories, scenarios, ...)`, где `scenarios: list[ScenarioConfig]`.

## Возвращаемые результаты
`run_scenarios(...)` возвращает dict:
1. `metrics` — pandas DataFrame (по `(scenario, algo)`)
2. `history` — pandas DataFrame по шагам (`avg_reward`, `cumulative_regret`, `avg_regret`)

## CLI запуск
Скрипт: `src/run_benchmark.py`

Примеры:
- `python src/run_benchmark.py --input data/events.tsv --test-ratio 0.2`
- `python src/run_benchmark.py --input data/events.tsv --simulate --stochastic-sim`
- `python src/run_benchmark.py --input data/events.tsv --live-plots --plot-every 20`

Артефакты:
- `artifacts/metrics.csv`
- `artifacts/history.csv`
- `artifacts/plots/*.png` (если доступен matplotlib)
