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
- `null/none/nan` в features заменяется на `-1e-6`
- `propensity = 1 / num_candidates`

## Ключевые изменения
- CatBoost убран из раннера: только `epsilon_greedy`, `ucb`, `thompson_sampling`.
- Обработка данных на polars, а `metrics` и `history` возвращаются как pandas DataFrame.
- Статистика в политиках обновляется батчами: каждые ~10% шагов теста.
- `tqdm` обновляется не на каждом шаге, а каждые ~5% шагов.
- Убрана онлайн-отрисовка графиков во время rollout (callback удалён).

## Сэмплирование и фильтрация теста
После split:
- `train_df.sample(fraction=1.0, shuffle=True).sort("date")`
- `test_df.sample(fraction=1.0, shuffle=True).sort("date")`
- далее тест ограничивается `policy == random`.

## Регрет
- Если есть `env_reward` (симуляция):
  - regret = best_reward_among_candidates - chosen_reward.
- Если среды нет:
  - regret считается как `max_a E[r|a] - E[r|a_chosen]`,
    где `E[r|a]` — эмпирическая оценка на train/pretrain.

## Сценарии
- `default_five_scenarios()` — базовые 5 сценариев.
- `default_five_ips_scenarios()` — те же 5 сценариев, но с IPS-ориентированными именами и запуском IPS-оценивания в раннере.

## IPS-режим
При запуске с `--ips-scenarios` раннер считает reward как IPS-оценку:
- `ips_reward_t = I[a_t == show_t] * reward_t / propensity_t`
- итоговый CTR = среднее IPS-наград по тесту.

## Возвращаемые результаты
`run_scenarios(...)` возвращает dict:
1. `metrics` — pandas DataFrame (по `(scenario, algo)`)
2. `history` — pandas DataFrame по шагам (`avg_reward`, `cumulative_regret`, `avg_regret`)

## CLI запуск
Скрипт: `src/run_benchmark.py`

Примеры:
- `python src/run_benchmark.py --input data/events.tsv --test-ratio 0.2`
- `python src/run_benchmark.py --input data/events.tsv --ips-scenarios`
- `python src/run_benchmark.py --input data/events.tsv --simulate --stochastic-sim`

Артефакты:
- `artifacts/metrics.csv`
- `artifacts/history.csv`
- `artifacts/plots/*.png` (если доступен matplotlib)

## Ноутбук для запуска
`notebooks/run_benchmark_demo.ipynb` повторяет flow из `run_benchmark.py`.
