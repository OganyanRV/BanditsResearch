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
Регрет считается на каждом шаге по доступным действиям:
- один раз считаем `CTR(action)` на данных, где `policy == random`
- на шаге берём максимум только по доступным действиям `candidates`: `max_available_ctr_t = max_{a in candidates_t} CTR(a)`
- далее `regret_t = max_available_ctr_t - reward_t_fact`.

Где `reward_t_fact`:
- для replay — фактический наблюдаемый reward (только на матчах replay),
- для IPS-трека — фактический reward, если был матч, иначе 0.

## Сценарии
- `default_five_scenarios()` — базовые 5 сценариев.
- `default_five_ips_scenarios()` — те же 5 сценариев, но с IPS-ориентированными именами и запуском IPS-оценивания в раннере.

## IPS-статистики (считаются одновременно)
Раннер всегда считает две ветки метрик одновременно:
- replay-метрики (`reward`, `avg_reward`, `ctr`)
- IPS-метрики (`ips_reward`, `ips_avg_reward`, `ips_ctr`)

IPS-награда: `ips_reward_t = I[a_t == show_t] * reward_t / propensity_t`.
В replay-ветке шаг без совпадения действия и показа не добавляется в history.

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
