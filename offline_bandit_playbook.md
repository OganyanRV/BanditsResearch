# Bandit/CatBoost experiment plan for logged ad data (pandas)

## Формат входных данных
Ожидается pandas-таблица (CSV/Parquet) с колонками:
- `policy`
- `reward`
- `puid`
- `features` — строка чисел, разделённых `"\t"`
- `show`
- `candidates` — строка id, разделённых `"\t"`
- `date`

Парсинг делается в `preprocess_bandit_dataframe(...)`:
- `candidates -> candidates_list: list[int]`
- `features -> features_list: list[float]`

## Разбиение train/test
`split_train_test_by_date(df, test_ratio=0.2)` — time-based split.

## Сценарии как параметры, а не хардкод
Вместо обязательного `run_five_scenarios` используется:
- `ScenarioConfig(name, pretrain_source, online_update)`
- `run_scenarios(train_df, test_df, policy_factories, scenarios, env_reward=None)`

То есть вы можете передавать любые наборы сценариев.

## Возвращаемые результаты
`run_scenarios(...)` возвращает dict из двух pandas-таблиц:
1. `metrics` — метрики по `(scenario, algo)`
2. `history` — история по шагам:
   - `avg_reward`
   - `cumulative_regret`
   - `avg_regret`

## Симуляция "как если бы знали все метки"
`make_simulated_environment(proba_predictor, stochastic=True)`
- обучаете модель среды (например CatBoost) для `P(click|x,a)`
- передаёте predictor в симулятор
- запускаете те же сценарии с `env_reward`

## CLI запуск
Скрипт: `src/run_benchmark.py`

Пример:
- `python src/run_benchmark.py --input data/events.csv --test-ratio 0.2`
- `python src/run_benchmark.py --input data/events.csv --simulate --stochastic-sim`

CLI сохраняет:
- `artifacts/metrics.csv`
- `artifacts/history.csv`
- графики по каждому сценарию в `artifacts/plots/*.png` (если есть matplotlib)
